from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from fast3r.dust3r.utils.image import load_images
from fast3r.dust3r.inference_multiview import inference
from fast3r.dust3r.viz_plotly import SceneViz
from fast3r.utils.checkpoint_utils import load_model


FALLBACK_FAILURE_STAGE = "unknown"


@dataclass
class EvalRecord:
    sample_name: str
    config_name: str
    input_count: int
    success: bool
    failure_stage: str
    total_time_sec: float | None
    peak_memory_gb: float | None
    loading_time_sec: float | None
    inference_time_sec: float | None
    viz_time_sec: float | None
    confidence_mean: float | None
    confidence_median: float | None
    quality_observation: str
    output_path: str


def _now_ms() -> int:
    return int(time.time() * 1000)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _safe_cuda_peak_gb() -> float | None:
    if not torch.cuda.is_available():
        return None
    return float(torch.cuda.max_memory_allocated() / (1024**3))


def _tensor_to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _collect_confidence(pred: dict[str, Any]) -> np.ndarray | None:
    for key in ("conf_local", "conf"):
        if key in pred:
            arr = _tensor_to_numpy(pred[key]).reshape(-1)
            if arr.size:
                return arr
    return None


def _prepare_images(image_dir: Path, size: int, max_images: int) -> tuple[list[dict[str, Any]], float]:
    start = time.time()
    image_paths = sorted(
        path
        for path in image_dir.rglob("*")
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    )
    if max_images > 0:
        image_paths = image_paths[:max_images]
    images = load_images([str(path) for path in image_paths], size=size, verbose=True)
    if max_images > 0:
        images = images[:max_images]
    for image, image_path in zip(images, image_paths):
        image.setdefault("dataset", image_dir.name or "images")
        image.setdefault("label", image_path.parent.name or image_dir.name or "images")
        image.setdefault("instance", image_path.stem)
    return images, time.time() - start


def _merge_output_dicts(items: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    merged_views: list[dict[str, Any]] = []
    merged_preds: list[dict[str, Any]] = []
    merged_profiling: dict[str, float] = {}

    for item in items:
        merged_views.extend(item["output_dict"].get("views", []))
        merged_preds.extend(item["output_dict"].get("preds", []))
        for key, value in (item.get("profiling_info") or {}).items():
            if isinstance(value, (int, float)):
                merged_profiling[key] = float(merged_profiling.get(key, 0.0)) + float(value)
            elif key not in merged_profiling:
                merged_profiling[key] = value

    return {"views": merged_views, "preds": merged_preds}, merged_profiling


def _build_viz(output: dict[str, Any], out_html: Path, downsample_stride: int) -> None:
    viz = SceneViz()
    preds = output["preds"]
    views = output["views"]

    for idx, (pred, view) in enumerate(zip(preds, views)):
        img = _tensor_to_numpy(view["img"].squeeze(0)).transpose(1, 2, 0)
        img = np.clip((img + 1.0) * 127.5, 0, 255).astype(np.uint8)
        pts_key = "pts3d_local_aligned_to_global" if "pts3d_local_aligned_to_global" in pred else "pts3d_in_other_view"
        conf_key = "conf_local" if "conf_local" in pred else "conf"
        pts = _tensor_to_numpy(pred[pts_key]).squeeze(0)
        conf = _tensor_to_numpy(pred[conf_key]).squeeze(0)
        mask = np.ones(pts.shape[:2], dtype=bool)
        if downsample_stride > 1:
            mask = np.zeros_like(mask)
            mask[::downsample_stride, ::downsample_stride] = True
        colors = img
        viz.add_pointcloud(pts, colors, mask=mask, point_size=2, view_idx=idx)

    viz.export_html(str(out_html))


def run_case(
    checkpoint_dir: str,
    image_dir: str,
    out_dir: str,
    config_name: str,
    input_count: int,
    image_size: int,
    max_images: int,
    chunk_size: int,
    no_viz: bool,
    save_metrics: bool,
    downsample_stride: int,
    is_lightning_checkpoint: bool,
    laf_k: int,
    dtype_name: str,
) -> EvalRecord:
    out_root = Path(out_dir)
    _ensure_dir(out_root)
    case_dir = out_root / f"{config_name}_{input_count}_{_now_ms()}"
    _ensure_dir(case_dir)

    status = {
        "sample_name": Path(image_dir).name,
        "config_name": config_name,
        "input_count": input_count,
        "success": False,
        "failure_stage": FALLBACK_FAILURE_STAGE,
        "total_time_sec": None,
        "peak_memory_gb": None,
        "loading_time_sec": None,
        "inference_time_sec": None,
        "viz_time_sec": None,
        "confidence_mean": None,
        "confidence_median": None,
        "quality_observation": "",
        "output_path": str(case_dir),
    }

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    total_start = time.time()
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        status["failure_stage"] = "loading"
        model, lit_module = load_model(checkpoint_dir, device=device, is_lightning_checkpoint=is_lightning_checkpoint)
        if hasattr(model, "set_laf_k"):
            model.set_laf_k(laf_k)
        if laf_k > 0 and hasattr(model, "encoder_chunk_size"):
            model.encoder_chunk_size = 20

        if dtype_name == "bf16":
            infer_dtype = torch.bfloat16
        elif dtype_name == "fp16":
            infer_dtype = torch.float16
        else:
            infer_dtype = torch.float32

        images, loading_time = _prepare_images(Path(image_dir), image_size, max_images)
        status["loading_time_sec"] = loading_time

        status["failure_stage"] = "inference"
        chunk_items: list[dict[str, Any]] = []
        if chunk_size > 0 and len(images) > chunk_size:
            chunk_start = 0
            while chunk_start < len(images):
                chunk = images[chunk_start : chunk_start + chunk_size]
                infer_start = time.time()
                with torch.no_grad():
                    output_dict, profiling_info = inference(
                        chunk,
                        model,
                        device=device,
                        dtype=infer_dtype,
                        verbose=False,
                        profiling=True,
                    )
                inference_time = time.time() - infer_start
                chunk_items.append(
                    {
                        "output_dict": output_dict,
                        "profiling_info": profiling_info,
                        "inference_time": inference_time,
                    }
                )
                chunk_start += chunk_size
            output_dict, profiling_info = _merge_output_dicts(chunk_items)
            inference_time = float(sum(item["inference_time"] for item in chunk_items))
        else:
            infer_start = time.time()
            with torch.no_grad():
                output_dict, profiling_info = inference(
                    images,
                    model,
                    device=device,
                    dtype=infer_dtype,
                    verbose=False,
                    profiling=True,
                )
            inference_time = time.time() - infer_start

        status["inference_time_sec"] = float(inference_time)

        status["failure_stage"] = "postprocess"
        if laf_k > 0 and output_dict.get("preds") and hasattr(lit_module, "align_local_pairwise"):
            try:
                lit_module.align_local_pairwise(output_dict["preds"], output_dict["views"])
                status["quality_observation"] = f"laf_k={laf_k}_pairwise_aligned"
            except Exception as align_exc:
                status["quality_observation"] = f"laf_align_skipped: {align_exc}"
        elif output_dict.get("preds") and hasattr(lit_module, "align_local_pts3d_to_global"):
            try:
                lit_module.align_local_pts3d_to_global(output_dict["preds"], output_dict["views"])
            except Exception as align_exc:
                status["quality_observation"] = f"align_skipped: {align_exc}"

        preds = output_dict.get("preds", [])
        conf_values: list[float] = []
        for pred in preds:
            conf = _collect_confidence(pred)
            if conf is not None:
                conf_values.extend(conf.astype(np.float64).tolist())
        if conf_values:
            status["confidence_mean"] = float(np.mean(conf_values))
            status["confidence_median"] = float(np.median(conf_values))

        if not no_viz:
            status["failure_stage"] = "visualization"
            viz_start = time.time()
            try:
                _build_viz(output_dict, case_dir / "scene_visualization.html", downsample_stride=downsample_stride)
                status["quality_observation"] = "viz_saved"
            except Exception as viz_exc:
                status["failure_stage"] = "visualization"
                status["quality_observation"] = f"visualization_failed: {viz_exc}"
            status["viz_time_sec"] = time.time() - viz_start
        else:
            status["quality_observation"] = "viz_disabled_metrics_only"

        if save_metrics:
            metrics = {
                "config_name": config_name,
                "input_count": input_count,
                "load_time_sec": loading_time,
                "inference_time_sec": status["inference_time_sec"],
                "confidence_mean": status["confidence_mean"],
                "confidence_median": status["confidence_median"],
                "profiling_info": profiling_info,
                "chunks": len(chunk_items) if chunk_items else 1,
                "laf_k": laf_k,
                "dtype": dtype_name,
            }
            (case_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

        status["success"] = True
        status["failure_stage"] = ""
        (case_dir / "status.json").write_text(json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception as exc:
        status["failure_stage"] = status["failure_stage"] if status["failure_stage"] != FALLBACK_FAILURE_STAGE else "loading"
        status["quality_observation"] = f"{type(exc).__name__}: {exc}"
        (case_dir / "error.txt").write_text(status["quality_observation"], encoding="utf-8")
        (case_dir / "status.json").write_text(json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8")
    finally:
        status["total_time_sec"] = time.time() - total_start
        status["peak_memory_gb"] = _safe_cuda_peak_gb()

    return EvalRecord(**status)


def main() -> None:
    parser = argparse.ArgumentParser(description="C-only Fast3R evaluation runner")
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--config-name", default="baseline")
    parser.add_argument("--input-count", type=int, default=100)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--max-images", type=int, default=100)
    parser.add_argument("--chunk-size", type=int, default=0)
    parser.add_argument("--no-viz", action="store_true")
    parser.add_argument("--save-metrics", action="store_true")
    parser.add_argument("--downsample-stride", type=int, default=4)
    parser.add_argument("--is-lightning-checkpoint", action="store_true")
    parser.add_argument("--laf-k", type=int, default=0, help="0 for dense alignment, >0 for B/LAF pairwise alignment")
    parser.add_argument("--dtype", choices=["fp32", "bf16", "fp16"], default="fp32")
    args = parser.parse_args()

    record = run_case(
        checkpoint_dir=args.checkpoint_dir,
        image_dir=args.image_dir,
        out_dir=args.out_dir,
        config_name=args.config_name,
        input_count=args.input_count,
        image_size=args.image_size,
        max_images=args.max_images,
        chunk_size=args.chunk_size,
        no_viz=args.no_viz,
        save_metrics=args.save_metrics,
        downsample_stride=args.downsample_stride,
        is_lightning_checkpoint=args.is_lightning_checkpoint,
        laf_k=args.laf_k,
        dtype_name=args.dtype,
    )

    out_root = Path(args.out_dir)
    _ensure_dir(out_root)
    with (out_root / "latest_record.json").open("w", encoding="utf-8") as f:
        json.dump(asdict(record), f, indent=2, ensure_ascii=False)
    with (out_root / "latest_record.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(record).keys()))
        writer.writeheader()
        writer.writerow(asdict(record))
    print(json.dumps(asdict(record), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
