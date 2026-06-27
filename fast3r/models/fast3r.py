# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from copy import deepcopy
import time
from typing import Optional, Sequence, Tuple, Union
from einops import rearrange
import huggingface_hub
from omegaconf import DictConfig, OmegaConf
import torch
import torch.distributed
import torch.nn as nn
import numpy as np
from fast3r.dust3r.datasets.base.base_stereo_view_dataset import view_name
from fast3r.dust3r.heads.postprocess import postprocess
from fast3r.dust3r.heads.dpt_head import PixelwiseTaskWithDPT
from fast3r.croco.models.blocks import Block, PositionGetter
from fast3r.croco.models.pos_embed import RoPE2D, get_1d_sincos_pos_embed_from_grid
from fast3r.models.components.llama import TransformerBlock, RMSNorm, precompute_freqs_cis
from packaging import version
from functools import partial

from fast3r.dust3r.patch_embed import get_patch_embed

from fast3r.dust3r.utils.misc import (
    freeze_all_params,
    transpose_to_landscape,
)
import torch.autograd.profiler as profiler

from fast3r.utils import pylogger

log = pylogger.RankedLogger(__name__, rank_zero_only=True)

hf_version_number = huggingface_hub.__version__
assert version.parse(hf_version_number) >= version.parse(
    "0.22.0"
), "Outdated huggingface_hub version, please reinstall requirements.txt"


class Fast3R(nn.Module,
             huggingface_hub.PyTorchModelHubMixin,
             repo_url="https://github.com/facebookresearch/fast3r",
             tags=["image-to-3d"]
             ):
    def __init__(
        self,
        encoder_args: dict,
        decoder_args: dict,
        head_args: dict,
        freeze="none",
        # --- Course project experiments ---
        # (A) Bottleneck: compress then restore to preserve interfaces.
        use_dim_reduction: Optional[bool] = None,
        dim_reduction_ratio: float = 0.5,
        dim_reduction_alpha: Optional[float] = None,
        dim_reduction_activation: str = "gelu",
        dim_reduction_verbose: Optional[bool] = None,
        # (B) Low-dim decoder: keep features low-dim for decoder/head to reduce compute.
        #     This is a more "real" lightweight mode but will generally require finetuning.
        use_lowdim_decoder: Optional[bool] = None,
        lowdim_ratio: float = 0.5,
        lowdim_verbose: Optional[bool] = None,
        # (C) Token subsampling: reduce #patch tokens (N) after encoder.
        #     This directly reduces attention cost (~N^2) but also lowers output resolution.
        use_token_subsample: Optional[bool] = None,
        token_subsample_stride: int = 2,
        token_subsample_verbose: Optional[bool] = None,
    ):
        super(Fast3R, self).__init__()

        self.encoder_args = OmegaConf.to_container(encoder_args) if isinstance(encoder_args, DictConfig) else encoder_args
        self.build_encoder(encoder_args)

        # Low-dim decoder flag must be decided before building decoder/head.
        self.use_lowdim_decoder = (
            self._parse_bool_env("FAST3R_USE_LOWDIM_DECODER", default=False)
            if use_lowdim_decoder is None
            else bool(use_lowdim_decoder)
        )
        # Allow env override for quick ablations in demo/inference scripts.
        self.lowdim_ratio = float(os.environ.get("FAST3R_LOWDIM_RATIO", str(lowdim_ratio)))
        self.lowdim_verbose = (
            self._parse_bool_env("FAST3R_LOWDIM_VERBOSE", default=True)
            if lowdim_verbose is None
            else bool(lowdim_verbose)
        )
        self._lowdim_logged_once = False
        self._lowdim_in_dim: Optional[int] = None
        self._lowdim_dim: Optional[int] = None
        self.lowdim_proj: Optional[nn.Linear] = None

        enc_dim = None
        if isinstance(self.encoder_args, dict):
            enc_dim = self.encoder_args.get("embed_dim", None)
        if enc_dim is None and hasattr(self.encoder, "model"):
            enc_dim = getattr(self.encoder.model, "embed_dim", None)
        if enc_dim is None:
            # Fallback: some configs store encoder dim in decoder args.
            enc_dim = decoder_args.get("enc_embed_dim", None) if isinstance(decoder_args, dict) else None
        if isinstance(enc_dim, int) and enc_dim > 0:
            self._init_lowdim_proj(enc_dim)

        self.use_token_subsample = (
            self._parse_bool_env("FAST3R_USE_TOKEN_SUBSAMPLE", default=False)
            if use_token_subsample is None
            else bool(use_token_subsample)
        )
        self.token_subsample_stride = int(os.environ.get("FAST3R_TOKEN_SUBSAMPLE_STRIDE", str(token_subsample_stride)))
        self.token_subsample_verbose = (
            self._parse_bool_env("FAST3R_TOKEN_SUBSAMPLE_VERBOSE", default=True)
            if token_subsample_verbose is None
            else bool(token_subsample_verbose)
        )
        self._token_subsample_logged_once = False

        self.decoder_args = OmegaConf.to_container(decoder_args) if isinstance(decoder_args, DictConfig) else decoder_args
        self.build_decoder(decoder_args)

        self.head_args = OmegaConf.to_container(head_args) if isinstance(head_args, DictConfig) else head_args
        self.build_head(head_args)

        self.max_parallel_views_for_head = 25  # how many views to process in parallel in the head, used to avoid OOM

        # ---------------------------------------------------------------------
        # Course project: lightweight backbone experiment (minimal-intrusion)
        #
        # Insert a bottleneck right AFTER the encoder output:
        #   encoder_feats -> Linear(D->D*r) -> activation -> Linear(D*r->D) -> decoder/fusion/head
        #
        # Rationale:
        # - This is the smallest change that preserves downstream interfaces (decoder input dim stays D).
        # - It enables an apples-to-apples comparison vs. the original model by flipping a single switch.
        # - It tests the engineering feasibility of "feature compression" without refactoring the pipeline.
        # ---------------------------------------------------------------------
        # Note: low-dim decoder and "compress-then-restore" bottleneck are mutually exclusive.
        if self.use_lowdim_decoder and (use_dim_reduction is True):
            raise ValueError("use_lowdim_decoder=True is incompatible with use_dim_reduction=True")
        if self.use_lowdim_decoder and use_dim_reduction is None:
            use_dim_reduction = False

        self.use_dim_reduction = (
            self._parse_bool_env("FAST3R_USE_DIM_REDUCTION", default=False)
            if use_dim_reduction is None
            else bool(use_dim_reduction)
        )
        self.dim_reduction_ratio = float(dim_reduction_ratio)
        self.dim_reduction_alpha = (
            float(os.environ.get("FAST3R_DIM_REDUCTION_ALPHA", "1.0"))
            if dim_reduction_alpha is None
            else float(dim_reduction_alpha)
        )
        self.dim_reduction_verbose = (
            self._parse_bool_env("FAST3R_DIM_REDUCTION_VERBOSE", default=True)
            if dim_reduction_verbose is None
            else bool(dim_reduction_verbose)
        )
        self._dim_reduction_status_logged_once = False
        self._dim_reduction_shapes_logged_once = False

        act = (dim_reduction_activation or "gelu").lower()
        if act == "relu":
            self.dim_reduction_act = nn.ReLU()
        elif act == "gelu":
            self.dim_reduction_act = nn.GELU()
        else:
            raise ValueError(f"Unsupported {dim_reduction_activation=}, expected 'relu' or 'gelu'")

        self._dim_reduction_in_dim: Optional[int] = None
        self._dim_reduction_hidden_dim: Optional[int] = None
        self.dim_reduction: Optional[nn.Linear] = None
        self.dim_restore: Optional[nn.Linear] = None

        # Best-effort eager init (helps training/optimizers); falls back to lazy init in forward.
        if self.use_dim_reduction:
            enc_dim = None
            if isinstance(self.encoder_args, dict):
                enc_dim = self.encoder_args.get("embed_dim", None)
            if enc_dim is None and hasattr(self.encoder, "model"):
                enc_dim = getattr(self.encoder.model, "embed_dim", None)
            if isinstance(enc_dim, int) and enc_dim > 0:
                self._init_dim_reduction_layers(enc_dim)

        self.set_freeze(freeze)

    @staticmethod
    def _parse_bool_env(name: str, default: bool) -> bool:
        val = os.environ.get(name, None)
        if val is None:
            return default
        return val.strip().lower() in {"1", "true", "yes", "y", "on"}

    def _init_lowdim_proj(self, in_dim: int, *, device=None, dtype=None) -> None:
        if in_dim <= 0:
            raise ValueError(f"Expected positive encoder feature dim, got {in_dim}")
        if not (0.0 < self.lowdim_ratio <= 1.0):
            raise ValueError(f"Expected 0 < lowdim_ratio <= 1, got {self.lowdim_ratio}")

        lowdim_dim = max(1, int(round(in_dim * self.lowdim_ratio)))
        if lowdim_dim > in_dim:
            lowdim_dim = in_dim

        self._lowdim_in_dim = int(in_dim)
        self._lowdim_dim = int(lowdim_dim)

        # Use a deterministic "channel truncation" projection to keep compatibility
        # with a simple weight truncation strategy when loading pretrained checkpoints.
        self.lowdim_proj = nn.Linear(in_dim, lowdim_dim, bias=False)
        with torch.no_grad():
            self.lowdim_proj.weight.zero_()
            eye = torch.eye(in_dim, dtype=self.lowdim_proj.weight.dtype, device=self.lowdim_proj.weight.device)
            self.lowdim_proj.weight.copy_(eye[:lowdim_dim])
        if device is not None or dtype is not None:
            self.lowdim_proj = self.lowdim_proj.to(device=device, dtype=dtype)

    @staticmethod
    def _choose_num_heads(target_dim: int, requested_heads: int) -> int:
        if target_dim % requested_heads == 0:
            return requested_heads
        # Pick a divisor close to requested_heads.
        for h in sorted({requested_heads, 16, 12, 8, 6, 4, 3, 2, 1}, reverse=True):
            if h <= requested_heads and target_dim % h == 0:
                return h
        return 1

    def _apply_lowdim_proj(self, encoded_feats: Union[Sequence[torch.Tensor], Tuple[torch.Tensor, ...]], *, profiling: bool = False):
        if not self.use_lowdim_decoder:
            return encoded_feats
        if len(encoded_feats) == 0:
            return encoded_feats
        ref = encoded_feats[0]
        if not isinstance(ref, torch.Tensor) or ref.ndim < 2:
            return encoded_feats

        in_dim = int(ref.shape[-1])
        if self.lowdim_proj is None or self._lowdim_in_dim != in_dim:
            self._init_lowdim_proj(in_dim, device=ref.device, dtype=ref.dtype)
        assert self.lowdim_proj is not None

        out_list = []
        projected_shape = None
        for idx, feat in enumerate(encoded_feats):
            projected = self.lowdim_proj(feat)
            out_list.append(projected)
            if idx == 0:
                projected_shape = tuple(projected.shape)

        if self.lowdim_verbose and (profiling or not self._lowdim_logged_once):
            if (not torch.distributed.is_initialized()) or torch.distributed.get_rank() == 0:
                print(
                    f"[lowdim-decoder] enabled={self.use_lowdim_decoder} "
                    f"encoder_feat={tuple(ref.shape)} -> lowdim_feat={projected_shape} "
                    f"(D={in_dim}, d={self._lowdim_dim}, ratio={self.lowdim_ratio})"
                )
            self._lowdim_logged_once = True

        return tuple(out_list) if isinstance(encoded_feats, tuple) else out_list

    def _apply_token_subsample(
        self,
        encoded_feats: Union[Sequence[torch.Tensor], Tuple[torch.Tensor, ...]],
        positions: Union[Sequence[torch.Tensor], Tuple[torch.Tensor, ...]],
        shapes: Union[Sequence[torch.Tensor], Tuple[torch.Tensor, ...]],
        *,
        profiling: bool = False,
    ):
        """
        Subsample patch tokens by taking a strided subset of the patch grid.
        This reduces N (and attention cost), but also reduces the effective resolution.
        """
        if not self.use_token_subsample:
            return encoded_feats, positions, shapes
        stride = int(self.token_subsample_stride)
        if stride <= 1:
            return encoded_feats, positions, shapes
        if len(encoded_feats) == 0:
            return encoded_feats, positions, shapes

        # Positions are (B, N, 2) integer patch coordinates.
        # We pick a regular subset where both coords are divisible by stride.
        out_feats, out_pos, out_shapes = [], [], []
        ref_shape_before = None
        ref_shape_after = None
        for feat, pos, tshape in zip(encoded_feats, positions, shapes):
            if not (isinstance(feat, torch.Tensor) and isinstance(pos, torch.Tensor) and isinstance(tshape, torch.Tensor)):
                out_feats.append(feat)
                out_pos.append(pos)
                out_shapes.append(tshape)
                continue
            if pos.ndim != 3 or pos.shape[-1] != 2 or feat.ndim != 3:
                out_feats.append(feat)
                out_pos.append(pos)
                out_shapes.append(tshape)
                continue

            # Assume all items in the batch share the same patch layout for this view.
            pos0 = pos[0]  # (N, 2)
            mask = (pos0[:, 0] % stride == 0) & (pos0[:, 1] % stride == 0)
            feat_s = feat[:, mask]
            pos_s = pos[:, mask]
            # Match downstream head reshaping to the compact sampled token grid.
            # Example: 224px with 16px patches gives 14 tokens per side; stride=4
            # keeps original coordinates 0,4,8,12, i.e. 4 compact rows/cols.
            patch = int(getattr(self, "patch_size", 16))
            pos0_s = pos0[mask]
            sampled_h = int(torch.unique(pos0_s[:, 0]).numel())
            sampled_w = int(torch.unique(pos0_s[:, 1]).numel())
            shape_s = tshape.new_tensor([sampled_h * patch, sampled_w * patch]).unsqueeze(0).repeat(tshape.shape[0], 1)

            out_feats.append(feat_s)
            out_pos.append(pos_s)
            out_shapes.append(shape_s)

            if ref_shape_before is None:
                ref_shape_before = tuple(feat.shape)
                ref_shape_after = tuple(feat_s.shape)

        if self.token_subsample_verbose and (profiling or not self._token_subsample_logged_once):
            if (not torch.distributed.is_initialized()) or torch.distributed.get_rank() == 0:
                print(
                    f"[token-subsample] enabled={self.use_token_subsample} stride={stride} "
                    f"feat={ref_shape_before} -> {ref_shape_after}"
                )
            self._token_subsample_logged_once = True

        return (
            tuple(out_feats) if isinstance(encoded_feats, tuple) else out_feats,
            tuple(out_pos) if isinstance(positions, tuple) else out_pos,
            tuple(out_shapes) if isinstance(shapes, tuple) else out_shapes,
        )

    def _init_dim_reduction_layers(self, in_dim: int, *, device=None, dtype=None) -> None:
        if in_dim <= 0:
            raise ValueError(f"Expected positive encoder feature dim, got {in_dim}")
        if not (0.0 < self.dim_reduction_ratio <= 1.0):
            raise ValueError(f"Expected 0 < dim_reduction_ratio <= 1, got {self.dim_reduction_ratio}")

        hidden_dim = max(1, int(round(in_dim * self.dim_reduction_ratio)))
        if hidden_dim > in_dim:
            hidden_dim = in_dim

        self._dim_reduction_in_dim = int(in_dim)
        self._dim_reduction_hidden_dim = int(hidden_dim)

        self.dim_reduction = nn.Linear(in_dim, hidden_dim, bias=True)
        self.dim_restore = nn.Linear(hidden_dim, in_dim, bias=True)
        # Initialize to a stable low-rank projector (instead of random mixing),
        # to avoid catastrophic degradation when running inference without finetuning.
        with torch.no_grad():
            nn.init.zeros_(self.dim_reduction.bias)
            nn.init.zeros_(self.dim_restore.bias)
            nn.init.orthogonal_(self.dim_reduction.weight)  # rows are orthonormal when hidden_dim <= in_dim
            self.dim_restore.weight.copy_(self.dim_reduction.weight.t())
        if device is not None or dtype is not None:
            self.dim_reduction = self.dim_reduction.to(device=device, dtype=dtype)
            self.dim_restore = self.dim_restore.to(device=device, dtype=dtype)

    def _apply_dim_reduction_bottleneck(
        self, encoded_feats: Union[Sequence[torch.Tensor], Tuple[torch.Tensor, ...]], *, profiling: bool = False
    ):
        if not self.use_dim_reduction:
            return encoded_feats
        if len(encoded_feats) == 0:
            return encoded_feats

        ref = encoded_feats[0]
        if not isinstance(ref, torch.Tensor) or ref.ndim < 2:
            return encoded_feats

        in_dim = int(ref.shape[-1])
        if self.dim_reduction is None or self.dim_restore is None or self._dim_reduction_in_dim != in_dim:
            self._init_dim_reduction_layers(in_dim, device=ref.device, dtype=ref.dtype)

        reduced_shape = None
        restored_shape = None
        out_list = []
        for idx, feat in enumerate(encoded_feats):
            reduced = self.dim_reduction(feat)
            reduced = self.dim_reduction_act(reduced)
            restored = self.dim_restore(reduced)
            if self.dim_reduction_alpha != 1.0:
                # AMP may produce different dtypes for `feat` (encoder output) and `restored` (linear output).
                # `torch.lerp` requires matching dtypes for start/end.
                if restored.dtype != feat.dtype:
                    restored = restored.to(dtype=feat.dtype)
                restored = torch.lerp(feat, restored, self.dim_reduction_alpha)
            out_list.append(restored)
            if idx == 0:
                reduced_shape = tuple(reduced.shape)
                restored_shape = tuple(restored.shape)

        if self.dim_reduction_verbose and (profiling or not self._dim_reduction_shapes_logged_once):
            # Keep logs concise: print shapes for a representative view only.
            if (not torch.distributed.is_initialized()) or torch.distributed.get_rank() == 0:
                print(
                    f"[dim-reduction] enabled={self.use_dim_reduction} "
                    f"encoder_feat={tuple(ref.shape)} -> reduced={reduced_shape} -> restored={restored_shape} "
                    f"(D={in_dim}, hidden={self._dim_reduction_hidden_dim}, ratio={self.dim_reduction_ratio}, alpha={self.dim_reduction_alpha})"
                )
            self._dim_reduction_shapes_logged_once = True

        return tuple(out_list) if isinstance(encoded_feats, tuple) else out_list

    def build_encoder(self, encoder_args: dict):
        # Initialize the encoder based on the encoder type
        if encoder_args["encoder_type"] == "croco":
            # Drop the encoder_type key
            encoder_args = deepcopy(encoder_args)
            encoder_args.pop("encoder_type")
            self.encoder = CroCoEncoder(**encoder_args)
        elif encoder_args["encoder_type"] == "dino_v2":
            # Drop the encoder_type key
            encoder_args = deepcopy(encoder_args)
            encoder_args.pop("encoder_type")
            self.encoder = DinoEncoder(**encoder_args)
        else:
            raise ValueError(f"Unsupported encoder type: {encoder_args['encoder_type']}")

    def build_decoder(self, decoder_args: dict):
        decoder_args["decoder_type"] = decoder_args.get('decoder_type', 'fast3r')  # default to fast3r if not specified
        if decoder_args["decoder_type"] == 'fast3r':
            decoder_args = deepcopy(decoder_args)
            decoder_args.pop('decoder_type')
            if self.use_lowdim_decoder:
                # Course project: keep the whole decoder/head width at low-dim for speed.
                # This changes parameter shapes; pretrained checkpoints are loaded via truncation.
                in_dim = self._lowdim_dim or decoder_args.get("enc_embed_dim", None) or self.encoder_args.get("embed_dim", None)
                if not isinstance(in_dim, int) or in_dim <= 0:
                    raise ValueError("Cannot infer low-dim width for decoder; please set encoder_args.embed_dim")
                decoder_args["enc_embed_dim"] = in_dim
                decoder_args["embed_dim"] = in_dim
                decoder_args["num_heads"] = self._choose_num_heads(in_dim, int(decoder_args.get("num_heads", 12)))
                # Keep the stored args consistent for head_factory/profiling/logging.
                if isinstance(self.decoder_args, dict):
                    self.decoder_args["enc_embed_dim"] = decoder_args["enc_embed_dim"]
                    self.decoder_args["embed_dim"] = decoder_args["embed_dim"]
                    self.decoder_args["num_heads"] = decoder_args["num_heads"]
            self.decoder = Fast3RDecoder(**decoder_args)
        elif decoder_args["decoder_type"] == 'llama':
            decoder_args = deepcopy(decoder_args)
            decoder_args.pop('decoder_type')
            self.decoder = LlamaDecoder(**decoder_args)
        else:
            raise ValueError(f"Unsupported decoder type: {decoder_args['decoder_type']}")

    def build_head(
        self,
        head_args: dict,
    ):
        self.output_mode = head_args['output_mode']
        self.head_type = head_args['head_type']
        self.depth_mode = head_args['depth_mode']
        self.conf_mode = head_args['conf_mode']

        # allocate primary downstream head
        self.downstream_head = self.head_factory(
            head_args['head_type'], head_args['output_mode'], has_conf=bool(head_args['conf_mode']), patch_size=head_args['patch_size']
        )

        # add the second head if with_local_head is True
        if head_args.get('with_local_head', False):
            self.downstream_head_local = self.head_factory(
                head_args['head_type'], head_args['output_mode'], has_conf=bool(head_args['conf_mode']), patch_size=head_args['patch_size']
            )
        else:
            self.downstream_head_local = None

        # magic wrapper
        self.head = transpose_to_landscape(
            self.downstream_head, activate=head_args['landscape_only']
        )

        if self.downstream_head_local:
            self.local_head = transpose_to_landscape(
                self.downstream_head_local, activate=head_args['landscape_only']
            )
        else:
            self.local_head = None

    def head_factory(self, head_type, output_mode, has_conf=False, patch_size=16):
        """ " build a prediction head for the decoder"""
        if head_type == "dpt" and output_mode == "pts3d":
            assert self.decoder_args["depth"] > 9
            l2 = self.decoder_args["depth"]
            feature_dim = 256
            last_dim = feature_dim // 2
            out_nchan = 3
            if self.use_lowdim_decoder and self._lowdim_dim is not None:
                ed = self._lowdim_dim
            else:
                ed = None
                if isinstance(self.encoder_args, dict):
                    ed = self.encoder_args.get("embed_dim", None)
                if ed is None and hasattr(self.encoder, "model"):
                    ed = getattr(self.encoder.model, "embed_dim", None)
                if not isinstance(ed, int):
                    raise ValueError("Cannot infer encoder embed dim for head; please set encoder_args.embed_dim")
            dd = self.decoder_args["embed_dim"]
            return PixelwiseTaskWithDPT(
                num_channels=out_nchan + has_conf,
                feature_dim=feature_dim,
                last_dim=last_dim,
                hooks_idx=[0, l2 * 2 // 4, l2 * 3 // 4, l2],
                dim_tokens=[ed, dd, dd, dd],
                postprocess=postprocess,
                depth_mode=self.head_args["depth_mode"],
                conf_mode=self.head_args["conf_mode"],
                head_type="regression",
                patch_size=patch_size,
            )
        else:
            raise NotImplementedError(f"unexpected {head_type=} and {output_mode=}")

    def load_state_dict(self, ckpt, **kw):
        # Backward/forward compatible loading:
        # - Old checkpoints won't have the course-project dim-reduction parameters.
        # - New checkpoints might include them, while older code/models might not.
        strict = kw.pop("strict", True)

        # If low-dim decoder is enabled, truncate compatible weights from a full-width checkpoint.
        if self.use_lowdim_decoder:
            ckpt = self._truncate_checkpoint_for_lowdim(ckpt)

        res = super().load_state_dict(ckpt, strict=False, **kw)
        if not strict:
            return res

        allowed_prefixes = ("dim_reduction.", "dim_restore.", "lowdim_proj.")
        missing = [k for k in res.missing_keys if not k.startswith(allowed_prefixes)]
        unexpected = [k for k in res.unexpected_keys if not k.startswith(allowed_prefixes)]
        if missing or unexpected:
            raise RuntimeError(
                "Error(s) in loading state_dict for Fast3R:\n"
                + (f"\tMissing key(s) in state_dict: {missing}\n" if missing else "")
                + (f"\tUnexpected key(s) in state_dict: {unexpected}\n" if unexpected else "")
            )
        return res

    def _truncate_checkpoint_for_lowdim(self, ckpt: dict) -> dict:
        """
        Truncate a full-width checkpoint to fit a low-dim decoder/head.
        This is a best-effort conversion for inference-only experimentation.
        """
        d = self._lowdim_dim
        if not isinstance(d, int) or d <= 0:
            return ckpt

        def _slice_1d(x: torch.Tensor, n: int) -> torch.Tensor:
            return x[:n].contiguous()

        def _slice_2d(x: torch.Tensor, n0: int, n1: int) -> torch.Tensor:
            return x[:n0, :n1].contiguous()

        def _slice_conv1x1_in(x: torch.Tensor, n_in: int) -> torch.Tensor:
            # (out, in, 1, 1)
            return x[:, :n_in, :, :].contiguous()

        new_ckpt = {}
        for k, v in ckpt.items():
            if not isinstance(v, torch.Tensor):
                new_ckpt[k] = v
                continue

            # Decoder
            if k == "decoder.decoder_embed.weight":
                new_ckpt[k] = _slice_2d(v, d, d)
                continue
            if k == "decoder.decoder_embed.bias":
                new_ckpt[k] = _slice_1d(v, d)
                continue
            if k.startswith("decoder.dec_blocks."):
                if k.endswith(("norm1.weight", "norm1.bias", "norm2.weight", "norm2.bias")):
                    new_ckpt[k] = _slice_1d(v, d)
                    continue
                if k.endswith("attn.qkv.weight"):
                    new_ckpt[k] = _slice_2d(v, 3 * d, d)
                    continue
                if k.endswith("attn.qkv.bias"):
                    new_ckpt[k] = _slice_1d(v, 3 * d)
                    continue
                if k.endswith("attn.proj.weight"):
                    new_ckpt[k] = _slice_2d(v, d, d)
                    continue
                if k.endswith("attn.proj.bias"):
                    new_ckpt[k] = _slice_1d(v, d)
                    continue
                if k.endswith("mlp.fc1.weight"):
                    hidden = self.decoder.dec_blocks[0].mlp.fc1.weight.shape[0]
                    new_ckpt[k] = _slice_2d(v, hidden, d)
                    continue
                if k.endswith("mlp.fc1.bias"):
                    hidden = self.decoder.dec_blocks[0].mlp.fc1.weight.shape[0]
                    new_ckpt[k] = _slice_1d(v, hidden)
                    continue
                if k.endswith("mlp.fc2.weight"):
                    hidden = self.decoder.dec_blocks[0].mlp.fc1.weight.shape[0]
                    new_ckpt[k] = _slice_2d(v, d, hidden)
                    continue
                if k.endswith("mlp.fc2.bias"):
                    new_ckpt[k] = _slice_1d(v, d)
                    continue
            if k == "decoder.dec_norm.weight" or k == "decoder.dec_norm.bias":
                new_ckpt[k] = _slice_1d(v, d)
                continue

            # DPT head: truncate 1x1 convs that take token channels as input
            if ".dpt.act_postprocess." in k and k.endswith(".0.weight") and v.ndim == 4 and v.shape[2:] == (1, 1):
                new_ckpt[k] = _slice_conv1x1_in(v, d)
                continue

            new_ckpt[k] = v

        return new_ckpt

    def load_from_dust3r_checkpoint(self, dust3r_checkpoint_path: str):
        """Load a Dust3R checkpoint into the model.
        Only load the patch_embed, enc_blocks, enc_norm, and downstream_head1 components from the checkpoint.

        Args:
            dust3r_checkpoint_path (str): Path to the Dust3R checkpoint.
        """
        # Load the checkpoint
        checkpoint = torch.load(dust3r_checkpoint_path, weights_only=False)['model']

        # Initialize state dictionaries for different components
        encoder_state_dict = {}
        downstream_head_state_dict = {}

        # Prepare to track loaded keys
        loaded_keys = set()

        # Split the checkpoint into encoder and downstream head
        for key, value in checkpoint.items():
            if key.startswith("patch_embed") or key.startswith("enc_blocks") or key.startswith("enc_norm"):
                if isinstance(self.encoder, CroCoEncoder):
                    new_key = key.replace("patch_embed", "encoder.patch_embed") \
                                 .replace("enc_blocks", "encoder.enc_blocks") \
                                 .replace("enc_norm", "encoder.enc_norm")
                    encoder_state_dict[new_key] = value
                    loaded_keys.add(key)  # Tentatively mark as loaded
            elif key.startswith("downstream_head1"):
                new_key = key.replace("downstream_head1", "downstream_head")
                downstream_head_state_dict[new_key] = value
                loaded_keys.add(key)  # Tentatively mark as loaded

        # Load the encoder part into the model if it is an instance of CroCoEncoder
        if isinstance(self.encoder, CroCoEncoder):
            load_result = self.load_state_dict(encoder_state_dict, strict=False)

            # Remove keys that failed to load
            missing_keys = set(load_result.missing_keys)
            unexpected_keys = set(load_result.unexpected_keys)
            loaded_keys -= (missing_keys | unexpected_keys)

        # Load the downstream head part into the model with try-catch logic
        # Save the original downstream head state to restore in case of failure
        downstream_head_original_state = {k: v.clone() for k, v in self.downstream_head.state_dict().items()}

        if not self.head_args.get('skip_load_pretrained_head', False):
            try:
                load_result = self.load_state_dict(downstream_head_state_dict, strict=False)

                # Remove keys that failed to load
                missing_keys = set(load_result.missing_keys)
                unexpected_keys = set(load_result.unexpected_keys)
                loaded_keys -= (missing_keys | unexpected_keys)
            except RuntimeError as e:
                log.warning(f"Error loading downstream head: {str(e)}")
                log.warning("Reverting downstream head to its original state")
                # Revert downstream head to its original state
                self.downstream_head.load_state_dict(downstream_head_original_state)

                del downstream_head_original_state

                # Remove downstream head keys from loaded_keys, as they were not loaded
                loaded_keys -= set([key for key in checkpoint.keys() if key.startswith("downstream_head1")])
        else:
            log.info("Skipping loading pretrained head")

        # Compute not loaded keys as difference between all checkpoint keys and loaded keys
        checkpoint_keys = set(checkpoint.keys())
        not_loaded_keys = checkpoint_keys - loaded_keys

        del checkpoint

        # Process keys to log only first-level names
        loaded_first_level_keys = {key.split('.')[0] for key in loaded_keys}
        not_loaded_first_level_keys = {key.split('.')[0] for key in not_loaded_keys}

        # Log unique first-level keys
        log.info(f"Loaded first-level keys: {sorted(loaded_first_level_keys)}")
        log.info(f"First-level keys not loaded: {sorted(not_loaded_first_level_keys)}")

    def set_freeze(self, freeze):  # this is for use by downstream models
        self.freeze = freeze
        to_be_frozen = {
            "none": [],
            "encoder": [self.encoder],
            "sandwich": [self.encoder, self.downstream_head],
        }
        freeze_all_params(to_be_frozen[freeze])

    def _encode_images(self, views, chunk_size=400):
        B = views[0]["img"].shape[0]

        # Check if all images have the same shape
        same_shape = all(view["img"].shape == views[0]["img"].shape for view in views)

        if same_shape:
            # Stack images along a new dimension to create a batch
            imgs = torch.cat([view["img"] for view in views], dim=0)  # Shape: [num_views * B, C, H, W]
            true_shapes = torch.cat(
                [view.get("true_shape", torch.tensor(view["img"].shape[-2:])[None].repeat(B, 1)) for view in views],
                dim=0
            )  # Shape: [num_views * B, 2]

            # Encode images in chunks to prevent OOM
            num_chunks = (imgs.shape[0] + chunk_size - 1) // chunk_size
            feats_chunks = []
            pos_chunks = []
            
            for i in range(num_chunks):
                start_idx = i * chunk_size
                end_idx = min((i + 1) * chunk_size, imgs.shape[0])
                chunk_feats, chunk_pos = self.encoder(imgs[start_idx:end_idx], true_shapes[start_idx:end_idx])
                feats_chunks.append(chunk_feats)
                pos_chunks.append(chunk_pos)
            
            feats = torch.cat(feats_chunks, dim=0)
            pos = torch.cat(pos_chunks, dim=0)

            # Split the encoded features and positions back into individual views
            encoded_feats = torch.split(feats, B, dim=0)
            positions = torch.split(pos, B, dim=0)
            shapes = torch.split(true_shapes, B, dim=0)
        else:
            # Process each image individually
            encoded_feats, positions, shapes = [], [], []
            for view in views:
                img = view["img"]
                true_shape = view.get(
                    "true_shape", torch.tensor(img.shape[-2:])[None].repeat(B, 1)
                )
                feat, pos = self.encoder(img, true_shape)
                encoded_feats.append(feat)
                positions.append(pos)
                shapes.append(true_shape)

        return encoded_feats, positions, shapes

    def set_max_parallel_views_for_head(self, max_parallel_views_for_head):
        # expose this to user to control the number of views processed in parallel in the head
        self.max_parallel_views_for_head = max_parallel_views_for_head

    def forward(self, views, profiling=False):
        """
        Args:
            views (list[dict]): a list of views, each view is a dict of tensors, the tensors are batched

        Returns:
            list[dict]: a list of results for each view
            dict: profiling information (if profiling=True)
        """
        # Initialize profiling dict
        profiling_info = {} if profiling else None
        
        # encode the images --> B,S,D
        encode_images_start_time = time.time()
        encoded_feats, positions, shapes = self._encode_images(views)
        encode_images_end_time = time.time()
        if profiling:
            torch.cuda.synchronize()
            encode_images_end_time = time.time()
            encode_time = encode_images_end_time - encode_images_start_time
            profiling_info["encode_images_time"] = encode_time
            print(f"encode_images time: {encode_time}")
        if encode_images_end_time - encode_images_start_time > 20:
            print(f"something is wrong with the encoder, it took: {encode_images_end_time - encode_images_start_time}")
            # print the image and true_shape
           # for view_idx, view in enumerate(views):
               # print(f"view_idx: {view_idx}\n, view name: {view_name(view)}\n, image content: {view['img']}\n, true_shape: {view['true_shape']}")

        if self.use_lowdim_decoder and self.lowdim_verbose and (profiling or not self._lowdim_logged_once):
            if (not torch.distributed.is_initialized()) or torch.distributed.get_rank() == 0:
                print(f"[lowdim-decoder] enabled={self.use_lowdim_decoder}")
        if self.dim_reduction_verbose and (profiling or not self._dim_reduction_status_logged_once):
            if (not torch.distributed.is_initialized()) or torch.distributed.get_rank() == 0:
                print(f"[dim-reduction] enabled={self.use_dim_reduction}")
            self._dim_reduction_status_logged_once = True

        # Course project experiment (B): keep features low-dim into decoder/head.
        encoded_feats = self._apply_lowdim_proj(encoded_feats, profiling=profiling)

        # Course project experiment (C): reduce #patch tokens after encoder.
        encoded_feats, positions, shapes = self._apply_token_subsample(
            encoded_feats, positions, shapes, profiling=profiling
        )

        # Course project experiment: bottleneck right after encoder output (before decoder).
        encoded_feats = self._apply_dim_reduction_bottleneck(encoded_feats, profiling=profiling)

        # Create image IDs for each patch
        pos_emb_start_time = time.time()
        num_images = len(views)
        B, _, _ = encoded_feats[0].shape

        different_resolution_across_views = not all(torch.equal(shapes[0], shape) for shape in shapes)

        # Initialize an empty list to collect image IDs for each patch.
        # Note that at inference time, different views may have different number of patches.
        image_ids = []

        # Loop through each encoded feature to get the actual number of patches
        for i, encoded_feat in enumerate(encoded_feats):
            num_patches = encoded_feat.shape[1]  # Get the number of patches for this image
            # Extend the image_ids list with the current image ID repeated num_patches times
            image_ids.extend([i] * num_patches)

        # Repeat the image_ids list B times and reshape it to match the expected shape
        image_ids = torch.tensor(image_ids * B).reshape(B, -1).to(encoded_feats[0].device)
        if profiling:
            pos_emb_time = time.time() - pos_emb_start_time
            profiling_info["pos_emb_time"] = pos_emb_time
            print(f"pos emb time: {pos_emb_time}")

        # combine all ref images into object-centric representation
        if profiling:
            torch.cuda.synchronize()
            decoder_start_time = time.time()
        dec_output = self.decoder(encoded_feats, positions, image_ids)
        if profiling:
            torch.cuda.synchronize()
            decoder_time = time.time() - decoder_start_time
            profiling_info["decoder_time"] = decoder_time
            print(f"decoder time: {decoder_time}")

        ################## Forward pass through the head ##################
        # TODO: optimize this

        # Initialize the final results list
        final_results = [{} for _ in range(num_images)]

        head_prepare_input_start_time = time.time()
        # Prepare the gathered outputs for each layer
        if different_resolution_across_views or self.training:
            # Precompute the number of patches per image
            num_patches_list = [encoded_feat.shape[1] for encoded_feat in encoded_feats]

            gathered_outputs_list = [[] for _ in range(num_images)]  # List per image
            for layer_output in dec_output:
                # layer_output: (B, P_total, D)
                # Split layer_output along dimension 1 according to num_patches_list
                split_layer_outputs = torch.split(layer_output, num_patches_list, dim=1)
                for img_id, gathered_output in enumerate(split_layer_outputs):
                    # gathered_output: (B, num_patches_list[img_id], D)
                    gathered_outputs_list[img_id].append(gathered_output)
        else:
            # All images have the same number of patches
            P_patches = encoded_feats[0].shape[1]
            gathered_outputs_list = []
            for layer_output in dec_output:
                # layer_output: (B, num_images * P_patches, D)
                # Rearrange to (num_images * B, P_patches, D)
                layer_output = rearrange(
                    layer_output,
                    'B (num_images P_patches) D -> (num_images B) P_patches D',
                    num_images=num_images,
                    P_patches=P_patches
                )
                gathered_outputs_list.append(layer_output)

        if profiling:
            head_prepare_input_time = time.time() - head_prepare_input_start_time
            profiling_info["head_prepare_input_time"] = head_prepare_input_time
            print(f"head prepare input time: {head_prepare_input_time}")
        
        head_forward_start_time = time.time()
        with profiler.record_function("head: forward pass"):
            if different_resolution_across_views or self.training:
                # If the views have different resolutions, we cannot batch the views together
                # or if we are in training mode, we can batch the views together, but we dont want to get OOM so we process them sequentially
                # Forward pass for each view separately
                final_results = [{} for _ in range(num_images)]
                for img_id in range(num_images):
                    img_result = self.head(gathered_outputs_list[img_id], shapes[img_id])
                    if self.local_head:
                        local_img_result = self.local_head(gathered_outputs_list[img_id], shapes[img_id])

                    # Re-map the results back to the original batch and image order
                    for key in img_result.keys():
                        if key == 'pts3d':
                            final_results[img_id]['pts3d_in_other_view'] = img_result[key]
                        else:
                            final_results[img_id][key] = img_result[key]

                    # Store local head output if available
                    if self.local_head:
                        final_results[img_id]['pts3d_local'] = local_img_result['pts3d']
                        if 'conf' in local_img_result:
                            final_results[img_id]['conf_local'] = local_img_result['conf']
            else:  # if we are in inference mode and all views have the same resolution, we can batch the views together
                concatenated_shapes = torch.cat(shapes, dim=0)

                # Split concatenated_shapes into chunks outside the loop
                shape_chunks = torch.split(concatenated_shapes, self.max_parallel_views_for_head, dim=0)
                num_chunks = len(shape_chunks)  # Determine number of chunks from shape_chunks

                # Initialize a list to hold chunked gathered outputs
                chunked_gathered_outputs_list = [[] for _ in range(num_chunks)]

                # Split gathered_outputs_list into chunks
                for layer_output in gathered_outputs_list:
                    # Split the layer_output along (num_images * B) dimension
                    split_layer_outputs = torch.split(layer_output, self.max_parallel_views_for_head, dim=0)
                    for chunk_idx, split_output in enumerate(split_layer_outputs):
                        chunked_gathered_outputs_list[chunk_idx].append(split_output)

                # Initialize lists to hold results for each chunk
                result_chunks = []
                local_result_chunks = [] if self.local_head else None

                # Process each chunk through self.head and local_head
                for chunk, chunk_shapes in zip(chunked_gathered_outputs_list, shape_chunks):
                    # Forward pass for self.head
                    result_chunk = self.head(chunk, chunk_shapes)
                    result_chunks.append(result_chunk)

                    # Forward pass for local head if available
                    if self.local_head:
                        local_result_chunk = self.local_head(chunk, chunk_shapes)
                        local_result_chunks.append(local_result_chunk)

                # Reassemble chunks
                result = {key: torch.cat([chunk[key] for chunk in result_chunks], dim=0) for key in result_chunks[0].keys()}

                if self.local_head:
                    local_result = {key: torch.cat([chunk[key] for chunk in local_result_chunks], dim=0) for key in local_result_chunks[0].keys()}

                #### Re-map the results from num_images * B tensor to list of B tensors
                # Initialize the final results list
                final_results = [{} for _ in range(num_images)]

                # Re-map the results back to the original batch and image order
                for key in result.keys():
                    for img_id in range(num_images):
                        img_result = result[key][img_id * B:(img_id + 1) * B]
                        if key == 'pts3d':
                            final_results[img_id]['pts3d_in_other_view'] = img_result
                        else:
                            final_results[img_id][key] = img_result

                        # Store local head output if available
                        if self.local_head:
                            local_img_result = local_result['pts3d'][img_id * B:(img_id + 1) * B]
                            final_results[img_id]['pts3d_local'] = local_img_result
                            if 'conf' in local_result:
                                final_results[img_id]['conf_local'] = local_result['conf'][img_id * B:(img_id + 1) * B]
        if profiling:
            torch.cuda.synchronize()
            end_time = time.time()
            profiling_info["head_forward_time"] = end_time - head_forward_start_time
            print(f"head forward time: {end_time - head_forward_start_time}")
            profiling_info["total_time"] = end_time - encode_images_start_time
            print(f"total Fast3R forward time: {end_time - encode_images_start_time}")

        if profiling:
            return final_results, profiling_info
        else:
            return final_results

class CroCoEncoder(nn.Module):
    def __init__(
        self,
        img_size=512,
        patch_size=16,
        patch_embed_cls="ManyAR_PatchEmbed",
        embed_dim=768,
        num_heads=12,
        depth=12,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        pos_embed="RoPE100",
        attn_implementation="pytorch_naive",
    ):
        super(CroCoEncoder, self).__init__()

        # patch embeddings  (with initialization done as in MAE)
        self.patch_embed_cls = patch_embed_cls
        self._set_patch_embed(img_size, patch_size, embed_dim)

        # Positional embedding
        self.pos_embed = pos_embed
        if pos_embed.startswith("RoPE"):  # eg RoPE100
            if RoPE2D is None:
                raise ImportError(
                    "Cannot find cuRoPE2D, please install it following the README instructions"
                )
            freq = float(pos_embed[len("RoPE") :])
            self.rope = RoPE2D(freq=freq)
        else:
            raise NotImplementedError("Unknown pos_embed " + pos_embed)

        # Transformer blocks
        self.enc_blocks = nn.ModuleList([
            Block(dim=embed_dim,
                  num_heads=num_heads,
                  mlp_ratio=mlp_ratio,
                  qkv_bias=True,
                  norm_layer=norm_layer,
                  rope=self.rope,
                  attn_implementation=attn_implementation)
            for _ in range(depth)
        ])
        self.enc_norm = norm_layer(embed_dim)

    def _set_patch_embed(self, img_size=224, patch_size=16, enc_embed_dim=768):
        self.patch_embed = get_patch_embed(
            self.patch_embed_cls, img_size, patch_size, enc_embed_dim
        )

    def forward(self, image, true_shape):
        # embed the image into patches  (x has size B x Npatches x C)
        x, pos = self.patch_embed(image, true_shape=true_shape)

        # Apply encoder blocks
        for blk in self.enc_blocks:
            x = blk(x, pos)

        # Apply final normalization
        x = self.enc_norm(x)
        return x, pos

class DinoEncoder(nn.Module):
    def __init__(
        self,
        patch_size=14,
        **kwargs
    ):
        super(DinoEncoder, self).__init__()
        # Load the pretrained DINOv2 model
        self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
        assert self.model.patch_size == patch_size == 14, "DINOv2 model must have patch size 14"
        self.patch_size = patch_size
        self.position_getter = PositionGetter()

    def forward(self, image, true_shape):
        # image shape: B x C x H x W
        B, C, H, W = image.shape

        # Split the batch into landscape and portrait based on true_shape
        landscape_mask = true_shape[:, 1] >= true_shape[:, 0]  # width >= height (landscape)
        portrait_mask = ~landscape_mask  # width < height (portrait)

        # Calculate the number of patches for the largest resolution in the batch
        true_height = true_shape[:, 0]  # Index 0 is height
        true_width = true_shape[:, 1]   # Index 1 is width
        num_patches_h = true_height // self.patch_size
        num_patches_w = true_width // self.patch_size
        num_patches = num_patches_h * num_patches_w  # Total number of patches

        # Pre-allocate tensors for the output
        encoded_feats = torch.empty((B, num_patches.max(), self.model.embed_dim), dtype=next(self.named_parameters())[1].dtype, device=image.device)
        encoded_pos = torch.empty((B, num_patches.max(), 2), dtype=torch.long, device=image.device)

        # If there are landscape images, process them
        if landscape_mask.any():
            landscape_images = image[landscape_mask]
            landscape_shapes = true_shape[landscape_mask]
            landscape_features, landscape_pos = self._process_images(landscape_images, landscape_shapes)
            encoded_feats[landscape_mask] = landscape_features
            encoded_pos[landscape_mask] = landscape_pos

        # If there are portrait images, process them
        if portrait_mask.any():
            portrait_images = image[portrait_mask]
            portrait_shapes = true_shape[portrait_mask]

            # Transpose the portrait images back to their original orientation
            portrait_images_transposed = portrait_images.transpose(2, 3)  # HxW -> WxH
            portrait_features, portrait_pos = self._process_images(portrait_images_transposed, portrait_shapes)

            # Unflatten the features, transpose back to match original batch order, then flatten again
            num_patches_h = portrait_shapes[:, 0] // self.patch_size  # Use true height
            num_patches_w = portrait_shapes[:, 1] // self.patch_size  # Use true width
            B_p, N, D = portrait_features.shape

            # Unflatten the features to (B, num_patches_h, num_patches_w, D)
            portrait_features_unflattened = portrait_features.view(B_p, num_patches_h[0], num_patches_w[0], D)

            # Transpose back (swap height and width)
            portrait_features_transposed = portrait_features_unflattened.transpose(1, 2)

            # Flatten again to match the expected shape
            portrait_features_flattened = portrait_features_transposed.flatten(1, 2)

            # Apply the same operation for positional embeddings (pos)
            B_p, N, _ = portrait_pos.shape  # Get the shape for pos
            portrait_pos_unflattened = portrait_pos.view(B_p, num_patches_h[0], num_patches_w[0], 2)
            portrait_pos_transposed = portrait_pos_unflattened.transpose(1, 2)
            portrait_pos_flattened = portrait_pos_transposed.flatten(1, 2)

            # Assign the processed features and positional embeddings back
            encoded_feats[portrait_mask] = portrait_features_flattened
            encoded_pos[portrait_mask] = portrait_pos_flattened

        return encoded_feats, encoded_pos

    def _process_images(self, images, true_shape):
        """
        Process a batch of images through the DINO encoder and compute positions.
        """
        # Forward pass through the DINO encoder to get encoded features
        features = self.model.forward_features(images)['x_norm_patchtokens']  # Shape: B x N_patches x D
        x = features  # Encoded features

        # Compute positions using PositionGetter
        true_height = true_shape[:, 0]  # Explicitly assign height
        true_width = true_shape[:, 1]   # Explicitly assign width
        num_patches_h = true_height // self.patch_size  # Height patches
        num_patches_w = true_width // self.patch_size  # Width patches
        pos = self.position_getter(images.shape[0], num_patches_h[0], num_patches_w[0], images.device)

        return x, pos


class Fast3RDecoder(nn.Module):
    def __init__(
        self,
        random_image_idx_embedding: bool,
        enc_embed_dim: int,
        embed_dim: int = 768,
        num_heads: int = 12,
        depth: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        attn_implementation: str = "pytorch_naive",
        attn_bias_for_inference_enabled=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
    ):
        super(Fast3RDecoder, self).__init__()

        # transfer from encoder to decoder
        self.decoder_embed = nn.Linear(enc_embed_dim, embed_dim, bias=True)

        self.dec_blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop,
                norm_layer=nn.LayerNorm,
                attn_implementation=attn_implementation,
                attn_bias_for_inference_enabled=attn_bias_for_inference_enabled
            ) for _ in range(depth)
        ])

        # initialize the positional embedding for the decoder
        self.random_image_idx_embedding = random_image_idx_embedding
        self.register_buffer(
            "image_idx_emb",
            torch.from_numpy(
                get_1d_sincos_pos_embed_from_grid(embed_dim, np.arange(1000))
            ).float(),
            persistent=False,
        )

        # final norm layer
        self.dec_norm = norm_layer(embed_dim)

    def _ensure_image_idx_emb(self, num_images: int, device: torch.device):
        if num_images <= self.image_idx_emb.shape[0]:
            self.image_idx_emb = self.image_idx_emb.to(device=device)
            return

        self.image_idx_emb = torch.from_numpy(
            get_1d_sincos_pos_embed_from_grid(
                self.image_idx_emb.shape[1], np.arange(num_images)
            )
        ).float().to(device=device)

    def _generate_per_rank_generator(self):
        # this way, the randperm will be different for each rank, but deterministic given a fixed number of forward passes (tracked by self.random_generator)
        # and to ensure determinism when resuming from a checkpoint, we only need to save self.random_generator to state_dict
        # generate a per-rank random seed
        per_forward_pass_seed = torch.randint(0, 2 ** 32, (1,)).item()
        world_rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        per_rank_seed = per_forward_pass_seed + world_rank

        # Set the seed for the random generator
        per_rank_generator = torch.Generator()
        per_rank_generator.manual_seed(per_rank_seed)
        return per_rank_generator

    def _get_random_image_pos(self, encoded_feats, batch_size, num_views, max_image_idx, device):
        """
        Generates non-repeating random image indices for each sample, retrieves corresponding
        positional embeddings for each view, and concatenates them.

        Args:
            encoded_feats (list of tensors): Encoded features for each view.
            batch_size (int): Number of samples in the batch.
            num_views (int): Number of views per sample.
            max_image_idx (int): Maximum image index for embedding.
            device (torch.device): Device to move data to.

        Returns:
            Tensor: Concatenated positional embeddings for the entire batch.
        """
        # Generate random non-repeating image IDs (on CPU)
        image_ids = torch.zeros(batch_size, num_views, dtype=torch.long)

        # First view is always 0 for all samples
        image_ids[:, 0] = 0

        # Get a generator that is unique to each rank, while also being deterministic based on the global across numbers of forward passes
        per_rank_generator = self._generate_per_rank_generator()

        # Generate random non-repeating IDs for the remaining views using the generator
        for b in range(batch_size):
            # Use the torch.Generator for randomness to ensure randomness between forward passes
            random_ids = torch.randperm(max_image_idx, generator=per_rank_generator)[:num_views - 1] + 1
            image_ids[b, 1:] = random_ids

        # Move the image IDs to the correct device
        image_ids = image_ids.to(device)

        # Initialize list to store positional embeddings for all views
        image_pos_list = []

        for i in range(num_views):
            # Retrieve the number of patches for this view
            num_patches = encoded_feats[i].shape[1]

            # Gather the positional embeddings for the entire batch based on the random image IDs
            image_pos_for_view = self.image_idx_emb[image_ids[:, i]]  # (B, D)

            # Expand the positional embeddings to match the number of patches
            image_pos_for_view = image_pos_for_view.unsqueeze(1).repeat(1, num_patches, 1)

            image_pos_list.append(image_pos_for_view)

        # Concatenate positional embeddings for all views along the patch dimension
        image_pos = torch.cat(image_pos_list, dim=1)  # (B, Npatches_total, D)

        return image_pos

    def forward(self, encoded_feats, positions, image_ids):
        """ Forward pass through the decoder.

        Args:
            encoded_feats (list of tensors): Encoded features for each view. Shape: B x Npatches x D
            positions (list of tensors): Positional embeddings for each view. Shape: B x Npatches x 2
            image_ids (tensor): Image IDs for each patch. Shape: B x Npatches
        """
        x = torch.cat(encoded_feats, dim=1)  # concate along the patch dimension
        pos = torch.cat(positions, dim=1)

        final_output = [x]  # before projection

        # project to decoder dim
        x = self.decoder_embed(x)

        # Add positional embedding based on image IDs
        if self.random_image_idx_embedding:
            self._ensure_image_idx_emb(len(encoded_feats), x.device)
            # Generate random positional embeddings for all views and samples
            image_pos = self._get_random_image_pos(encoded_feats=encoded_feats,
                                                   batch_size=encoded_feats[0].shape[0],
                                                   num_views=len(encoded_feats),
                                                   max_image_idx=self.image_idx_emb.shape[0] - 1,
                                                   device=x.device)
        else:
            # Use default image IDs from input
            num_images = int((torch.max(image_ids) + 1).cpu().item())
            self._ensure_image_idx_emb(num_images, x.device)
            image_idx_emb = self.image_idx_emb[:num_images]
            image_pos = image_idx_emb[image_ids]
        # Apply positional embedding based on image IDs and positions
        x += image_pos  # x has size B x Npatches x D, image_pos has size Npatches x D, so this is broadcasting

        for blk in self.dec_blocks:
            x = blk(x, pos)
            final_output.append(x)

        x = self.dec_norm(x)
        final_output[-1] = x

        return final_output

class LlamaDecoder(nn.Module):
    def __init__(
        self,
        random_image_idx_embedding: bool,
        enc_embed_dim: int,
        embed_dim: int = 4096,
        n_layers: int = 32,
        n_heads: int = 32,
        n_kv_heads: Optional[int] = None,
        multiple_of: int = 256,  # make SwiGLU hidden layer size multiple of large power of 2
        ffn_dim_multiplier: Optional[float] = None,
        norm_eps: float = 1e-5,
        rope_theta: float = 10000,
        max_seq_len: int = 1000,
        is_causal: bool = False,  # use bidirectional attention
        depth_init: bool = True,
        **kwargs
    ):
        super(LlamaDecoder, self).__init__()

        # assign the flags to attributes for later use
        self.random_image_idx_embedding = random_image_idx_embedding
        self.rope_theta = rope_theta

        # Compute head dimension
        self.head_dim = embed_dim // n_heads

        # Precompute freqs_cis
        self.precomputed_freqs_cis = self._precompute_freqs_cis(max_seq_len=max_seq_len)  # complex64, it is a tensor and not a parameter or buffer because otherwise DeepSpeed will convert it to float32

        # **Learnable embedding for view 0**
        self.view0_embed = nn.Parameter(torch.zeros(embed_dim))
        nn.init.normal_(self.view0_embed, mean=0.0, std=0.02)

        # Transfer from encoder to decoder dimensions
        self.decoder_embed = nn.Linear(enc_embed_dim, embed_dim, bias=True)

        # Initialize Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(layer_id=i, n_heads=n_heads, n_kv_heads=n_kv_heads, dim=embed_dim, multiple_of=multiple_of,
                             ffn_dim_multiplier=ffn_dim_multiplier, n_layers=n_layers, is_causal=is_causal, norm_eps=norm_eps, depth_init=depth_init)
            for i in range(n_layers)
        ])

        self.norm = RMSNorm(dim=embed_dim, eps=norm_eps)

    def _precompute_freqs_cis(self, max_seq_len) -> torch.Tensor:
        return precompute_freqs_cis(
            self.head_dim,
            # Need to compute until at least the max token limit for generation
            # (use 2x max sequence length to be safe)
            max_seq_len,
            self.rope_theta,
        )

    def _ensure_precomputed_freqs_cis(self, num_images: int, device: torch.device):
        if num_images <= self.precomputed_freqs_cis.shape[0]:
            self.precomputed_freqs_cis = self.precomputed_freqs_cis.to(device=device)
            return

        self.precomputed_freqs_cis = self._precompute_freqs_cis(
            max_seq_len=num_images
        ).to(device=device)

    def _generate_per_rank_generator(self):
        # Generate a per-rank random seed
        per_forward_pass_seed = torch.randint(0, 2 ** 32, (1,)).item()
        world_rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        per_rank_seed = per_forward_pass_seed + world_rank

        # Set the seed for the random generator
        per_rank_generator = torch.Generator()
        per_rank_generator.manual_seed(per_rank_seed)
        return per_rank_generator

    def _get_random_freqs_cis(self, encoded_feats, batch_size, num_views, max_image_idx, device):
        """
        Generates freqs_cis for each patch based on random image indices.

        Args:
            encoded_feats (list of tensors): Encoded features for each view.
            batch_size (int): Number of samples in the batch.
            num_views (int): Number of views per sample.
            max_image_idx (int): Maximum image index for embedding.
            device (torch.device): Device to move data to.

        Returns:
            torch.Tensor: freqs_cis of shape (batch_size, total_num_patches, head_dim)
        """
        # Generate random image IDs (on CPU)
        image_ids = torch.zeros(batch_size, num_views, dtype=torch.long)

        # First view is always 0 for all samples
        image_ids[:, 0] = 0

        # Get a generator that is unique to each rank
        per_rank_generator = self._generate_per_rank_generator()

        # Generate random IDs for the remaining views
        for b in range(batch_size):
            # Use the torch.Generator for randomness
            random_ids = torch.randperm(max_image_idx, generator=per_rank_generator)[:num_views - 1] + 1
            image_ids[b, 1:] = random_ids

        # Move the image IDs to the correct device
        image_ids = image_ids.to(device)
        self.precomputed_freqs_cis = self.precomputed_freqs_cis.to(device)

        # Initialize list to store positional embeddings for all views
        freqs_cis_list = []

        for i in range(num_views):
            # Retrieve the number of patches for this view
            num_patches = encoded_feats[i].shape[1]

            # Gather the positional embeddings for the entire batch based on the random image IDs
            freqs_cis_for_view = self.precomputed_freqs_cis[image_ids[:, i]]  # (B, D)

            # Expand the positional embeddings to match the number of patches
            freqs_cis_for_view = freqs_cis_for_view.unsqueeze(1).repeat(1, num_patches, 1)  # (B, Npatches, D)

            freqs_cis_list.append(freqs_cis_for_view)

        # Concatenate positional embeddings for all views along the patch dimension
        freqs_cis = torch.cat(freqs_cis_list, dim=1)  # (B, Npatches_total, D)

        return freqs_cis

    def forward(self, encoded_feats, positions, image_ids):
        x = torch.cat(encoded_feats, dim=1)  # Concatenate along the patch dimension
        pos = torch.cat(positions, dim=1)
        batch_size = x.shape[0]
        device = x.device

        x = self.decoder_embed(x)

        # Generate freqs_cis based on image_ids
        if self.random_image_idx_embedding:
            self._ensure_precomputed_freqs_cis(len(encoded_feats), device)
            freqs_cis = self._get_random_freqs_cis(
                encoded_feats=encoded_feats,
                batch_size=batch_size,
                num_views=len(encoded_feats),
                max_image_idx=self.precomputed_freqs_cis.shape[0] - 1,
                device=device
            )
        else:
            # Use image_ids to index into precomputed_freqs_cis
            num_images = int((torch.max(image_ids) + 1).cpu().item())
            self._ensure_precomputed_freqs_cis(num_images, device)
            image_idx_emb = self.precomputed_freqs_cis[:num_images]
            freqs_cis = image_idx_emb[image_ids]
        # Create a mask for view 0 patches
        view0_mask = (image_ids == 0).unsqueeze(-1).float()  # Shape: (batch_size, total_num_patches, 1)

        final_output = [x]

        for layer in self.layers:
            # Add the view0_embedding to the features of view 0 before each transformer layer
            x = x + view0_mask * self.view0_embed  # Broadcasts self.view0_embed over the last dimension

            x = layer(x, freqs_cis)
            final_output.append(x)

        x = self.norm(x)
        final_output[-1] = x

        return final_output
