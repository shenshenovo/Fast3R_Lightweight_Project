from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def load_records(root: Path) -> list[dict]:
    records = []
    for path in root.rglob("latest_record.json"):
        try:
            records.append(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            continue
    return records


def write_csv(records: list[dict], out_csv: Path) -> None:
    if not records:
        out_csv.write_text("", encoding="utf-8")
        return
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        for row in records:
            writer.writerow(row)


def write_markdown(records: list[dict], out_md: Path) -> None:
    headers = [
        "config_name",
        "input_count",
        "success",
        "failure_stage",
        "total_time_sec",
        "peak_memory_gb",
        "confidence_mean",
        "confidence_median",
        "quality_observation",
    ]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for r in records:
        lines.append("| " + " | ".join(str(r.get(h, "")) for h in headers) + " |")
    out_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize C-only Fast3R evaluations")
    parser.add_argument("--root", required=True)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--out-md", required=True)
    args = parser.parse_args()

    root = Path(args.root)
    records = load_records(root)
    write_csv(records, Path(args.out_csv))
    write_markdown(records, Path(args.out_md))


if __name__ == "__main__":
    main()
