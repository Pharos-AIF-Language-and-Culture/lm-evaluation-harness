#!/usr/bin/env python3
"""
Generate lm-eval-harness YAML tasks for BLEnD SAQ by bypassing the broken HF
'annotations' builder.

We instead load raw JSON files from:
  nayeon212/BLEnD/data/annotations_hf/*.json

This avoids the idks schema-casting failure shown on the dataset page. :contentReference[oaicite:2]{index=2}

Usage:
  python generate_blend.py --out_dir ../lm_eval/tasks/pharos/blend
"""

from pathlib import Path
import argparse
import yaml

HF_BASE = "https://huggingface.co/datasets/nayeon212/BLEnD/resolve/main/data/annotations_hf"

# Map country code -> (pretty name, filename in annotations_hf)
# Files visible in the repo tree. :contentReference[oaicite:3]{index=3}
REGIONS = {
    "DZ": ("Algeria", "Algeria_data.json"),
    "AS": ("Assam", "Assam_data.json"),
    "AZ": ("Azerbaijan", "Azerbaijan_data.json"),
    "CN": ("China", "China_data.json"),
    "ET": ("Ethiopia", "Ethiopia_data.json"),
    "GR": ("Greece", "Greece_data.json"),
    "ID": ("Indonesia", "Indonesia_data.json"),
    "IR": ("Iran", "Iran_data.json"),
    "MX": ("Mexico", "Mexico_data.json"),
    "KP": ("North Korea", "North_Korea_data.json"),
    "NG": ("Northern Nigeria", "Northern_Nigeria_data.json"),
    "KR": ("South Korea", "South_Korea_data.json"),
    "ES": ("Spain", "Spain_data.json"),
    "GB": ("United Kingdom", "UK_data.json"),
    "US": ("United States", "US_data.json"),
    "JB": ("West Java", "West_Java_data.json"),
}

def make_task_cfg(task_name: str, url: str, region_name: str):
    # English question prompt for consistency (dataset file includes en_question). :contentReference[oaicite:4]{index=4}
    doc_to_text = "Question: {{ en_question }}\nAnswer:"

    # Pick top-voted annotation's first English answer if available.
    doc_to_target = (
        "{{ "
        'annotations[0]["en_answers"][0] '
        'if annotations and annotations[0].get("en_answers") '
        "else '' "
        "}}"
    )

    return {
        "task": task_name,

        # Use generic JSON loader instead of the broken scripted config
        "dataset_path": "json",
        "dataset_kwargs": {
            "data_files": {
                "validation": url
            }
        },
        "validation_split": "validation",

        "output_type": "generate_until",
        "doc_to_text": doc_to_text,
        "doc_to_target": doc_to_target,
        "generation_kwargs": {"until": ["\n"]},

        # Keep it simple for smoke tests
        "metric_list": [
            {"metric": "exact_match", "aggregation": "mean", "higher_is_better": True}
        ],

        "metadata": {
            "version": "1.0",
            "description": f"BLEnD SAQ annotations (raw JSON) for {region_name}",
        },
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    task_names = []

    for code, (region_name, filename) in REGIONS.items():
        url = f"{HF_BASE}/{filename}"
        task_name = f"blend_saq_{code.lower()}"
        task_names.append(task_name)

        cfg = make_task_cfg(task_name, url, region_name)
        (out_dir / f"{task_name}.yaml").write_text(
            yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True),
            encoding="utf-8"
        )

    # Group file (IMPORTANT: remove/avoid any old broken _blend.yaml)
    group_cfg = {
        "group": "blend",
        "task": task_names,
        "metadata": {
            "version": "1.0",
            "description": "BLEnD short-answer group across 16 regions (raw JSON loader)",
        },
    }

    (out_dir / "blend.yaml").write_text(
        yaml.safe_dump(group_cfg, sort_keys=False, allow_unicode=True),
        encoding="utf-8"
    )

    # Optional
    init_path = out_dir / "__init__.py"
    if not init_path.exists():
        init_path.write_text("# BLEnD tasks\n", encoding="utf-8")

    print(f"Wrote {len(task_names)} tasks + blend group to: {out_dir}")
    print("Run with:")
    print(rf"  lm_eval --model dummy --tasks blend --limit 2 --include_path {out_dir.parent}")

if __name__ == "__main__":
    main()
