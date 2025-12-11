# scripts/generate_humaneval_xl_all_yaml.py
# Generates YAML tasks for HumanEval-XL across all natural-language splits
# (default: Python problems, which are the only ones the built-in code_eval
# metric in lm-eval currently supports). Set HF_DATASETS_TRUST_REMOTE_CODE=1.

import os
from string import Template

DATASET = "floatai/HumanEval-XL"
# Python is the only programming language with a working evaluation path in the
# harness today. Extend this list if you add execution support for other PLs.
PROGRAMMING_LANGUAGES = ["python"]
# HumanEval-XL natural language splits
NATURAL_LANGUAGE_SPLITS = [
    "English",
    "Russian",
    "Chinese",
    "German",
    "Spanish",
    "French",
    "Italian",
    "Portuguese",
    "Greek",
    "Hungarian",
    "Dutch",
    "Finnish",
    "Indonesian",
    "Turkish",
    "Arabic",
    "Vietnamese",
    "Bulgarian",
    "Persian",
    "Malay",
    "Hebrew",
    "Estonian",
    "Tagalog",
    "Afrikaans",
]


YAML_TEMPLATE = Template(
    """task: $task_name
task_alias: HumanEval-XL ($split_name)
dataset_path: $dataset
dataset_name: $dataset_name
test_split: $split_name
unsafe_code: true
output_type: generate_until
doc_to_text: "{{prompt}}"
doc_to_target: "{{test}}\\ncheck({{entry_point}})"
metric_list:
  - metric: !function ../../../humaneval/utils.pass_at_k
    aggregation: mean
    higher_is_better: true
    k: [1]
generation_kwargs:
  until:
    - "\\nclass"
    - "\\ndef"
    - "\\n#"
    - "\\nif"
    - "\\nprint"
  max_gen_toks: 1024
  do_sample: false
repeats: 1
num_fewshot: 0
filter_list:
  - name: "create_test"
    filter:
      - function: "custom"
        filter_fn: !function ../../humaneval/utils.build_predictions
        filter_fn: !function ../../../humaneval/utils.build_predictions
metadata:
  version: 1.0
"""
)


def main() -> None:
    total_yaml = 0
    for pl in PROGRAMMING_LANGUAGES:
        print(f"Generating static natural-language splits for {pl}…")
        splits = NATURAL_LANGUAGE_SPLITS
        outdir = os.path.join("lm_eval", "tasks", "pharos", "humaneval_xl", pl)
        os.makedirs(outdir, exist_ok=True)

        task_names = []
        for split in splits:
            task_name = f"humaneval_xl_{pl}_{split.lower()}"
            yaml_path = os.path.join(outdir, f"{task_name}.yaml")
            with open(yaml_path, "w", encoding="utf-8") as f:
                f.write(
                    YAML_TEMPLATE.substitute(
                        task_name=task_name,
                        split_name=split,
                        dataset=DATASET,
                        dataset_name=pl,
                    )
                )
            task_names.append(task_name)
            total_yaml += 1

        init_path = os.path.join(outdir, "__init__.py")
        with open(init_path, "w", encoding="utf-8") as f:
            f.write(
                "TASKS = {\n"
                f'    "humaneval_xl_{pl}_all": {repr(task_names)}\n'
                "}\n"
            )

        print(
            f"- {pl}: wrote {len(task_names)} tasks → {outdir} "
            f"(group: humaneval_xl_{pl}_all)"
        )

    print(f"\nDone. Total YAML tasks written: {total_yaml}")
    print("\nNext steps:")
    print("  1) export HF_DATASETS_TRUST_REMOTE_CODE=1  # datasets>=2.20.0")
    print("  2) export HF_ALLOW_CODE_EVAL=1  # code_eval executes generated code")
    print("  3) pip install -e .")
    print("  4) List tasks: python -m lm_eval.list_tasks | grep humaneval_xl_")
    print(
        '  5) Run a smoke test: '
        'lm_eval --confirm_run_unsafe_code --model dummy '
        '--tasks humaneval_xl_python_english --limit 2'
    )


if __name__ == "__main__":
    main()
