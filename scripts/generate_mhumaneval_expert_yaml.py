# scripts/generate_mhumaneval_expert_yaml.py
# Generates YAML tasks for the mHumanEval-Expert subset (Python only).

import os
from string import Template


BASE_URL = (
    "https://raw.githubusercontent.com/mraihan-gmu/mHumanEval-Benchmark/"
    "main/mHuamnEval-Expert/Human_Expert_{code}.json"
)

# Python is the only programming language we can execute in the harness today.
PROGRAMMING_LANGUAGE = "python"

# Human-translated natural languages in the Expert subset.
LANGUAGES = [
    ("arb_Arab", "Arabic"),
    ("ben_Beng", "Bengali"),
    ("fra_Latn", "French"),
    ("hin_Deva", "Hindi"),
    ("ita_Latn", "Italian"),
    ("jpn_Jpan", "Japanese"),
    ("kor_Hang", "Korean"),
    ("por_Latn", "Portuguese"),
    ("sin_Sinh", "Sinhala"),
    ("spa_Latn", "Spanish"),
    ("swh_Latn", "Swahili"),
    ("tel_Telu", "Telugu"),
    ("zho_Hans", "Chinese (Simplified)"),
    ("zul_Latn", "Zulu"),
]


YAML_TEMPLATE = Template(
    """task: $task_name
task_alias: mHumanEval-Expert ($language_name)
dataset_path: json
dataset_kwargs:
  data_files:
    train: $data_file
test_split: train
process_docs: !function ../utils.process_docs
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
        filter_fn: !function ../../../humaneval/utils.build_predictions
metadata:
  version: 1.0
"""
)


def main() -> None:
    outdir = os.path.join("lm_eval", "tasks", "pharos", "mhumaneval_expert", PROGRAMMING_LANGUAGE)
    os.makedirs(outdir, exist_ok=True)

    task_names = []
    for code, language_name in LANGUAGES:
        slug = code.lower()
        task_name = f"mhumaneval_expert_{PROGRAMMING_LANGUAGE}_{slug}"
        yaml_path = os.path.join(outdir, f"{task_name}.yaml")
        with open(yaml_path, "w", encoding="utf-8") as f:
            f.write(
                YAML_TEMPLATE.substitute(
                    task_name=task_name,
                    language_name=language_name,
                    data_file=BASE_URL.format(code=code),
                )
            )
        task_names.append(task_name)

    init_path = os.path.join(outdir, "__init__.py")
    with open(init_path, "w", encoding="utf-8") as f:
        f.write(
            "TASKS = {\n"
            f'    "mhumaneval_expert_{PROGRAMMING_LANGUAGE}_all": {repr(task_names)}\n'
            "}\n"
        )

    print(f"Wrote {len(task_names)} tasks → {outdir} (group: mhumaneval_expert_{PROGRAMMING_LANGUAGE}_all)")
    print("\nNext steps:")
    print("  1) export HF_DATASETS_TRUST_REMOTE_CODE=1  # datasets>=2.20.0")
    print("  2) export HF_ALLOW_CODE_EVAL=1  # code_eval executes generated code")
    print("  3) pip install -e .")
    print("  4) List tasks: python -m lm_eval.list_tasks | grep mhumaneval_expert")
    print(
        '  5) Run a smoke test: '
        'lm_eval --confirm_run_unsafe_code --model dummy '
        '--tasks mhumaneval_expert_python_spa_latn --limit 2'
    )


if __name__ == "__main__":
    main()
