# scripts/generate_mmlux_all_yaml.py
# Generates YAML tasks for ALL languages (from your LANG_NORMALIZE),
# one YAML per subject (dataset config), and a per-language group alias.
#
# Requires: HF_DATASETS_TRUST_REMOTE_CODE=1  (see notes below)

import os
from string import Template
from collections import defaultdict
from datasets import get_dataset_config_names

DATASET = "Eurolingua/mmlux"

# Your language universe
LANG_NORMALIZE = {
    "en": "eng_Latn", "el": "ell_Grek", "grc": "grc_Grek", "sq": "sqi_Latn",
    "bg": "bul_Cyrl", "ca": "cat_Latn", "cs": "ces_Latn", "da": "dan_Latn",
    "de": "deu_Latn", "et": "est_Latn", "eu": "eus_Latn", "fi": "fin_Latn",
    "fr": "fra_Latn", "ga": "gle_Latn", "gl": "glg_Latn", "hr": "hrv_Latn",
    "hu": "hun_Latn", "hy": "hye_Armn", "is": "isl_Latn", "it": "ita_Latn",
    "lt": "lit_Latn", "lv": "lvs_Latn", "mk": "mkd_Cyrl", "mt": "mlt_Latn",
    "nl": "nld_Latn", "nn": "nno_Latn", "nb": "nob_Latn", "pl": "pol_Latn",
    "pt": "por_Latn", "ro": "ron_Latn", "sk": "slk_Latn", "sl": "slv_Latn",
    "es": "spa_Latn", "sr": "srp_Latn", "sr_cy": "srp_Cyrl", "sv": "swe_Latn",
    "tr": "tur_Latn", "uk": "ukr_Cyrl", "ar": "arb_Arab", "fa": "fas_Arab",
    "zh": "zho_Hans", "ja": "jpn_Jpan", "ru": "rus_Cyrl", "id": "ind_Latn",
    "vi": "vie_Latn", "la": "lat_Latn", "az": "azj_Latn", "bs": "bos_Latn",
    "ko": "kor_Hang", "th": "tha_Thai", "hi": "hin_Deva", "bn": "ben_Beng",
    "tl": "fil_Latn", "ka": "kat_Geor", "ur": "urd_Arab", "kk": "kaz_Cyrl",
    "ta": "tam_Taml", "te": "tel_Telu", "kn": "kan_Knda", "mr": "mar_Deva",
    "ne": "npi_Deva", "af": "afr_Latn", "ml": "mal_Mlym", "uz": "uzb_Cyrl",
    "uzn": "uzn_Cyrl", "tg": "tgk_Cyrl", "ky": "kir_Cyrl", "cy": "cym_Latn",
}

def lang_to_suffix(lang: str) -> str:
    # Most are uppercased ISO; Portuguese uses PT-PT; Serbian Cyrillic we map from sr_cy
    specials = {"pt": "PT-PT", "sr_cy": "CYRL"}  # 'srp_Cyrl' configs end with '_CYRL'
    return specials.get(lang, lang.upper())

YAML_TEMPLATE = Template(r"""task: $task_name
task_alias: MMLU-X $lang_upper - $subject
dataset_path: Eurolingua/mmlux
dataset_name: $dataset_name
output_type: multiple_choice
validation_split: test
doc_to_text: |
  {{question}}
  A. {{choices[0]}}
  B. {{choices[1]}}
  C. {{choices[2]}}
  D. {{choices[3]}}
  Answer:
doc_to_choice:
  - "A"
  - "B"
  - "C"
  - "D"
# Normalize gold label robustly:
#  - accepts 0..3, 1..4, or letters "A".."D"
#  - accepts string digits (e.g., "1")
doc_to_target: >
  {%- set L = ["A","B","C","D"] -%}
  {%- if answer is number -%}
    {%- if 0 <= answer < 4 -%}{{ L[answer] }}
    {%- elif 1 <= answer <= 4 -%}{{ L[answer-1] }}
    {%- else -%}A
    {%- endif -%}
  {%- elif answer in L -%}{{ answer }}
  {%- elif answer|string in ["0","1","2","3","4"] -%}
    {%- set i = (answer|string)|int -%}
    {%- if 0 <= i < 4 -%}{{ L[i] }}
    {%- elif 1 <= i <= 4 -%}{{ L[i-1] }}
    {%- else -%}A
    {%- endif -%}
  {%- else -%}A
  {%- endif -%}
""")

def main():
    print("Discovering dataset configs (subjects)…")
    cfgs = get_dataset_config_names(DATASET, trust_remote_code=True)
    # Group configs by suffix
    by_suffix = defaultdict(list)
    for c in cfgs:
        parts = c.rsplit("_", 1)
        if len(parts) == 2:
            subj, suf = parts
            by_suffix[suf].append((subj, c))

    total_yaml = 0
    for lang in LANG_NORMALIZE.keys():
        suf = lang_to_suffix(lang)
        if suf not in by_suffix:
            print(f"- {lang}: no configs with suffix _{suf} → skipping")
            continue

        # Output directory per language
        outdir = os.path.join("lm_eval", "tasks\\pharos\\mmlux", f"mmlux_{lang}")
        os.makedirs(outdir, exist_ok=True)

        task_names = []
        for subj, cfg in sorted(by_suffix[suf]):
            task_name = f"mmlux_{lang}_{subj.lower()}"
            yaml_path = os.path.join(outdir, f"{task_name}.yaml")
            with open(yaml_path, "w", encoding="utf-8") as f:
                f.write(
                    YAML_TEMPLATE.substitute(
                        task_name=task_name,
                        lang_upper=lang.upper(),
                        subject=subj,
                        dataset_name=cfg,
                    )
                )
            task_names.append(task_name)
            total_yaml += 1

        # Create a tiny group alias so you can run all subjects as one name:
        # lm_eval --model dummy --tasks mmlux_<lang>_all --limit 2
        init_py = os.path.join(outdir, "__init__.py")
        with open(init_py, "w", encoding="utf-8") as f:
            f.write(
                "TASKS = {\n"
                f'    "mmlux_{lang}_all": {repr(task_names)}\n'
                "}\n"
            )

        print(f"- {lang}: wrote {len(task_names)} tasks → {outdir} (group: mmlux_{lang}_all)")

    print(f"\nDone. Total YAML tasks written: {total_yaml}")

    print("\nNext steps:")
    print("  1) Ensure HF remote code is allowed (once per environment):")
    print("     Windows (PowerShell):  setx HF_DATASETS_TRUST_REMOTE_CODE 1; $env:HF_DATASETS_TRUST_REMOTE_CODE='1'")
    print("     Windows (CMD):         setx HF_DATASETS_TRUST_REMOTE_CODE 1 && set HF_DATASETS_TRUST_REMOTE_CODE=1")
    print("     WSL/Linux/macOS:       export HF_DATASETS_TRUST_REMOTE_CODE=1")
    print("  2) pip install -e .")
    print("  3) List tasks:  python -m lm_eval.list_tasks | grep mmlux_  (or: findstr /I mmlux_ on Windows)")
    print("  4) Run a smoke test (Greek, all subjects):")
    print("     lm_eval --model dummy --tasks mmlux_el_all --limit 2")
    print("  5) Or a single subject (e.g., anatomy):")
    print("     lm_eval --model dummy --tasks mmlux_el_anatomy --limit 5")

if __name__ == "__main__":
    main()
