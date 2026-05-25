# Greek History Evaluation Tasks

This directory contains evaluation tasks for Greek history benchmarks curated and integrated into the Pharos benchmark suite by ATHENA/ILSP.

## Tasks Included

### 1. **Greek History Chronological Order QA (`pharos_greek_history_co_qa`)**
* **Dataset**: [ilsp/greek-history-trapeza-thematon-co-qa](https://huggingface.co/datasets/ilsp/greek-history-trapeza-thematon-co-qa)
* **Description**: A dataset consisting of 237 questions containing randomly ordered lists of Greek historical events. The model's task is to output the correct chronological sequence of events starting from the oldest.
* **Evaluation Type**: Generative sequence task (`output_type: generate_until`).
* **Few-shot Configuration**: Default few-shot (n=5). The 5 items of the `dev` split are used as few-shot context examples, and the evaluation is performed on the `test` split (232 samples).

#### Parsing Strategy
To isolate the model's sequence response from any conversational padding or punctuation:
1. Predictions are lowercased and split by non-Greek alphabetical characters (using `[^α-ω]`).
2. Only single-letter tokens corresponding to the valid event keys (`α-ε`) are extracted in order of occurrence.
3. If no isolated tokens are found, a fallback mechanism searches for joined answer sequences of keys (e.g., `αδβεγ`).

#### Metrics
* **Exact Match (`exact_match`)**: A binary metric assessing whether the entire predicted sequence matches the ground truth sequence exactly.
* **Kendall's Tau Correlation (`kendalls_tau`)**: A rank correlation metric assessing the relative order of all event pairs. It awards partial credit to partially correct relative rankings, handling permutation swaps robustly without external dependencies.
