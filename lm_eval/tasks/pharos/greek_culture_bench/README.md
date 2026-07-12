# Greek Culture Benchmark — lm-evaluation-harness Setup

This document describes **everything** we have configured for evaluating models on the
Greek Linguistic & Cultural Competency Benchmark ([ilsp/greek_culture_bench](https://huggingface.co/datasets/ilsp/greek_culture_bench)) inside
[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) (EleutherAI).

It explains the dataset, how grading works (substring inclusion + CNF logic, regular
expressions, partial credit), the prompts, the exact run commands, the metrics, current
limitations, and planned work.

> **Scope of this task folder.** Only the **five closed-ended formats** (`mc`, `bool`,
> `list`, `order`, `match`) are wired into lm-eval here. The dataset also contains an
> **open-ended** (`open`) format, which is **intentionally not shipped** in this folder,
> see [§5](#5-why-the-open-ended-format-is-not-included).

---

## Table of contents
- [Greek Culture Benchmark — lm-evaluation-harness Setup](#greek-culture-benchmark--lm-evaluation-harness-setup)
  - [Table of contents](#table-of-contents)
  - [1. The dataset and the `Verification` schema](#1-the-dataset-and-the-verification-schema)
    - [The CNF (Conjunctive Normal Form) verification object](#the-cnf-conjunctive-normal-form-verification-object)
  - [2. Grading philosophy](#2-grading-philosophy)
    - [The graders (functions in `utils.py`)](#the-graders-functions-in-utilspy)
  - [3. Metrics reference](#3-metrics-reference)
    - [3.1 `exact_match` (binary) — the **headline** metric](#31-exact_match-binary--the-headline-metric)
    - [3.2 `partial_credit` (graded, 0..1) — a secondary "how close" signal](#32-partial_credit-graded-01--a-secondary-how-close-signal)
  - [4. The lm-evaluation-harness setup](#4-the-lm-evaluation-harness-setup)
    - [Files (`lm_eval/tasks/pharos/greek_culture_bench/`)](#files-lm_evaltaskspharosgreek_culture_bench)
    - [Task YAML settings](#task-yaml-settings)
    - [Model serving](#model-serving)
    - [Run commands](#run-commands)
    - [Inspecting results](#inspecting-results)
  - [5. Why the open-ended format is not included](#5-why-the-open-ended-format-is-not-included)
  - [6. Prompts and system messages](#6-prompts-and-system-messages)
  - [7. Few-shot examples (planned, not yet implemented)](#7-few-shot-examples-planned-not-yet-implemented)
  - [8. Known model behaviours and limitations](#8-known-model-behaviours-and-limitations)
  - [9. Future work](#9-future-work)

---

## 1. The dataset and the `Verification` schema

- **Hub id:** `ilsp/greek_culture_bench` · **split:** `train` · **rows:** 1948 (public set).
- **Fields:** `ID`, `Category`, `Subcategory`, `Question`, `Verification`, `Url`,
  `Question_Type`, `File_Path`, `Language`.
- **Categories (6):** `vocabulary`, `grammar`, `art & entertainment`, `history`,
  `culture & tradition`, `geography`.
- **Question types (6):** `open` (408), `mc` (368), `bool` (314), `list` (306),
  `order` (277), `match` (275). The **five closed-ended types wired up here total 1540
  rows**. The 408 `open` rows are not evaluated by these tasks (see [§5](#5-why-the-open-ended-format-is-not-included)).
- **`Verification` is a JSON *string*** (not a nested object), e.g.
  `'{"include": [["Σωστό"]], "exclude": [["Λάθος"]]}'`. It must be parsed with
  `json.loads` before use.

### The CNF (Conjunctive Normal Form) verification object
Every question carries the ground truth as up to three keys:

```json
{
  "include": [["variantA1", "variantA2"], ["variantB1"]],
  "exclude": [["forbidden1", "forbidden2"]],
  "order":   [["1"], ["3"], ["2"]]
}
```

- **`include`** — a list of **clauses**. Logical **AND across clauses**, **OR within a
  clause**: the answer must contain *at least one* accepted variant of *every* clause.
  The inner lists enumerate acceptable surface forms (e.g. accented/unaccented spellings,
  inflected forms) because grading is literal (see §2).
- **`exclude`** — forbidden strings. If any appears, the answer is wrong.
- **`order`** — the required left-to-right sequence of items (used only by `order`).

---

## 2. Grading philosophy

Grading is **not** a neural or semantic method and it uses **no NLP libraries**
(no nltk/spaCy, no tokenization, no lemmatization). It combines three plain techniques:

1. **Substring inclusion (the core).** After lowercasing (accents preserved), the check is
   Python's `in` operator: an `include` variant matches if its characters appear
   consecutively inside the answer. Example: the accepted variant `Δρακων` is a substring
   of the model's `Δρακωντας`, so it matches. This is **lenient by design**, it tolerates
   extra words around the key phrase (e.g. `Η ψυχή της Παναγίας.` matches the accepted
   `ψυχή της`). The trade-off is possible false positives with very short accepted strings.
2. **Regular expressions**, used only for:
   - **`mc`** — extract the first *standalone* option label (a single digit or A–E/Α–Ε
     letter not glued to other letters/digits, so `2` inside `1823` or a Greek letter
     inside a word is ignored).
   - **`order`** — find each item's first-occurrence position with word boundaries.
3. **CNF logic** — the AND/OR structure described in §1.

Because Greek is morphologically rich, the grader cannot know that `Δράκων` and
`Δράκοντας` are the same lemma unless **both are listed** in the `include` clause. The
dataset handles this by **manually enumerating** accepted forms. Where a valid form is not
enumerated, a correct answer can be wrongly marked 0.

### The graders (functions in `utils.py`)
- `grade_include_exclude` — CNF substring check (`bool` / `list` / `match`).
- `grade_mc` — robust single-label extraction, compare to gold (script-insensitive: Greek
  `Β` == Latin `B`).
- `grade_order` — each item's first occurrence must be strictly ascending.
- `_strip_explanation` — cuts a trailing free-text explanation (cued by
  «Εξήγηση/Σημείωση/…») before grading `list`/`order`, so distractors mentioned in the
  explanation don't trip `exclude`.

---

## 3. Metrics reference

### 3.1 `exact_match` (binary) — the **headline** metric
Per question: **1.0** if the CNF rules pass, else **0.0**. The dataset-level number is the
mean (an accuracy). Reported for **all five** closed-ended formats.

> Note: the name "exact_match" is the metric *label*. The computation is our custom
> rule-based grader (substring inclusion + CNF + regex), **not** literal string equality.

### 3.2 `partial_credit` (graded, 0..1) — a secondary "how close" signal
Deterministic. Reported **only for `list`, `order`, `match`** (for `bool`/`mc` it is not
emitted, it would equal the binary score). Definitions:

- **`list` / `match`:** `max(0, (matched_include_clauses − violated_exclude_clauses) / total_include_clauses)`.
  Correct items add credit. A violated `exclude` clause subtracts one point (a mild
  penalty). *Example* (`art_and_entertainment_music_list_001`): 2 required names, both
  found (+2), plus one forbidden name present (−1), over 2 required ⇒ **(2−1)/2 = 0.5**,
  while `exact_match = 0`. Note: all distractors are usually grouped in **one** `exclude`
  clause, so including one or several forbidden items incurs the same single-point penalty.
- **`order`:** *position accuracy* — the fraction of items whose rank (by first occurrence
  in the answer) equals their rank in the target order. *Example:* target `1,2,3,4`,
  answer `1,3,2,4` ⇒ items 1 and 4 correctly placed ⇒ **0.5**. Target `1,3,2`, answer
  `1,2,3` ⇒ only item 1 correct ⇒ **0.33**.

Because `partial_credit` exists only for three of the five subtasks, the aggregate group
averages it over just those subtasks (size-weighted) and lm-eval prints a warning that it
is missing in `bool`/`mc`. **That warning is expected and harmless.**

---

## 4. The lm-evaluation-harness setup

### Files (`lm_eval/tasks/pharos/greek_culture_bench/`)
- `bool.yaml`, `mc.yaml`, `list.yaml`, `match.yaml`, `order.yaml` — one task per
  closed-ended `Question_Type`, named `greek_culture_bench_<type>`.
- `closed_aggregate.yaml` — the group task **`greek_culture_bench_closed_aggregate`**,
  which bundles the five tasks above and aggregates `exact_match` and `partial_credit`.
- `utils.py` — the grader (`process_results`), per-format dataset filters, and the
  partial-credit functions. (It also still contains **dormant** BERTScore helper code,
  intentionally kept for possible future use on a Linux/GPU box. It is never called.)

The group is registered in the Pharos suite via `../pharos_benchmarks.yaml`, so it also
runs as part of the umbrella `pharos_benchmarks` task.

### Task YAML settings
- `dataset_path: ilsp/greek_culture_bench`, `test_split: train`, `output_type: generate_until`.
- `doc_to_text: "{{Question}}"` (the question is sent as-is. No extra instructions appended).
- `doc_to_target` — **display only**. Scoring happens in `process_results`. It is
  `utils.doc_to_target_closed` for `bool`/`mc` and `"{{Verification}}"` for
  `list`/`order`/`match`.
- `process_docs` — `utils.process_<type>` filters the split down to that one
  `Question_Type` (and shuffles with `seed=42`).
- `metric_list`: `exact_match` for all five. **`list`/`order`/`match` also list
  `partial_credit`**.
- `generation_kwargs`: `temperature: 0.0`. `until: ["<|end_of_text|>"]` (with `"\n\n"`
  additionally for `bool`/`mc`/`match`). `max_gen_toks`: 16 (`bool`/`mc`),
  128 (`list`/`match`/`order`).

> The `<|end_of_text|>` stop string is the EOS token of Llama-Krikri, the model we
> developed these tasks against. It is harmless for other models (it simply never matches).

### Model serving
The model is served OpenAI-compatibly. We use
`--model local-chat-completions` (not `openai-chat-completions`, which sends an
incompatible `max_completion_tokens`). A `.env` provides `OPENAI_BASE_URL`,
`OPENAI_API_KEY`, and `HF_TOKEN` (the latter is needed to download the dataset).

### Run commands

Run the whole closed-ended benchmark (from a checkout of this repo, where the task is
auto-discovered):

```bash
lm_eval \
  --model local-chat-completions \
  --model_args "model= ,base_url= ,num_concurrent=4,eos_string=<|end_of_text|>,max_retries=10" \
  --apply_chat_template \
  --tasks greek_culture_bench_closed_aggregate \
  --log_samples \
  --output_path results/greek_culture_bench
```

- **Single format:** swap `greek_culture_bench_closed_aggregate` for
  `greek_culture_bench_mc`, `_bool`, `_list`, `_match` or `_order`.
- **Quick test:** add `--limit N`.
- **Smoke test with no model endpoint:** `--model dummy --limit 2` — this exercises task
  discovery, dataset download, filtering, prompting and the grader end-to-end (the scores
  are meaningless by construction).
- **Running from outside this repo:** add
  `--include_path <path>/lm_eval/tasks/pharos/greek_culture_bench`.

### Inspecting results
Per-sample outputs are written under `results/<run>/…/samples_*.jsonl` when `--log_samples`
is passed. The source project additionally ships a `view_results` notebook that discovers all runs and builds summary
tables (by format, by format×category, by category×subcategory) showing **Exact Match**
and, where available, **Partial Credit**.

---

## 5. Why the open-ended format is not included

The dataset's 408 `open` questions are **deliberately not wired into lm-eval here.** Free
text cannot be graded reliably by the rule-based approach described in §2:

- **`exact_match`** under-credits valid answers whose wording, synonyms or inflections are
  not enumerated in the `include` clauses.
- **`partial_credit`** does not help either, since an open answer is a single free-text
  span rather than a set of items to score fractionally.

Open-ended questions are therefore planned to be evaluated separately with an
**LLM-as-a-judge** setup, which can reason about semantic correctness instead of matching
substrings. Shipping them here under `exact_match` would report misleadingly low scores, so
only the five closed-ended formats are exposed in this folder.

---

## 6. Prompts and system messages

No system prompt is used: the model receives the raw `Question`, which itself contains the
formatting instructions (e.g. «Απάντησε σε μία λέξη»). `--apply_chat_template` wraps it
with the model's chat template.

---

## 7. Few-shot examples (planned, not yet implemented)

Few-shot is not currently used.

Purpose: improve **format compliance** (answer with only the term/label/numbers).

---

## 8. Known model behaviours and limitations

- **Markdown formatting (`**bold**`)** — Krikri emits Markdown on its own. We do not ask for
  it. It is harmless for grading (substring/label matching sees through the `**`).
- **Truncation** — answers can be cut when `max_gen_toks` is reached, usually
  mid-explanation. Harmless because the model answers first and we strip explanations before
  grading `list`/`order`. Raising the limit mostly wastes tokens on text we ignore.
- **Substring leniency** — very short accepted strings can cause false positives. Verbose
  answers can pass.
- **Morphology** — the rule grader needs enumerated forms. Unlisted valid inflections are
  wrongly failed.
- **BERTScore** — not used. Dormant code is kept in
  `utils.py` for a future GPU/Linux run.

---

## 9. Future work

- Wire few-shot demonstrations if format compliance needs it.
- Evaluate the open-ended set with an LLM-as-a-judge grader and compare it against the
  rule-based scores, to quantify how many valid answers the enumerated `include` lists miss.
- Re-enable BERTScore on a GPU/Linux machine for an additional soft metric.
- Evaluate additional models beyond `ilsp/Llama-Krikri-8B-Instruct-v1.5`.
