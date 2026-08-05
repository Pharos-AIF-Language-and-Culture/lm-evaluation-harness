# Greek Culture Benchmark — lm-evaluation-harness Setup

This document describes **everything** we have configured for evaluating models on the
Greek Linguistic & Cultural Competency Benchmark ([ilsp/greek_culture_bench](https://huggingface.co/datasets/ilsp/greek_culture_bench)) inside
[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) (EleutherAI).

It explains the dataset, how grading works (substring inclusion + CNF logic, regular
expressions, partial credit), the three **system-prompt modes** (`full` / `persona` /
`none`), the exact run commands, the metrics, the results we obtained for
`ilsp/Llama-Krikri-8B-Instruct-v1.5`, current limitations, and planned work.

> **Scope of this task folder.** Only the **five closed-ended formats** (`mc`, `bool`,
> `list`, `order`, `match`) are wired into lm-eval here. The dataset also contains an
> **open-ended** (`open`) format, which is **intentionally not shipped** in this folder,
> see [§6](#6-why-the-open-ended-format-is-not-included).

---

## Table of contents
- [1. The dataset and the `Verification` schema](#1-the-dataset-and-the-verification-schema)
- [2. Grading philosophy](#2-grading-philosophy)
- [3. Metrics reference](#3-metrics-reference)
- [4. System-prompt modes (`full` / `persona` / `none`)](#4-system-prompt-modes-full--persona--none)
- [5. The lm-evaluation-harness setup](#5-the-lm-evaluation-harness-setup)
- [6. Why the open-ended format is not included](#6-why-the-open-ended-format-is-not-included)
- [7. Results (`ilsp/Llama-Krikri-8B-Instruct-v1.5`)](#7-results-ilspllama-krikri-8b-instruct-v15)
- [8. Known model behaviours and limitations](#8-known-model-behaviours-and-limitations)
- [9. Few-shot (planned) and future work](#9-few-shot-planned-and-future-work)

---

## 1. The dataset and the `Verification` schema

- **Hub id:** `ilsp/greek_culture_bench` · **split:** `train` · **rows:** **1,951** (public
  set). A separate held-out subset (~569 items) is kept private to avoid contamination and
  is not distributed.
- **Fields:** `ID`, `Category`, `Subcategory`, `Question`, `Verification`, `Url`,
  `Question_Type`, `File_Path`, `Language`.
- **Categories (6):** `art & entertainment`, `culture & tradition`, `geography`, `grammar`,
  `history`, `vocabulary`.
- **Question types (6):** `open` (408), `mc` (368), `bool` (318), `list` (305),
  `order` (277), `match` (275). The **five closed-ended types wired up here total 1,543
  rows**. The 408 `open` rows are not evaluated by these tasks (see [§6](#6-why-the-open-ended-format-is-not-included)).
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

## 4. System-prompt modes (`full` / `persona` / `none`)

Each task can prepend a system prompt to every question, selectable per run in **three
modes**. The prompts come from `prompts.yaml`, which ships **inside this task folder**:

| Mode | What the model receives | Purpose |
|---|---|---|
| **`none`** | The bare `Question` only (no system text). | Reference protocol — matches the "no system prompt, temperature 0" recommendation in the dataset card. This is the **default**. |
| **`persona`** | A **per-category persona** (e.g. *lexicographer* for `vocabulary`, *linguist-philologist* for `grammar`) + the `Question`. | Domain role-setting, without dictating answer format. |
| **`full`** | The persona **+ per-question-type formatting rules** + the `Question`. | Role-setting *and* strict output discipline (e.g. "answer with only the number", "only comma-separated numbers, no explanations"). |

`prompts.yaml` has two parts:

- **Per-category personas** (`<category>: { system_instruction: … }`) — keyed by the doc's
  `Category`. Subcategory personas are also present for possible finer-grained use.
- **`format_rules:`** — one block per question type (`bool` / `mc` / `list` / `match` /
  `order` / `default`), giving the terse output contract for that format.

So `persona` = persona only; `full` = persona + `format_rules[question_type]`; `none` = empty.

### Where the system prompt goes (an lm-eval detail)

lm-eval has **no per-document system role** (its `--system_instruction` flag is a single
*global* string, whereas our personas are **per-category**). We therefore **prepend** the
selected system text to the **user** message inside `doc_to_text` (the
`doc_to_text_dynamic` function). In `none` mode the message is exactly
`[{"role":"user","content": Question}]`; in `persona`/`full` it is
`[{"role":"user","content": persona (+ rules) + "\n\n" + Question}]`. This is why the
default (`none`) behaviour is byte-for-byte identical to sending the bare question.

---

## 5. The lm-evaluation-harness setup

### Files (`lm_eval/tasks/pharos/greek_culture_bench/`)
- `bool.yaml`, `mc.yaml`, `list.yaml`, `match.yaml`, `order.yaml` — one task per
  closed-ended `Question_Type`, named `greek_culture_bench_<type>`.
- `closed_aggregate.yaml` — the group task **`greek_culture_bench_closed_aggregate`**,
  which bundles the five tasks above and aggregates `exact_match` and `partial_credit`.
- `utils.py` — the grader (`process_results`), per-format dataset filters, the
  partial-credit functions, **and the system-prompt logic** (`doc_to_text_dynamic` +
  `_system_prompt`, which read the local `prompts.yaml`). It also still contains **dormant**
  BERTScore helper code, intentionally kept for possible future use on a Linux/GPU box. It
  is never called.
- `prompts.yaml` — the per-category personas + per-format rules used by the `persona` and
  `full` modes.

The group is registered in the Pharos suite via `../pharos_benchmarks.yaml`, so it also
runs as part of the umbrella `pharos_benchmarks` task.

### Task YAML settings
- `dataset_path: ilsp/greek_culture_bench`, `test_split: train`, `output_type: generate_until`.
- **`doc_to_text: !function utils.doc_to_text_dynamic`** — builds the prompt from the
  `SYSTEM_PROMPT_MODE` environment variable (see below). The function returns the bare
  question when the mode is `none`, so the default behaviour is unchanged.
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

### Selecting the system-prompt mode
The mode is chosen with the **`SYSTEM_PROMPT_MODE`** environment variable (default `none`),
read inside `doc_to_text_dynamic`. No YAML edit is needed to switch modes:

```
SYSTEM_PROMPT_MODE=none      -> bare Question (default)
SYSTEM_PROMPT_MODE=persona   -> per-category persona + Question
SYSTEM_PROMPT_MODE=full      -> persona + format rules + Question
```

### Model serving
The model is served OpenAI-compatibly (e.g. vLLM). We use
`--model local-chat-completions` (not `openai-chat-completions`, which sends an
incompatible `max_completion_tokens`). A `.env` provides `OPENAI_BASE_URL`,
`OPENAI_API_KEY`, and `HF_TOKEN` (the latter is needed to download the dataset).

### Run commands
Run the whole closed-ended benchmark (from a checkout of this repo, where the task is
auto-discovered). Prefix the command with `SYSTEM_PROMPT_MODE=<mode>`:

```bash
SYSTEM_PROMPT_MODE=full lm_eval \
  --model local-chat-completions \
  --model_args "model=<model-id>,base_url=<endpoint>/v1/chat/completions,num_concurrent=4,eos_string=<|end_of_text|>,max_retries=10" \
  --apply_chat_template \
  --tasks greek_culture_bench_closed_aggregate \
  --log_samples \
  --output_path results/full/greek_culture_bench
```

- **All three modes:** rerun with `SYSTEM_PROMPT_MODE=full|persona|none` and a matching
  `--output_path results/<mode>/…`. Omitting the variable defaults to `none`.
- **Single format:** swap `greek_culture_bench_closed_aggregate` for
  `greek_culture_bench_mc`, `_bool`, `_list`, `_match` or `_order`.
- **Quick test:** add `--limit N`.
- **Smoke test with no model endpoint:** `--model dummy --limit 2` — this exercises task
  discovery, dataset download, filtering, prompting and the grader end-to-end (the scores
  are meaningless by construction).
- **Running from outside this repo:** add
  `--include_path <path>/lm_eval/tasks/pharos/greek_culture_bench`.

### Inspecting results
Per-sample outputs are written under `results/<mode>/…/samples_*.jsonl` when `--log_samples`
is passed. The source project additionally ships a `view_results` notebook that discovers
all runs, adds a **`Mode`** column, and builds per-mode summary tables (by format, by
format×category, by category×subcategory) showing **Exact Match** and, where available,
**Partial Credit**.

---

## 6. Why the open-ended format is not included

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

## 7. Results (`ilsp/Llama-Krikri-8B-Instruct-v1.5`)

Runs cover the **1,543 closed-ended** items in all **three** system-prompt modes.
`exact_match` is the headline; `partial_credit` refines `list`/`order`/`match`.

**Overall — closed-ended only (size-weighted over the 5 formats):**

| Mode | Exact Match | Partial Credit (list/order/match) |
|---|---|---|
| **full** | **53.2%** | **56.0%** |
| none | 48.7% | 52.0% |
| persona | 47.2% | 50.4% |

> The Exact-Match overall above is a size-weighted mean of the five closed-format rows
> below (computed from the rounded per-format numbers, so it is approximate to ~0.1pp).
> The `partial_credit` overall is the value lm-eval aggregates over `list`/`order`/`match`.

**Exact Match — format × mode** (best mode per row in **bold**):

| Format | Samples | full | none | persona |
|---|---|---|---|---|
| bool | 318 | **68.9%** | 61.9% | 65.4% |
| mc | 368 | **74.2%** | 72.0% | 69.0% |
| list | 305 | **40.7%** | 40.0% | 34.4% |
| match | 275 | **46.2%** | 41.8% | 41.1% |
| order | 277 | **27.8%** | 18.8% | 17.3% |

**Partial Credit — format × mode** (`list` / `order` / `match` only):

| Format | Samples | full | none | persona |
|---|---|---|---|---|
| list | 305 | **61.2%** | 61.1% | 58.1% |
| match | 275 | **62.3%** | 57.6% | 56.9% |
| order | 277 | **44.0%** | 36.4% | 35.3% |

### Which mode is best?
- **`full` is the best mode overall and on every closed format.** The gains over `none` are
  largest exactly where **output discipline matters for a literal grader**: `order`
  **+9.0pp** (27.8 vs 18.8) and `bool` **+7.0pp** (68.9 vs 61.9). Partial credit agrees
  (`full` is best on all three of `list`/`match`/`order`).
- **`persona` ranks *below* `none` overall** (47.2% vs 48.7%). Because lm-eval cannot send a
  per-document system role, the persona is prepended into the **user** turn; without the
  disciplining format rules (that is `persona`, not `full`) the extra preamble slightly
  nudges the model toward longer, chattier answers, which the literal grader penalises. Add
  the format rules back (`full`) and the benefit of terse, well-formatted output dominates.

**Recommendation:** use **`full`** for the headline and for all closed formats.

---

## 8. Known model behaviours and limitations

- **Markdown formatting (`**bold**`)** — Krikri emits Markdown on its own; we do not ask for
  it. It is harmless for grading (substring/label matching sees through the `**`).
- **Truncation** — answers can be cut when `max_gen_toks` is reached, usually
  mid-explanation. Harmless because the model answers first and we strip explanations before
  grading `list`/`order`. Raising the limit mostly wastes tokens on text we ignore.
- **Substring leniency** — very short accepted strings can cause false positives; verbose
  answers can pass.
- **Morphology** — the rule grader needs enumerated forms; unlisted valid inflections are
  wrongly failed.
- **System-prompt placement** — because lm-eval cannot send a per-document system role,
  `persona`/`full` prompts are prepended to the user turn rather than delivered as a real
  system message. This is a faithful workaround, and it is why `persona` ranks below `none`
  here (§7); `none` and `full` are the most robust choices.
- **BERTScore** — not used. Dormant helper code is kept in `utils.py` for a future GPU/Linux run. BERTScore ≠ LLM-judge.

---

## 9. Few-shot (planned) and future work

Few-shot is not currently used. lm-eval supports it via `--num_fewshot N` (which needs a
clean `doc_to_target` first, since ours is display-only), or via a fixed set of
hand-authored demonstrations using `fewshot_config: samples:` in the task YAML. Purpose:
improve **format compliance** (answer with only the term/label/numbers). Krikri already
mostly complies thanks to the in-question instructions and the `full`-mode format rules, so
few-shot is optional.

**Other planned work:**
- Re-enable BERTScore on a GPU/Linux machine for an additional soft metric.
- Evaluate additional models beyond `ilsp/Llama-Krikri-8B-Instruct-v1.5`.
