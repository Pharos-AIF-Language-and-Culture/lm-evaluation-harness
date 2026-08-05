"""
Grading utilities for the Greek Culture Benchmark inside lm-evaluation-harness.

Each per-format task YAML (bool/mc/list/match/order/open) points here:
  - process_<fmt>(dataset)  -> keeps only that question type
  - process_results(doc, results) -> the custom rule-based grader (the metric)
  - doc_to_target_closed(doc) -> the gold answer shown in the logs (display only)

Grading per format
------------------
  bool  : include/exclude substring (answer is the word «Σωστό»/«Λάθος»)
  list  : include/exclude substring (answer is a set of distinctive items)
  match : include substring of pair tokens (e.g. «Α-4»)
  open  : include/exclude substring  (+ a secondary, soft BERTScore-F1)
  mc    : robust single-label extraction (digit 1-9 or letter A-E / Α-Ε)
  order : robust label-position check (first occurrences must be ascending)

The `Verification` value is parsed as proper JSON (it is stored on the Hub via
json.dumps); a single-quote fallback is only used if strict JSON parsing fails,
so it never corrupts valid JSON that contains apostrophes.
"""

import json
import os
import re
import logging

import yaml

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# System prompt (full / persona / none)
# --------------------------------------------------------------------------- #
# lm-eval has no per-document *system* role and its --system_instruction flag is a
# single GLOBAL string, but our personas are PER-CATEGORY. So we select the prompt
# per doc here and prepend it to the user message (doc_to_text). The mode is chosen
# with an environment variable so no per-run YAML edit is needed:
#
#     SYSTEM_PROMPT_MODE=none      -> bare Question (default; matches the Hub protocol)
#     SYSTEM_PROMPT_MODE=persona   -> per-category persona + Question
#     SYSTEM_PROMPT_MODE=full      -> persona + per-Question_Type format rules + Question
#
# The prompts live in prompts.yaml next to this file (per-category personas + per
# Question_Type format rules); it is loaded lazily, only when the mode is not "none".

_PROMPTS_CACHE = None
_FORMAT_MARKER = "ΑΠΑΙΤΗΣΕΙΣ ΜΟΡΦΟΠΟΙΗΣΗΣ"  # legacy inline-rules marker; split it off for `persona`
_DEFAULT_SYS = (
    "Είσαι ένα εξειδικευμένο γλωσσικό μοντέλο για την ελληνική γλώσσα και τον πολιτισμό. "
    "Απάντησε στα Ελληνικά με ακρίβεια, ακολουθώντας τη μορφή που ζητά η κάθε ερώτηση."
)


def _load_prompts():
    """Load prompts.yaml (per-category personas + format rules) from this task's folder."""
    global _PROMPTS_CACHE
    if _PROMPTS_CACHE is None:
        here = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(here, "prompts.yaml")
        try:
            with open(path, "r", encoding="utf-8") as f:
                _PROMPTS_CACHE = yaml.safe_load(f) or {}
        except Exception as e:  # missing file / parse error -> fall back to bare persona
            logger.warning("Could not load prompts.yaml (%s); using default persona.", e)
            _PROMPTS_CACHE = {}
    return _PROMPTS_CACHE


def _system_prompt(doc):
    """Return the system-prompt text for the current SYSTEM_PROMPT_MODE, or '' for none."""
    mode = os.environ.get("SYSTEM_PROMPT_MODE", "none").lower().strip()
    if mode == "none":
        return ""
    prompts = _load_prompts()
    category = (doc.get("Category") or "general")
    raw = prompts.get(category, {}).get("system_instruction", _DEFAULT_SYS)
    persona = str(raw).split(_FORMAT_MARKER)[0].strip()
    if mode == "persona":
        return persona
    # full: persona + the formatting rules for THIS question's Question_Type
    qtype = str(doc.get("Question_Type", "")).lower().strip()
    fmt = prompts.get("format_rules", {}) or {}
    rule = str(fmt.get(qtype) or fmt.get("default", "")).strip()
    return (persona + "\n\n" + rule) if rule else persona


def doc_to_text_dynamic(doc):
    """Prompt text = optional system prompt (per SYSTEM_PROMPT_MODE) prepended to the
    Question. With SYSTEM_PROMPT_MODE=none this is just the Question (the default)."""
    question = str(doc.get("Question", ""))
    sys = _system_prompt(doc)
    return f"{sys}\n\n{question}" if sys else question


# --------------------------------------------------------------------------- #
# BERTScore (lazy import so closed-only runs don't pay the startup cost)
# --------------------------------------------------------------------------- #
_bert_score_loaded = False
_bert_score_func = None


def get_bert_score():
    global _bert_score_loaded, _bert_score_func
    if not _bert_score_loaded:
        try:
            from bert_score import score
            _bert_score_func = score
        except Exception:
            logger.warning("bert_score unavailable; bertscore_f1 will be 0.0.")
            _bert_score_func = None
        _bert_score_loaded = True
    return _bert_score_func


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
# Any Greek (incl. accented/polytonic) or Latin letter or digit. Used as the
# "is this character part of a word/number?" boundary class.
_LET = r"0-9A-Za-zͰ-Ͽἀ-῿"

# Map option letters (Latin and Greek, both scripts) to a position label so that
# "B" and "Β" (and "C"/"Γ", "D"/"Δ", ...) are treated as the same choice.
_LETTER_ORDER = {
    "A": "L1", "Α": "L1",
    "B": "L2", "Β": "L2",
    "C": "L3", "Γ": "L3",
    "D": "L4", "Δ": "L4",
    "E": "L5", "Ε": "L5",
}

# A standalone multiple-choice label token: a single digit, or a single
# A-E / Α-Ε letter, not glued to another letter/digit (so it is not part of a
# word like «Δάφνη» or a number like «1823»).
_MC_TOKEN = re.compile(
    r"(?<![" + _LET + r"])([0-9]|[A-EΑ-Εa-eα-ε])(?![" + _LET + r"])"
)


def parse_rules(verification):
    """Return the Verification dict. Accepts a dict or a JSON string."""
    if isinstance(verification, dict):
        return verification
    s = str(verification)
    try:
        return json.loads(s)                 # normal path: proper JSON
    except Exception:
        try:
            return json.loads(s.replace("'", '"'))  # fallback for single-quoted strings
        except Exception:
            logger.warning("Could not parse Verification: %.80r", s)
            return {}


def _norm(s):
    """Case-folding only. Accents/spelling are preserved on purpose."""
    return str(s).lower()


def _variants(clause):
    return clause if isinstance(clause, (list, tuple)) else [clause]


def _canon_label(label):
    lab = str(label).strip().upper()
    return _LETTER_ORDER.get(lab, lab)      # letters -> L1..L5; digits stay as-is


def _first_label_pos(text, label):
    """Position of the first *standalone* occurrence of `label`, else None."""
    label = str(label).strip()
    if not label:
        return None
    if label.isdigit():
        pat = r"(?<!\d)" + re.escape(label) + r"(?!\d)"
    elif len(label) == 1:
        pat = r"(?<![" + _LET + r"])" + re.escape(label) + r"(?![" + _LET + r"])"
    else:
        pat = re.escape(label)              # multi-char: plain (case-insensitive)
    m = re.search(pat, text, flags=re.IGNORECASE)
    return m.start() if m else None


# Cues that mark the start of a free-text explanation the model appends AFTER its
# answer. We cut the answer here so that distractors mentioned in the explanation
# ("...το X δεν ανήκει...") do not trip the `exclude` check. We deliberately do NOT
# cut on a blank line, because some answers put a preamble first and the items after.
_EXPL_CUE = re.compile(r"εξήγηση|επεξήγηση|σημείωση|διευκρίν|\n\s*\(", re.IGNORECASE)


def _strip_explanation(text):
    m = _EXPL_CUE.search(str(text))
    return text[:m.start()] if m else text


# --------------------------------------------------------------------------- #
# Graders
# --------------------------------------------------------------------------- #
def grade_include_exclude(completion, rules):
    """CNF: every include clause must match (AND of clauses, OR within a clause);
    no exclude term may appear. Used for bool / list / match / open."""
    text = _norm(completion)
    for clause in rules.get("include", []) or []:
        if not any(_norm(v) in text for v in _variants(clause)):
            return False
    for clause in rules.get("exclude", []) or []:
        for v in _variants(clause):
            if _norm(v) and _norm(v) in text:
                return False
    return True


def grade_mc(completion, rules):
    """Single-choice: find the first option label the model actually picks and
    compare it to the gold label (script-insensitive). Avoids the substring
    pitfalls of letters inside Greek words and digits inside years."""
    include = rules.get("include", []) or []
    exclude = rules.get("exclude", []) or []
    gold = {_canon_label(v) for clause in include for v in _variants(clause)}
    valid = set(gold) | {_canon_label(v) for clause in exclude for v in _variants(clause)}
    if not gold:
        return grade_include_exclude(completion, rules)
    for m in _MC_TOKEN.finditer(completion):
        tok = _canon_label(m.group(1))
        if tok in valid:                     # first real option label the model states
            return tok in gold
    # Model gave no recognizable label -> fall back to substring CNF.
    return grade_include_exclude(completion, rules)


def grade_order(completion, rules):
    """Sorting: each item's first occurrence must appear in the required order."""
    seq = rules.get("order")
    if not seq:                              # items mistakenly stored under include
        return grade_include_exclude(completion, rules)
    positions = []
    for clause in seq:
        best = None
        for v in _variants(clause):
            p = _first_label_pos(completion, v)
            if p is not None and (best is None or p < best):
                best = p
        if best is None:
            return False                     # a required item is missing
        positions.append(best)
    return all(positions[i] < positions[i + 1] for i in range(len(positions) - 1))


# --------------------------------------------------------------------------- #
# Partial-credit graders (return a float in [0, 1]) — a secondary "how close"
# signal for list / match / order. Binary exact_match stays the headline metric.
# --------------------------------------------------------------------------- #
def partial_list_match(completion, rules):
    """Fraction of required include-clauses found, minus one point per violated
    exclude-clause, floored at 0. Used for list and match."""
    include = rules.get("include", []) or []
    total = len(include)
    if total == 0:
        return 1.0 if grade_include_exclude(completion, rules) else 0.0
    text = _norm(completion)
    matched = sum(
        1 for clause in include if any(_norm(v) in text for v in _variants(clause))
    )
    violated = 0
    for clause in rules.get("exclude", []) or []:
        if any(_norm(v) and _norm(v) in text for v in _variants(clause)):
            violated += 1
    return max(0.0, min(1.0, (matched - violated) / total))


def partial_order(completion, rules):
    """Position accuracy: fraction of required items whose rank (by first
    occurrence in the answer) matches their rank in the target order."""
    seq = rules.get("order")
    if not seq:
        return 1.0 if grade_include_exclude(completion, rules) else 0.0
    n = len(seq)
    if n == 0:
        return 0.0
    positions = []
    for clause in seq:
        best = None
        for v in _variants(clause):
            p = _first_label_pos(completion, v)
            if p is not None and (best is None or p < best):
                best = p
        positions.append(best)
    present = sorted((pos, i) for i, pos in enumerate(positions) if pos is not None)
    model_rank = {i: r for r, (_pos, i) in enumerate(present)}
    correct = sum(1 for i in range(n) if model_rank.get(i) == i)
    return correct / n


def partial_credit(completion, rules, qtype):
    """Graded 'how close' score in [0, 1]. Equals the binary score for bool/mc/open."""
    qtype = (qtype or "").lower().strip()
    if qtype == "order":
        return partial_order(_strip_explanation(completion), rules)
    if qtype == "list":
        return partial_list_match(_strip_explanation(completion), rules)
    if qtype == "match":
        return partial_list_match(completion, rules)
    return 0.0  # not used for bool/mc/open (binary only)


def compute_bertscore_metrics(completion, rules):
    """Max BERTScore-F1 of the answer against the canonical include phrases."""
    include = rules.get("include", []) or []
    references = [clause[0] for clause in include if clause]
    if not references:
        return 0.0
    bert_score = get_bert_score()
    if bert_score is None:
        return 0.0
    try:
        candidates = [completion] * len(references)
        _, _, F1 = bert_score(candidates, references, lang="el", verbose=False)
        return float(F1.max().item())
    except Exception as e:
        logger.error("BERTScore failed: %s", e)
        return 0.0


# Backward-compatible alias (takes a JSON string, like the old helper).
def verify_open_answer(model_answer, verification_string):
    return grade_include_exclude(model_answer, parse_rules(verification_string))


# --------------------------------------------------------------------------- #
# lm-eval entry points
# --------------------------------------------------------------------------- #
def process_results(doc, results):
    completion = results[0] if results else ""
    rules = parse_rules(doc.get("Verification", ""))
    qtype = str(doc.get("Question_Type", "")).lower().strip()

    if qtype == "mc":
        correct = grade_mc(completion, rules)
    elif qtype == "order":
        correct = grade_order(_strip_explanation(completion), rules)
    elif qtype == "list":
        # strip the trailing explanation so it can't re-introduce excluded distractors
        correct = grade_include_exclude(_strip_explanation(completion), rules)
    else:                                    # bool, match, open
        correct = grade_include_exclude(completion, rules)

    out = {"exact_match": 1.0 if correct else 0.0}
    # Secondary "how close" signal for multi-part answers. Only emitted for
    # list/order/match (their YAMLs list `partial_credit` in metric_list);
    # bool/mc/open stay binary-only. exact_match remains the headline metric.
    if qtype in ("list", "order", "match"):
        out["partial_credit"] = partial_credit(completion, rules, qtype)
    return out
    # NOTE: BERTScore is intentionally NOT computed. transformers>=5 requires
    # torch>=2.4, which has no Intel-mac wheels, so it cannot run on this machine
    # (it would only spam errors and report 0.0). To re-enable on a Linux box, add
    # `bertscore_f1` back to open.yaml's metric_list and, for open docs, return
    #   metrics["bertscore_f1"] = compute_bertscore_metrics(completion, rules)


def doc_to_target_closed(doc):
    """Gold answer for the logs (display only; scoring is done in process_results)."""
    try:
        return parse_rules(doc["Verification"])["include"][0][0]
    except Exception:
        return ""


# --------------------------------------------------------------------------- #
# Per-format dataset filters
# --------------------------------------------------------------------------- #
def _filter(dataset, qtype):
    return dataset.filter(
        lambda x: str(x["Question_Type"]).lower().strip() == qtype
    ).shuffle(seed=42)


def process_mc(dataset):    return _filter(dataset, "mc")
def process_bool(dataset):  return _filter(dataset, "bool")
def process_open(dataset):  return _filter(dataset, "open")
def process_list(dataset):  return _filter(dataset, "list")
def process_order(dataset): return _filter(dataset, "order")
def process_match(dataset): return _filter(dataset, "match")
