import re
import numpy as np
import sacrebleu
from rouge_score import rouge_scorer

try:
    from bert_score import score as bert_score_fn
    BERTSCORE_AVAILABLE = True
except ImportError:
    BERTSCORE_AVAILABLE = False


# --- Utility functions for Chronological Order QA task ---

def doc_to_text(doc):
    # Dynamically utilize the dataset's native instruction phrasing
    prompt = f"{doc['question'].strip()}\n\n"
    for event in doc["input"]:
        prompt += f"- {event}\n"
    # Append standardized formatting instruction to ensure parse-robust outputs
    prompt += (
        "\nΑπαντήστε αποκλειστικά με τα γράμματα που προηγούνται κάθε στοιχείου, "
        "χωρισμένα με κόμμα (π.χ., α, δ, β, ε, γ).\n\n"
        "Απάντηση:"
    )
    return prompt

def doc_to_target(doc):
    return ", ".join(doc["answer"])

def parse_prediction(pred_str, valid_keys):
    pred_lower = pred_str.lower()
    
    # Split by standard punctuation and whitespace to isolate words/tokens cleanly
    split_tokens = [t for t in re.split(r'[\s,.:;!\-\[\]\'\"()]+', pred_lower) if t]
    extracted = [t for t in split_tokens if t in valid_keys]
    
    # Fallback: if no isolated keys found, check if model generated a joint string of keys (e.g. "αδβεγ")
    if not extracted:
        for t in split_tokens:
            if len(t) == len(valid_keys) and all(c in valid_keys for c in t):
                extracted = list(t)
                break
                
    # Deduplicate keeping the LAST occurrence of each key (prioritizes final answers over preambles)
    seen = set()
    deduped = [x for x in reversed(extracted) if not (x in seen or seen.add(x))]
    deduped.reverse()
    
    return deduped

def kendall_tau_distance(true_order, pred_order):
    # Deduplicate true_order keeping order of appearance to handle any duplicate keys in the dataset safely
    seen_true = set()
    true_order_dedup = [x for x in true_order if not (x in seen_true or seen_true.add(x))]
    
    # Align pred_order with true_order_dedup to ensure they are permutations of the same elements
    cleaned_pred = [x for x in pred_order if x in true_order_dedup]
    
    # Append any missing elements from true_order_dedup to the end
    missing = [x for x in true_order_dedup if x not in cleaned_pred]
    cleaned_pred = cleaned_pred + missing
    
    n = len(true_order_dedup)
    if n <= 1:
        return 1.0
        
    true_index = {val: idx for idx, val in enumerate(true_order_dedup)}
    pred_indices = [true_index[x] for x in cleaned_pred]
    
    concordant = 0
    discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            if pred_indices[i] < pred_indices[j]:
                concordant += 1
            else:
                discordant += 1
                 
    total_pairs = n * (n - 1) // 2
    return (concordant - discordant) / total_pairs

def process_results(doc, results):
    pred_str = results[0]
    true_answer = doc["answer"]
    valid_keys = set(true_answer)
    
    extracted_pred = parse_prediction(pred_str, valid_keys)
    
    # Exact match: 1.0 if lists match exactly, else 0.0
    exact_match = 1.0 if extracted_pred == true_answer else 0.0
    
    # Kendall's Tau distance
    kt_score = kendall_tau_distance(true_answer, extracted_pred)
    
    return {
        "exact_match": exact_match,
        "kendalls_tau": kt_score
    }


# --- Utility functions for Generative Modern History QA task ---

class GreekTokenizer:
    def tokenize(self, text):
        text = text.lower()
        # Keep English/Greek letters and numbers, discard other symbols
        return re.findall(r'[a-z0-9\u0370-\u03ff\u1f00-\u1fff]+', text)


ROUGE_SCORER = None


def process_results_gen(doc, results):
    completion = results[0]
    true_refs = [doc["answer"]]

    # BLEU (sacrebleu with international tokenization)
    bleu_scores = [bleu([[ref]], [completion]) for ref in true_refs]
    bleu_max = np.nanmax(bleu_scores)

    # ROUGE with custom Greek-safe tokenizer
    global ROUGE_SCORER
    if ROUGE_SCORER is None:
        ROUGE_SCORER = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeLsum"], 
            tokenizer=GreekTokenizer()
        )
    
    rouge_scores = []
    for ref in true_refs:
        scores = ROUGE_SCORER.score(ref, completion)
        rouge_scores.append({
            "rouge1": scores["rouge1"].fmeasure * 100.0,
            "rouge2": scores["rouge2"].fmeasure * 100.0,
            "rougeL": scores["rougeLsum"].fmeasure * 100.0,
        })

    rouge1_max = np.nanmax([s["rouge1"] for s in rouge_scores])
    rouge2_max = np.nanmax([s["rouge2"] for s in rouge_scores])
    rougeL_max = np.nanmax([s["rougeL"] for s in rouge_scores])

    # BERTScore using multilingual BERT to handle semantic similarity in Greek
    bertscore_f1_max = 0.0
    if BERTSCORE_AVAILABLE:
        # P, R, F1 are returned as tensors
        P, R, F1 = bert_score_fn(
            [completion],
            true_refs,
            lang="el",
            model_type="bert-base-multilingual-cased",
            verbose=False,
        )
        bertscore_f1_max = F1.max().item()

    return {
        "bleu_max": bleu_max,
        "rouge1_max": rouge1_max,
        "rouge2_max": rouge2_max,
        "rougeL_max": rougeL_max,
        "bertscore_f1_max": bertscore_f1_max,
    }


def bleu(refs, preds):
    score = sacrebleu.corpus_bleu(
        preds,
        refs,
        smooth_method="exp",
        smooth_value=0.0,
        force=False,
        lowercase=False,
        tokenize="intl",
        use_effective_order=False,
    ).score
    return score
