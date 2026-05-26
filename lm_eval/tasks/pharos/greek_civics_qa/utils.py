import re
import numpy as np
import sacrebleu
from rouge_score import rouge_scorer

try:
    from bert_score import score as bert_score_fn
    BERTSCORE_AVAILABLE = True
except ImportError:
    BERTSCORE_AVAILABLE = False


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
