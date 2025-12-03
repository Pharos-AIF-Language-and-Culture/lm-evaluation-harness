import re


def extract_boxed_answer(text):
    """
    Finds the \boxed{...} pattern in the text and returns only the content inside it.
    """
    if not text:
        return ""
    
    # Search for \boxed{...} with or without $
    match = re.search(r'\\boxed\{([^}]+)\}', text)
    
    if match:
        return match.group(1).strip()
    
    return ""


def process_results(doc, results):
    """
    Takes the model's answer, extracts the boxed content, and compares it to the target.
    """
    prediction = results[0] if results else ""
    target = str(doc.get("answer", "")).strip()
    pred_boxed = extract_boxed_answer(prediction)
    is_correct = pred_boxed == target
    
    return {
        "exact_match": float(is_correct)
    }
