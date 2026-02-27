from functools import partial
import numpy as np
import re
import string

choices = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]


def format_cot_example(example, including_answer=True):
    prompt = "Ερώτηση:\n"
    question = example["question"]
    options = example["options"]
    prompt += question + "\n"
    prompt += "Πιθανές απαντήσεις:\n"

    for i, opt in enumerate(options):
        if i >= len(choices):
            break
        prompt += "{}. {}\n".format(choices[i], opt)

    if including_answer:
        answer_idx = example["answer_index"]
        cot_content = example["cot_content"].replace(
            "Α: Σκέψου βήμα προς βήμα", "Απάντηση¨Σκέψου βήμα προς βήμα."
        )
        prompt += cot_content + "\n" + f"Η απάντηση είναι {choices[answer_idx]}\n\n"
    else:
        prompt += "Απάντηση: Σκέψου βήμα προς βήμα."

    return prompt


doc_to_text = partial(format_cot_example, including_answer=False)
fewshot_to_text = partial(format_cot_example, including_answer=True)


def process_docs(dataset, subject):
    return dataset.filter(lambda x: x["category"] == subject)


process_biology = partial(process_docs, subject="biology")
process_business = partial(process_docs, subject="business")
process_chemistry = partial(process_docs, subject="chemistry")
process_computer_science = partial(process_docs, subject="computer science")
process_economics = partial(process_docs, subject="economics")
process_engineering = partial(process_docs, subject="engineering")
process_health = partial(process_docs, subject="health")
process_history = partial(process_docs, subject="history")
process_law = partial(process_docs, subject="law")
process_math = partial(process_docs, subject="math")
process_other = partial(process_docs, subject="other")
process_philosophy = partial(process_docs, subject="philosophy")
process_physics = partial(process_docs, subject="physics")
process_psychology = partial(process_docs, subject="psychology")

def exact_match_greek_char_fix(
    predictions,
    references,
    regexes_to_ignore=None,
    ignore_case=False,
    ignore_punctuation=False,
    ignore_numbers=False,
):
    # fix Greek vs English choice issues
    predictions = [pred.replace("Α", "A").replace("Β", "B") for pred in predictions]
    if regexes_to_ignore is not None:
        for s in regexes_to_ignore:
            predictions = np.array([re.sub(s, "", x) for x in predictions])
            references = np.array([re.sub(s, "", x) for x in references])
    else:
        predictions = np.asarray(predictions)
        references = np.asarray(references)

    if ignore_case:
        predictions = np.char.lower(predictions)
        references = np.char.lower(references)

    if ignore_punctuation:
        repl_table = string.punctuation.maketrans("", "", string.punctuation)
        predictions = np.char.translate(predictions, table=repl_table)
        references = np.char.translate(references, table=repl_table)

    if ignore_numbers:
        repl_table = string.digits.maketrans("", "", string.digits)
        predictions = np.char.translate(predictions, table=repl_table)
        references = np.char.translate(references, table=repl_table)

    score_list = predictions == references

    return {"exact_match": np.mean(score_list)}
