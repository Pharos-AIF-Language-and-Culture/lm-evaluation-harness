"""
MT-MATH100 evaluator notes:
- Parser-based rule verification requires `math_verify`.
- Install via: `pip install -e ".[math]"` (from repo root) or `pip install "lm_eval[math]"`.
- If `math_verify` is unavailable, this task falls back to the Hendrycks-style
  normalization matcher (`is_equiv`), which is less robust for symbolic answers.
"""

from typing import Dict, List

import datasets

try:
    from math_verify import LatexExtractionConfig, parse, verify

    HAS_MATH_VERIFY = True
except ModuleNotFoundError:
    HAS_MATH_VERIFY = False


HF_DATASET_PATH = "amphora/MCLM"
HF_DATASET_NAME = "MT-MATH100"


def load_dataset(
    language_column: str,
    dataset_path: str = HF_DATASET_PATH,
    dataset_name: str = HF_DATASET_NAME,
    split: str = "test",
    **_,
):
    if not language_column:
        raise ValueError("dataset_kwargs.language_column must be provided")

    dataset = datasets.load_dataset(
        dataset_path,
        name=dataset_name,
        split=split,
    )

    if language_column not in dataset.column_names:
        raise ValueError(
            f"Language column '{language_column}' not found in dataset columns. "
            f"Available columns: {dataset.column_names}"
        )
    if "answer" not in dataset.column_names:
        raise ValueError(
            "Required 'answer' column not found in dataset. "
            f"Available columns: {dataset.column_names}"
        )

    cols_to_drop = [
        col for col in dataset.column_names if col not in {language_column, "answer"}
    ]
    dataset = dataset.remove_columns(cols_to_drop)
    dataset = dataset.rename_column(language_column, "question")
    dataset = dataset.filter(
        lambda doc: doc["question"] is not None and str(doc["question"]).strip() != ""
    )

    return {split: dataset}


def process_results(doc: dict, results: List[str]) -> Dict[str, int]:
    if not results:
        return {"exact_match": 0}

    response = results[0]
    target = str(doc["answer"]).strip()
    extracted_answer = extract_final_answer(response)

    if verify_with_math_parser(response, extracted_answer, target):
        return {"exact_match": 1}

    return {"exact_match": int(is_equiv(extracted_answer, target))}


def extract_final_answer(response: str) -> str:
    indices = [pos for pos, char in enumerate(response) if char == "$"]
    if len(indices) <= 1:
        answer = response
    else:
        answer = response[indices[0] + 1 : indices[-1]]

    boxed_answer = last_boxed_only_string(response)
    if boxed_answer is not None:
        try:
            boxed_content = remove_boxed(boxed_answer)
            if boxed_content is not None:
                answer = boxed_content
        except (AssertionError, IndexError):
            pass

    return str(answer).strip()


def verify_with_math_parser(response: str, extracted_answer: str, target: str) -> bool:
    if not HAS_MATH_VERIFY:
        return False

    try:
        parsed_target = parse(target)
        if not parsed_target:
            parsed_target = parse(
                f"\\boxed{{{target}}}",
                extraction_config=[LatexExtractionConfig()],
            )
        if not parsed_target:
            return False

        candidate_texts = []
        for candidate in (
            response,
            extracted_answer,
            f"\\boxed{{{extracted_answer}}}",
            f"The final answer is {extracted_answer}.",
        ):
            if candidate and candidate not in candidate_texts:
                candidate_texts.append(candidate)

        for candidate in candidate_texts:
            parsed_candidate = parse(candidate)
            if not parsed_candidate and candidate == extracted_answer:
                parsed_candidate = parse(
                    f"\\boxed{{{candidate}}}",
                    extraction_config=[LatexExtractionConfig()],
                )
            if parsed_candidate and verify(parsed_target, parsed_candidate):
                return True
    except Exception:
        return False

    return False


# string normalization from hendrycks_math task

def is_equiv(str1, str2, verbose=False):
    if str1 is None and str2 is None:
        print("WARNING: Both None")
        return True
    if str1 is None or str2 is None:
        return False

    try:
        ss1 = strip_string(str1)
        ss2 = strip_string(str2)
        if verbose:
            print(ss1, ss2)
        return ss1 == ss2
    except Exception:
        return str1 == str2


def remove_boxed(s):
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[: len(left)] == left
        return s[len(left) :]

    left = "\\boxed{"

    assert s[: len(left)] == left
    assert s[-1] == "}"

    return s[len(left) : -1]


def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx : right_brace_idx + 1]

    return retval


def fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except AssertionError:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string


def fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except AssertionError:
        return string


def remove_right_units(string):
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    else:
        return string


def fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string


def strip_string(string):
    string = string.replace("\n", "")
    string = string.replace("\\!", "")
    string = string.replace("\\\\", "\\")
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")
    string = string.replace("\\$", "")
    string = remove_right_units(string)
    string = string.replace("\\%", "")
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")

    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    string = fix_sqrt(string)
    string = string.replace(" ", "")
    string = fix_fracs(string)

    if string == "0.5":
        string = "\\frac{1}{2}"

    string = fix_a_slash_b(string)

    return string
