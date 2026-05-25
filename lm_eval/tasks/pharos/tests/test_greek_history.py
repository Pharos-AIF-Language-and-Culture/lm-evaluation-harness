import pytest
from lm_eval.tasks.pharos.greek_history.utils import parse_prediction, kendall_tau_distance, process_results

def test_parse_prediction():
    valid_keys = {'α', 'β', 'γ', 'δ', 'ε'}
    
    # Standard format
    assert parse_prediction("α, δ, β, ε, γ", valid_keys) == ['α', 'δ', 'β', 'ε', 'γ']
    assert parse_prediction("α - δ - β - ε - γ", valid_keys) == ['α', 'δ', 'β', 'ε', 'γ']
    
    # Joint fallback
    assert parse_prediction("αδβεγ", valid_keys) == ['α', 'δ', 'β', 'ε', 'γ']
    
    # Stringified list format
    assert parse_prediction("['α', 'δ', 'β', 'ε', 'γ']", valid_keys) == ['α', 'δ', 'β', 'ε', 'γ']
    assert parse_prediction('["α", "δ", "β", "ε", "γ"]', valid_keys) == ['α', 'δ', 'β', 'ε', 'γ']
    
    # Greek prose preamble
    assert parse_prediction("Η σωστή σειρά είναι α, δ, β, ε, γ", valid_keys) == ['α', 'δ', 'β', 'ε', 'γ']
    assert parse_prediction("Η σωστή σειρά είναι: α, δ, β, ε, γ.", valid_keys) == ['α', 'δ', 'β', 'ε', 'γ']
    
    # Numbered listing
    assert parse_prediction("1. α\n2. δ\n3. β\n4. ε\n5. γ", valid_keys) == ['α', 'δ', 'β', 'ε', 'γ']

def test_kendall_tau_distance():
    true_order = ['α', 'δ', 'β', 'ε', 'γ']
    
    # Exact match: correlation should be 1.0
    assert kendall_tau_distance(true_order, ['α', 'δ', 'β', 'ε', 'γ']) == pytest.approx(1.0)
    
    # Complete inversion: correlation should be -1.0
    assert kendall_tau_distance(true_order, ['γ', 'ε', 'β', 'δ', 'α']) == pytest.approx(-1.0)
    
    # Partial correct order (e.g., single adjacent swap: δ and α swapped)
    # true:   α, δ, β, ε, γ
    # pred:   δ, α, β, ε, γ
    # Out of 10 pairs, only (α, δ) is discordant. 9 concordant, 1 discordant.
    # Tau = (9 - 1) / 10 = 0.8
    assert kendall_tau_distance(true_order, ['δ', 'α', 'β', 'ε', 'γ']) == pytest.approx(0.8)

def test_process_results():
    doc = {
        "answer": ['α', 'δ', 'β', 'ε', 'γ']
    }
    
    # Test perfect match prediction
    res_perfect = process_results(doc, ["α, δ, β, ε, γ"])
    assert res_perfect["exact_match"] == 1.0
    assert res_perfect["kendalls_tau"] == pytest.approx(1.0)
    
    # Test stringified list prediction
    res_list = process_results(doc, ["['α', 'δ', 'β', 'ε', 'γ']"])
    assert res_list["exact_match"] == 1.0
    assert res_list["kendalls_tau"] == pytest.approx(1.0)
    
    # Test incorrect prediction
    res_wrong = process_results(doc, ["δ, α, β, ε, γ"])
    assert res_wrong["exact_match"] == 0.0
    assert res_wrong["kendalls_tau"] == pytest.approx(0.8)
    
    # Test preamble repeating the input keys (e.g. alphabetical list in preamble, correct order at end)
    res_preamble = process_results(
        doc,
        ["Η ερώτηση περιλαμβάνει τα στοιχεία α, β, γ, δ, ε. Η σωστή σειρά είναι: α, δ, β, ε, γ."]
    )
    assert res_preamble["exact_match"] == 1.0
    assert res_preamble["kendalls_tau"] == pytest.approx(1.0)

