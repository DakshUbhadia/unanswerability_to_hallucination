from scripts.import_judge_labels import _safe_kappa


def test_safe_kappa_returns_none_when_both_label_vectors_are_constant():
    assert _safe_kappa(["ANSWER"] * 10, ["ANSWER"] * 10) is None


def test_safe_kappa_preserves_finite_values():
    assert _safe_kappa(["ANSWER", "ABSTAIN"], ["ANSWER", "ABSTAIN"]) == 1.0
