# tests/test_stage1.py
import copy
from collections import Counter

import pytest
from src.config import CONFIG
from src.template_catalog import (
    generate_raw_template_catalog,
    calibrate_template_token_parity,
)
from src.dataset_builder import (
    create_template_splits,
    instantiate_pairs_and_b_ctrl,
    compute_max_ngram_phi,
    stratify_and_sample_popqa,
    normalize_aliases,
    find_unanswerable_style_confounds,
)


class WhitespaceMockTokenizer:
    """Fast deterministic mock tokenizer for offline unit testing."""

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [hash(tok) % 50000 for tok in text.replace("\n", " \n ").split()]


@pytest.fixture(scope="module")
def qwen_tokenizer():
    """The real target tokenizer; the catalog is calibrated against its BPE, not whitespace."""
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained(CONFIG.primary_model_name)
    except OSError as exc:
        pytest.skip(f"{CONFIG.primary_model_name} tokenizer unavailable: {exc}")


@pytest.fixture(scope="module")
def calibrated_dataset(qwen_tokenizer):
    calibrated, audit_rows = calibrate_template_token_parity(
        generate_raw_template_catalog(48, tokenizer=qwen_tokenizer), tokenizer=qwen_tokenizer, max_diff=CONFIG.max_token_diff_pair
    )
    # Unit tests provide an explicit in-memory review fixture; production reads the CSV.
    for template in calibrated:
        for flag in ("check_answer_unambiguous", "check_zero_unans_support", "check_distinct_predicate", "check_b_ctrl_valid"):
            template[flag] = True
    splits = create_template_splits(calibrated, seed_split=CONFIG.seed_split)
    pairs, b_ctrl = instantiate_pairs_and_b_ctrl(calibrated, splits, tokenizer=qwen_tokenizer)
    return calibrated, audit_rows, splits, pairs, b_ctrl


def test_template_catalog_and_token_parity_calibration(calibrated_dataset):
    """Verifies that all 96 templates calibrate to |delta_L| <= 2."""
    calibrated, audit_rows, _, _, _ = calibrated_dataset
    assert len(calibrated) == 96
    assert Counter(t["family"] for t in calibrated) == {"A": 48, "B": 48}
    assert len({t["template_id"] for t in calibrated}) == 96
    assert all("canonical_unans_template" not in t for t in calibrated)

    assert all(row["post_fix_passed"] is True for row in audit_rows)


def test_template_level_splits_disjoint_and_balanced(calibrated_dataset):
    """Verifies strict disjointness and 48/24/24 family balance in splits.json."""
    _, _, splits, _, _ = calibrated_dataset
    tr, va, te = (
        splits["train_template_ids"],
        splits["val_template_ids"],
        splits["test_template_ids"],
    )
    assert splits["seed_split"] == CONFIG.seed_split
    assert len(tr) == 48 and len(va) == 24 and len(te) == 24
    assert set(tr).isdisjoint(va) and set(tr).isdisjoint(te) and set(va).isdisjoint(te)

    # Check exact 50/25/25 split within each family
    for fam in ("famA_", "famB_"):
        assert sum(1 for x in tr if x.startswith(fam)) == 24
        assert sum(1 for x in va if x.startswith(fam)) == 12
        assert sum(1 for x in te if x.startswith(fam)) == 12


def test_cut3_splits_are_16_8_8_per_family():
    """Verifies the Cut 3 schedule (64 templates -> 32/16/16 total, 16/8/8 per family)."""
    raw = generate_raw_template_catalog(n_per_family=CONFIG.n_templates_cut3 // 2)
    splits = create_template_splits(raw, seed_split=CONFIG.seed_split)
    assert len(splits["train_template_ids"]) == 32
    assert len(splits["val_template_ids"]) == 16
    assert len(splits["test_template_ids"]) == 16
    assert sum(1 for x in splits["test_template_ids"] if x.startswith("famB_")) == 8


def test_pairs_b_ctrl_and_ngram_leakage_gate(calibrated_dataset):
    """Verifies 384 pairs, 48 b_ctrl prompts, |delta_L| <= 2, and max |phi| <= 0.15."""
    _, _, _, pairs, b_ctrl = calibrated_dataset

    assert len(pairs) == 384
    assert Counter(p["split"] for p in pairs) == {"train": 192, "val": 96, "test": 96}
    assert [p["pair_id"] for p in pairs] == [f"pair_{i:04d}" for i in range(1, 385)]
    assert len(b_ctrl) == 48

    for p in pairs:
        assert abs(p["token_len_diff"]) <= CONFIG.max_token_diff_pair
        assert p["token_len_diff"] == p["ans_token_len"] - p["unans_token_len"]
        assert p["ground_truth"] in p["ans_prompt"]
        if p["family"] == "A":
            # Family B keeps the profile (and its stated fact) identical; only the question changes.
            assert p["ground_truth"] not in p["unans_prompt"]

    # Verify b_ctrl links exclusively to Family B test pairs
    test_famb_pair_ids = {
        p["pair_id"] for p in pairs if p["split"] == "test" and p["family"] == "B"
    }
    assert {b["pair_id"] for b in b_ctrl} == test_famb_pair_ids

    # b_ctrl reuses the exact unanswerable question and states its answer in the profile
    pair_by_id = {p["pair_id"]: p for p in pairs}
    for b in b_ctrl:
        unans_question = pair_by_id[b["pair_id"]]["unans_prompt"].split("\nQuestion: ")[1]
        profile, question = b["prompt"].split("\nQuestion: ")
        assert question == unans_question
        assert b["ground_truth"] in profile

    max_phi, worst_ngram, _ = compute_max_ngram_phi(pairs, max_n=3)
    assert max_phi <= CONFIG.max_ngram_phi, (
        f"N-gram '{worst_ngram}' exceeded leakage threshold: {max_phi:.4f} > {CONFIG.max_ngram_phi}"
    )


def test_calibrator_never_rewrites_authored_text():
    """Parity failures are reported without changing authored template wording."""
    tok = WhitespaceMockTokenizer()
    balanced = {
        "template_id": "synthetic_rewrite", "family": "A",
        "ans_template": "Context: {entity} was examined by {attr1}.\nQuestion: Who examined {entity}?",
        "unans_template": "Context: {entity} was reviewed by {attr2}.\nQuestion: Who examined {entity}?",
        "slots": [{"entity": "Northstar Project", "attr1": "Mara Ellison", "attr2": "Caleb Mercer", "ground_truth": "Mara Ellison"}],
    }
    fixed, audit = calibrate_template_token_parity([copy.deepcopy(balanced)], tokenizer=tok, max_diff=2)
    assert audit == []
    assert fixed[0]["unans_template"] == balanced["unans_template"]

    unbalanced = copy.deepcopy(balanced)
    unbalanced["unans_template"] = "Context: {entity} underwent a highly comprehensive secondary technical inspection by {attr2}.\nQuestion: Who examined {entity}?"
    original_text = unbalanced["unans_template"]
    fixed, audit = calibrate_template_token_parity([unbalanced], tokenizer=tok, max_diff=2)
    assert len(audit) == 1 and audit[0]["post_fix_passed"] is False
    assert fixed[0]["unans_template"] == original_text
    assert fixed[0]["slots"] == balanced["slots"]


def test_unanswerable_style_gate_rejects_unans_only_ly_and_filler():
    synthetic = [{
        "template_id": "synthetic_style", "slots": [{"entity": "Mara Ellison"}],
        "ans_template": "Profile: {entity} works outdoors.",
        "unans_template": "Profile: {entity} works methodically outdoors at present.",
    }]
    failures = find_unanswerable_style_confounds(synthetic)
    assert "synthetic_style" in failures
    assert any("methodically" in reason for reason in failures["synthetic_style"])
    assert any("at present" in reason for reason in failures["synthetic_style"])


def test_ngram_phi_detects_synthetic_leakage():
    """Verifies that compute_max_ngram_phi catches a leaked trigger phrase (|phi| == 1.0)."""
    leaked_pairs = [
        {
            "ans_prompt": f"Context {i}: normal fact stated.",
            "unans_prompt": f"Context {i}: magic_leak_phrase stated.",
        }
        for i in range(20)
    ]
    max_phi, _, top_10 = compute_max_ngram_phi(leaked_pairs, max_n=2)
    assert max_phi > 0.99
    # The trigger phrase is perfectly unans-correlated (+1); its replaced words are perfectly ans-correlated (-1).
    unans_leaks = {g for g, phi in top_10.items() if phi > 0.99}
    assert unans_leaks and all("magic" in g or "leak" in g or "phrase" in g for g in unans_leaks)
    assert top_10["normal"] < -0.99


def test_popqa_decile_stratification_on_synthetic_rows():
    """Verifies 60/decile (1-5) and 140/decile (6-10) sampling, determinism, and no replacement."""
    rows = [
        {
            "hf_row_idx": i,
            "question": f"q{i}",
            "prop": "occupation",
            "subj": f"s{i}",
            "obj": f"o{i}",
            "possible_answers": [f"o{i}"],
            "s_pop": float((i * 7919) % 5000),
        }
        for i in range(5000)
    ]
    sample = stratify_and_sample_popqa(rows, n_total=1000, seed_popqa=CONFIG.seed_popqa)
    assert len(sample) == 1000
    counts = Counter(r["s_pop_decile"] for r in sample)
    assert all(counts[d] == 60 for d in range(1, 6))
    assert all(counts[d] == 140 for d in range(6, 11))
    assert all("log_s_pop" in r and "possible_answers_clean" in r for r in sample)
    assert sample[0]["log_s_pop"] == pytest.approx(__import__("math").log10(sample[0]["s_pop"] + 1))
    assert len({r["hf_row_idx"] for r in sample}) == 1000
    assert [r["popqa_id"] for r in sample[:2]] == ["popqa_0001", "popqa_0002"]
    assert all(r["baseline_partition"] == "ungraded" and r["matched_pair_id"] is None for r in sample)
    assert "s_pop_decile" not in rows[0]

    again = stratify_and_sample_popqa(rows, n_total=1000, seed_popqa=CONFIG.seed_popqa)
    assert [r["hf_row_idx"] for r in again] == [r["hf_row_idx"] for r in sample]


def test_possible_answers_clean_removes_formatting_and_short_aliases():
    assert normalize_aliases([" A ", "Alpha\u200b Beta", "alpha beta", "XY"]) == ["Alpha Beta"]
    assert normalize_aliases(["A", "a", " "]) == ["A"]
