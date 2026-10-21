"""Stage 4: extract abstention directions and matched negative controls."""
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from src.config import CFG


def compute_candidate_abstention_directions(train_ans: torch.Tensor, train_unans: torch.Tensor,
                                             eps: float = CFG.eps_norm) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert train_ans.shape == train_unans.shape and train_ans.ndim == 3
    diff = train_unans.to(torch.float64) - train_ans.to(torch.float64)
    norms = torch.linalg.vector_norm(diff, ord=2, dim=-1)
    assert bool((norms > 1e-3).all()), "A learned direction has near-zero norm"
    unit = diff / (norms.unsqueeze(-1) + eps)
    unit = (unit / torch.linalg.vector_norm(unit, ord=2, dim=-1, keepdim=True)).to(torch.float32)
    return diff.to(torch.float32), unit, norms.to(torch.float32)


def generate_monte_carlo_random_controls(n_controls: int = CFG.n_ctrl_random, n_layers: int = 28,
                                        k_suffix: int = 5, d_model: int = 3072,
                                        seed: int = CFG.seed_random, eps: float = CFG.eps_norm) -> torch.Tensor:
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    raw = torch.randn((n_controls, n_layers, k_suffix, d_model), generator=gen, dtype=torch.float64)
    unit = (raw / torch.linalg.vector_norm(raw, ord=2, dim=-1, keepdim=True).clamp(min=eps)).to(torch.float32)
    return unit / torch.linalg.vector_norm(unit, ord=2, dim=-1, keepdim=True).clamp(min=eps)


def get_norm_scaled_random_controls(random_controls_unit: torch.Tensor, norms: Union[torch.Tensor, float]) -> torch.Tensor:
    if random_controls_unit.ndim == 2:
        return random_controls_unit * float(norms)
    return random_controls_unit * torch.as_tensor(norms, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)


def generate_shuffled_label_controls(train_pair_acts_ans: torch.Tensor, train_pair_acts_unans: torch.Tensor,
                                     n_controls: int = CFG.n_ctrl_shuffled, seed: int = CFG.seed_shuffled,
                                     eps: float = CFG.eps_norm) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    assert train_pair_acts_ans.shape == train_pair_acts_unans.shape and train_pair_acts_ans.ndim == 4
    n = train_pair_acts_ans.shape[0]
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    swaps = torch.randint(0, 2, (n_controls, n), generator=gen, dtype=torch.int64)
    lo, hi = n // 4, 3 * n // 4
    counts = swaps.sum(dim=1)
    assert bool(((counts >= lo) & (counts <= hi)).all()), "Degenerate shuffled-label permutation"
    assert len({tuple(row.tolist()) for row in swaps}) == n_controls, "Duplicate shuffled-label permutations"
    signs = (1 - 2 * swaps).to(torch.int8)
    diffs = train_pair_acts_unans.to(torch.float64) - train_pair_acts_ans.to(torch.float64)
    shuffled = torch.einsum("mn,nlkd->mlkd", signs.to(torch.float64), diffs) / float(n)
    norms = torch.linalg.vector_norm(shuffled, ord=2, dim=-1)
    assert bool((norms > eps).all()), "A shuffled control has near-zero norm"
    unit = shuffled / norms.unsqueeze(-1)
    unit = unit.to(torch.float32)
    unit = unit / torch.linalg.vector_norm(unit, ord=2, dim=-1, keepdim=True).clamp(min=eps)
    return shuffled.to(torch.float32), unit, norms.to(torch.float32), signs


def compute_orthogonalized_concept_control(train_pair_acts_ans: torch.Tensor, families: List[str],
                                            unit_directions: torch.Tensor, norms: torch.Tensor,
                                            eps: float = CFG.eps_norm) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    assert len(families) == train_pair_acts_ans.shape[0]
    mask_a = torch.tensor([f == "A" for f in families], dtype=torch.bool)
    mask_b = torch.tensor([f == "B" for f in families], dtype=torch.bool)
    assert int(mask_a.sum()) == int(mask_b.sum()) == len(families) // 2
    acts = train_pair_acts_ans.to(torch.float64)
    raw = acts[mask_a].mean(0) - acts[mask_b].mean(0)
    rhat = unit_directions.to(torch.float64)
    rhat = rhat / torch.linalg.vector_norm(rhat, ord=2, dim=-1, keepdim=True).clamp(min=eps)
    raw_norm = torch.linalg.vector_norm(raw, ord=2, dim=-1)
    dot = (raw * rhat).sum(-1)
    cosine = dot / raw_norm.clamp(min=eps)
    assert bool((cosine.abs() < 0.999).all()), "Concept control is nearly collinear with learned direction"
    ortho = raw - dot.unsqueeze(-1) * rhat
    ortho_norm = torch.linalg.vector_norm(ortho, ord=2, dim=-1, keepdim=True)
    assert bool((ortho_norm.squeeze(-1) > 1e-3).all()), "Orthogonal concept residual is too small"
    unit = (ortho / ortho_norm).to(torch.float32)
    # Remove float32 casting residue against the exact serialized learned unit vectors.
    unit = unit - (unit * unit_directions).sum(-1, keepdim=True) * unit_directions
    unit = unit / torch.linalg.vector_norm(unit, ord=2, dim=-1, keepdim=True).clamp(min=eps)
    scaled = unit * norms.unsqueeze(-1)
    return raw.to(torch.float32), unit, scaled, cosine.to(torch.float32)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_all_directions_and_controls(cache_dict: Dict[str, Any], pairs_path: Optional[Path] = CFG.data_dir / "pairs.jsonl") -> Tuple[Dict[str, Any], Dict[str, Any]]:
    started = time.perf_counter()
    ans, unans = cache_dict["train_pair_acts_ans"], cache_dict["train_pair_acts_unans"]
    n, layers, k, d = ans.shape
    assert torch.allclose(ans.to(torch.float64).mean(0).float(), cache_dict["train_ans"], atol=1e-5, rtol=1e-5)
    assert torch.allclose(unans.to(torch.float64).mean(0).float(), cache_dict["train_unans"], atol=1e-5, rtol=1e-5)
    if pairs_path is not None and Path(pairs_path).is_file():
        with Path(pairs_path).open(encoding="utf-8") as f:
            parsed = [json.loads(line) for line in f if line.strip()]
        records = sorted((r for r in parsed if r.get("split") == "train"), key=lambda r: r["pair_id"])
        assert [r["pair_id"] for r in records] == cache_dict["pair_ids"]
        assert [r["template_id"] for r in records] == cache_dict["template_ids"]
        assert [r["family"] for r in records] == cache_dict["families"]
    diff, unit, norms = compute_candidate_abstention_directions(cache_dict["train_ans"], cache_dict["train_unans"])
    random_unit = generate_monte_carlo_random_controls(CFG.n_ctrl_random, layers, k, d)
    shuf_diff, shuf_unit, shuf_norms, signs = generate_shuffled_label_controls(ans, unans)
    concept_raw, concept_unit, concept_scaled, cos = compute_orthogonalized_concept_control(ans, cache_dict["families"], unit, norms)
    rand_error = (torch.linalg.vector_norm(get_norm_scaled_random_controls(random_unit, norms), dim=-1) - norms).abs().max().item()
    concept_error = (torch.linalg.vector_norm(concept_scaled, dim=-1) - norms).abs().max().item()
    post_dot = (unit * concept_unit).sum(-1).abs().max().item()
    out = {"diff_in_means": diff, "unit_directions": unit, "norms": norms,
           "random_controls_unit": random_unit, "shuffled_controls_diff": shuf_diff,
           "shuffled_controls_unit": shuf_unit, "shuffled_controls_norms": shuf_norms,
           "shuffled_swap_signs": signs, "r_concept_raw": concept_raw,
           "r_concept_unit": concept_unit, "r_concept_ortho_scaled": concept_scaled,
           "cos_pre_ortho": cos, "model_name": cache_dict["model_name"], "n_layers": layers,
           "k_suffix": k, "d_model": d, "suffix_indices": cache_dict["suffix_indices"],
           "suffix_token_ids": cache_dict["suffix_token_ids"]}
    manifest = {"stage": 4, "model_name": cache_dict["model_name"], "n_train_pairs": n,
                "n_layers": layers, "k_suffix": k, "d_model": d,
                "n_ctrl_random": CFG.n_ctrl_random, "n_ctrl_shuffled": CFG.n_ctrl_shuffled,
                "n_ctrl_concept": CFG.n_ctrl_concept, "seed_random": CFG.seed_random, "seed_shuffled": CFG.seed_shuffled,
                "min_learned_norm": float(norms.min()), "max_learned_norm": float(norms.max()),
                "min_shuffled_norm": float(shuf_norms.min()), "max_shuffled_norm": float(shuf_norms.max()),
                "cos_pre_ortho_min": float(cos.min()), "cos_pre_ortho_max": float(cos.max()),
                "cos_pre_ortho_mean_abs": float(cos.abs().mean()), "max_post_ortho_abs_dot": post_dot,
                "max_concept_norm_diff": concept_error, "max_random_norm_diff": rand_error,
                "elapsed_sec": time.perf_counter() - started, "passed_acceptance_gate": True}
    return out, manifest


def save_stage4_artifacts(directions_dict: Dict[str, Any], manifest_dict: Dict[str, Any],
                          directions_path: Path = CFG.stage4_out_path,
                          manifest_path: Path = CFG.stage4_manifest_path) -> None:
    directions_path, manifest_path = Path(directions_path), Path(manifest_path)
    directions_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = directions_path.with_suffix(directions_path.suffix + ".tmp")
    torch.save(directions_dict, tmp)
    tmp.replace(directions_path)
    manifest_dict["directions_file_size_bytes"] = directions_path.stat().st_size
    manifest_dict["directions_sha256"] = _sha256(directions_path)
    with manifest_path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(manifest_dict, f, indent=2, ensure_ascii=False)
        f.write("\n")


def verify_stage4_outputs(directions_path: Path = CFG.stage4_out_path,
                          manifest_path: Path = CFG.stage4_manifest_path,
                          stage0_path: Optional[Path] = None) -> Dict[str, Any]:
    directions_path, manifest_path = Path(directions_path), Path(manifest_path)
    assert directions_path.is_file() and manifest_path.is_file(), "Stage 4 output or manifest is missing"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    data = torch.load(directions_path, map_location="cpu", weights_only=False)
    manifest_required = {"stage", "model_name", "n_train_pairs", "n_layers", "k_suffix", "d_model",
                         "n_ctrl_random", "n_ctrl_shuffled", "n_ctrl_concept", "seed_random", "seed_shuffled",
                         "min_learned_norm", "max_learned_norm", "min_shuffled_norm", "max_shuffled_norm",
                         "cos_pre_ortho_min", "cos_pre_ortho_max", "cos_pre_ortho_mean_abs",
                         "max_post_ortho_abs_dot", "max_concept_norm_diff", "max_random_norm_diff",
                         "directions_file_size_bytes", "directions_sha256", "source_stage3_sha256",
                         "elapsed_sec", "passed_acceptance_gate"}
    assert manifest_required <= manifest.keys(), f"Manifest is missing keys: {sorted(manifest_required - manifest.keys())}"
    vector_keys = ("diff_in_means", "unit_directions", "random_controls_unit", "shuffled_controls_diff",
                   "shuffled_controls_unit", "r_concept_raw", "r_concept_unit", "r_concept_ortho_scaled")
    required = set(vector_keys) | {"norms", "shuffled_controls_norms", "shuffled_swap_signs", "cos_pre_ortho", "model_name", "n_layers", "k_suffix", "d_model", "suffix_indices", "suffix_token_ids"}
    assert required <= data.keys(), f"Missing Stage 4 keys: {sorted(required - data.keys())}"
    L, K, D = int(data["n_layers"]), int(data["k_suffix"]), int(data["d_model"])
    assert manifest["stage"] == 4 and manifest["model_name"] == data["model_name"]
    assert (manifest["n_layers"], manifest["k_suffix"], manifest["d_model"]) == (L, K, D)
    assert (manifest["n_ctrl_random"], manifest["n_ctrl_shuffled"]) == (CFG.n_ctrl_random, CFG.n_ctrl_shuffled)
    expected = {"diff_in_means": (L,K,D), "unit_directions": (L,K,D), "norms": (L,K),
                "random_controls_unit": (CFG.n_ctrl_random,L,K,D), "shuffled_controls_diff": (CFG.n_ctrl_shuffled,L,K,D),
                "shuffled_controls_unit": (CFG.n_ctrl_shuffled,L,K,D), "shuffled_controls_norms": (CFG.n_ctrl_shuffled,L,K),
                "shuffled_swap_signs": (CFG.n_ctrl_shuffled, manifest["n_train_pairs"]), "r_concept_raw": (L,K,D),
                "r_concept_unit": (L,K,D), "r_concept_ortho_scaled": (L,K,D), "cos_pre_ortho": (L,K)}
    for key, shape in expected.items():
        t = data[key]
        assert isinstance(t, torch.Tensor) and tuple(t.shape) == shape, f"Invalid shape for {key}: {getattr(t, 'shape', None)}"
        assert t.dtype == (torch.int8 if key == "shuffled_swap_signs" else torch.float32), f"Invalid dtype for {key}"
        assert bool(torch.isfinite(t).all()), f"Non-finite values in {key}"
    for key in ("unit_directions", "random_controls_unit", "shuffled_controls_unit", "r_concept_unit"):
        norm = torch.linalg.vector_norm(data[key], dim=-1)
        assert float((norm - 1).abs().max()) < 1e-5, f"{key} is not unit normalized"
    assert torch.allclose(data["unit_directions"] * data["norms"].unsqueeze(-1), data["diff_in_means"], atol=1e-5, rtol=1e-5)
    post_dot = (data["unit_directions"] * data["r_concept_unit"]).sum(-1).abs().max().item()
    assert post_dot < CFG.stage4_ortho_tol
    concept_err = (torch.linalg.vector_norm(data["r_concept_ortho_scaled"], dim=-1) - data["norms"]).abs().max().item()
    random_scaled = get_norm_scaled_random_controls(data["random_controls_unit"], data["norms"])
    random_err = (torch.linalg.vector_norm(random_scaled, dim=-1) - data["norms"].unsqueeze(0)).abs().max().item()
    assert concept_err < 1e-5 and random_err < 1e-5
    assert bool((data["cos_pre_ortho"].abs() < 0.999).all())
    assert manifest.get("passed_acceptance_gate") is True
    assert manifest["directions_file_size_bytes"] == directions_path.stat().st_size
    assert manifest["directions_sha256"] == _sha256(directions_path)
    assert manifest["max_post_ortho_abs_dot"] < CFG.stage4_ortho_tol
    assert manifest["max_concept_norm_diff"] < 1e-5 and manifest["max_random_norm_diff"] < 1e-5
    if stage0_path is not None and Path(stage0_path).is_file():
        stage0 = json.loads(Path(stage0_path).read_text(encoding="utf-8"))
        assert (L,K,D) == (int(stage0["n_layers"]), int(stage0["k_suffix"]), int(stage0["d_model"]))
        assert data["suffix_token_ids"] == stage0["suffix_token_ids"]
    return manifest
