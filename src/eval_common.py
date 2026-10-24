"""Shared runtime, IO and batched-evaluation helpers for Stages 5-10."""
import csv
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from src.config import CFG

import torch
import torch.nn.functional as F


# ----------------------------- IO -------------------------------------------
def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return v if math.isfinite(v) else None
    if isinstance(obj, Path):
        return str(obj)
    return obj


def atomic_write_json(path: Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(_jsonable(obj), indent=2, allow_nan=False, ensure_ascii=False) + "\n",
                   encoding="utf-8", newline="\n")
    tmp.replace(path)


def atomic_write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(_jsonable(row), ensure_ascii=False, allow_nan=False) + "\n")
    tmp.replace(path)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _fmt_cell(v: Any, digits: int = 4) -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, float):
        if not math.isfinite(v):
            return ""
        return f"{v:.{digits}g}" if (abs(v) >= 1e4 or (0 < abs(v) < 1e-3)) else f"{v:.{digits}f}"
    return str(v)


def write_csv_and_md(rows: List[Dict[str, Any]], columns: Sequence[str], csv_path: Path, md_path: Path,
                     title: Optional[str] = None) -> None:
    csv_path, md_path = Path(csv_path), Path(md_path)
    for p in (csv_path, md_path):
        p.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(columns), extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({c: _fmt_cell(row.get(c), 6) for c in columns})
    lines = [f"**{title}**", ""] if title else []
    lines += ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    lines += ["| " + " | ".join(_fmt_cell(r.get(c)).replace("|", "/") for c in columns) + " |" for r in rows]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ----------------------------- contracts & data ------------------------------
def load_runtime_contract(stage0_path: Path = CFG.stage0_out_path,
                          stage2_path: Path = CFG.stage2_out_path) -> Dict[str, Any]:
    from src.stage_contract import load_stage2_model_contract
    stage0 = read_json(stage0_path)
    stage2 = load_stage2_model_contract(stage2_path)
    if stage0.get("passed_all_checks") is not True:
        raise RuntimeError(f"Stage 0 did not pass in {stage0_path}")
    if stage0["model_name"] != stage2["model_name"]:
        raise RuntimeError(f"Stage 0 model {stage0['model_name']!r} != Stage 2 model {stage2['model_name']!r}")
    contract = {**stage0, **stage2}
    if len(contract["suffix_token_ids"]) != int(contract["k_suffix"]):
        raise RuntimeError("Stage 0 suffix_token_ids length does not match k_suffix")
    contract["unknown_token_ids"] = list(dict.fromkeys(
        [int(contract["unknown_token_id_nospace"]), int(contract["unknown_token_id_space"])]))
    return contract


def load_split_pairs(split: str, pairs_path: Path = CFG.data_dir / "pairs.jsonl",
                     splits_path: Path = CFG.data_dir / "splits.json") -> List[Dict[str, Any]]:
    splits = read_json(splits_path)
    templates = set(splits[f"{split}_template_ids"])
    others = set()
    for s in ("train", "val", "test"):
        if s != split:
            others |= set(splits[f"{s}_template_ids"])
    assert not (templates & others), f"{split} templates overlap another split"
    pairs = sorted((p for p in read_jsonl(pairs_path) if p.get("split") == split), key=lambda p: p["pair_id"])
    assert len(pairs) == CFG.n_pairs_per_template * len(templates), f"Unexpected {split} pair count"
    assert all(p["template_id"] in templates for p in pairs)
    assert len({p["pair_id"] for p in pairs}) == len(pairs)
    fams = [p["family"] for p in pairs]
    assert fams.count("A") == fams.count("B") == len(pairs) // 2, f"{split} families must be balanced"
    return pairs


def load_b_ctrl_rows(b_ctrl_path: Path = CFG.data_dir / "b_ctrl.jsonl",
                     test_pairs: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    rows = sorted(read_jsonl(b_ctrl_path), key=lambda r: r["b_ctrl_id"])
    assert len(rows) == CFG.n_b_ctrl, f"Expected {CFG.n_b_ctrl} B-ctrl rows, found {len(rows)}"
    if test_pairs is not None:
        by_id = {p["pair_id"]: p for p in test_pairs}
        b_templates = {p["template_id"] for p in test_pairs if p["family"] == "B"}
        assert {r["template_id"] for r in rows} == b_templates, "B-ctrl templates != Family B test templates"
        for r in rows:
            pair = by_id[r["pair_id"]]
            assert pair["family"] == "B" and pair["template_id"] == r["template_id"]
    return rows


# ----------------------------- model & tensors -------------------------------
def load_stage_model(contract: Dict[str, Any], device: str = "cuda:0", requested_dtype: str = "auto"):
    """Load the Stage 2 locked model + left-padded tokenizer (pad = eos when missing)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from src.model_utils import resolve_runtime_dtype
    dtype, dtype_str, _ = resolve_runtime_dtype(requested_dtype, device)
    print(f"[Load] {contract['model_name']} as {dtype_str} on {device}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(contract["model_name"])
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    raw_model = AutoModelForCausalLM.from_pretrained(
        contract["model_name"], torch_dtype=dtype, device_map=device, low_cpu_mem_usage=True)
    raw_model.eval()
    assert int(raw_model.config.num_hidden_layers) == int(contract["n_layers"])
    assert int(raw_model.config.hidden_size) == int(contract["d_model"])
    return raw_model, tokenizer, dtype_str


def ensure_stage4_directions(directions_path: Path = CFG.stage4_out_path,
                             manifest_path: Path = CFG.stage4_manifest_path,
                             stage0_path: Optional[Path] = CFG.stage0_out_path) -> Tuple[Dict[str, Any], str]:
    """Restore stage4_directions.pt from the HF dataset when absent, verify it, and load it."""
    from src.direction_extraction import verify_stage4_outputs
    directions_path = Path(directions_path)
    if not directions_path.exists():
        repo = os.environ.get("HF_TENSOR_REPO_ID", "")
        if not repo:
            raise FileNotFoundError(f"Missing {directions_path}; set HF_TENSOR_REPO_ID to restore it")
        from scripts.kaggle_sync import ensure_tensor_cached_from_hf
        ensure_tensor_cached_from_hf(directions_path.name, repo, directions_path.parent.parent)
    manifest = verify_stage4_outputs(directions_path, manifest_path, stage0_path)
    directions = torch.load(directions_path, map_location="cpu", weights_only=False)
    return directions, manifest["directions_sha256"]


def load_locked_selection(stage5_path: Path = CFG.stage5_out_path,
                          stage4_sha256: Optional[str] = None) -> Dict[str, Any]:
    sel = read_json(stage5_path)
    assert sel.get("passed_acceptance_gate") is True, "Stage 5 selection did not pass its gate"
    L, K = int(sel["n_layers"]), int(sel["k_suffix"])
    assert 0 <= int(sel["l_star"]) < L and int(sel["i_star"]) in range(-K, 0)
    assert int(sel["pos_idx"]) == list(sel["suffix_indices"]).index(int(sel["i_star"]))
    if stage4_sha256 is not None:
        assert sel["source_stage4_sha256"] == stage4_sha256, "Stage 5 was selected on different Stage 4 directions"
    return sel


def slice_vectors_at_star(directions: Dict[str, Any], l_star: int, pos_idx: int) -> Dict[str, Any]:
    """Learned direction plus the 119 controls at (l*, i*). Unit vectors ablate; scaled vectors steer."""
    l, p = int(l_star), int(pos_idx)
    norm = float(directions["norms"][l, p])
    out = {
        "learned_unit": directions["unit_directions"][l, p].float(),
        "learned_diff": directions["diff_in_means"][l, p].float(),
        "random_unit": directions["random_controls_unit"][:, l, p, :].float(),
        "shuffled_unit": directions["shuffled_controls_unit"][:, l, p, :].float(),
        "shuffled_diff": directions["shuffled_controls_diff"][:, l, p, :].float(),
        "concept_unit": directions["r_concept_unit"][l, p].float(),
        "concept_scaled": directions["r_concept_ortho_scaled"][l, p].float(),
        "norm": norm,
    }
    out["random_scaled"] = out["random_unit"] * norm
    assert out["random_unit"].shape[0] == CFG.n_ctrl_random and out["shuffled_unit"].shape[0] == CFG.n_ctrl_shuffled
    return out


def control_specs(vectors: Dict[str, Any]) -> List[Tuple[str, int, Any, Any]]:
    """(tier, index, unit_vector, steering_vector) for all 119 controls in a fixed order."""
    specs = [("random", m, vectors["random_unit"][m], vectors["random_scaled"][m])
             for m in range(vectors["random_unit"].shape[0])]
    specs += [("shuffled", m, vectors["shuffled_unit"][m], vectors["shuffled_diff"][m])
              for m in range(vectors["shuffled_unit"].shape[0])]
    specs.append(("concept", 0, vectors["concept_unit"], vectors["concept_scaled"]))
    return specs


def build_chat_batches(tokenizer: Any, prompts: List[str], system_prompt: str, suffix_token_ids: List[int],
                       batch_size: int = CFG.batch_size_forward) -> List[Dict[str, Any]]:
    """Left-padded CPU batches with mask-derived position IDs and a verified chat suffix."""
    from src.activation_cache import format_and_verify_batch_inputs
    return [format_and_verify_batch_inputs(tokenizer, prompts[s:s + batch_size], system_prompt,
                                           list(suffix_token_ids), "cpu")
            for s in range(0, len(prompts), batch_size)]


def to_device(batch: Dict[str, Any], device: Any) -> Dict[str, Any]:
    return {k: v.to(device) for k, v in batch.items()}


def model_device(raw_model: Any) -> Any:
    return next(raw_model.parameters()).device


def log_odds_and_p_abs(logits: Any, unknown_ids: Sequence[int]) -> Tuple[Any, Any]:
    probs = F.softmax(logits.float(), dim=-1)
    mass = probs[..., list(dict.fromkeys(map(int, unknown_ids)))].sum(-1).clamp(1e-7, 1 - 1e-7)
    return torch.log(mass) - torch.log1p(-mass), mass


@torch.no_grad()
def evaluate_condition(raw_model: Any, batches: List[Dict[str, Any]], contract: Dict[str, Any],
                       mode: str = "none", vec: Any = None, target_layer: Optional[int] = None,
                       alpha: float = 0.0, return_logits: bool = False) -> Dict[str, np.ndarray]:
    """Per-prompt abstention log-odds S(p) and P_abs(p) (and optional float32 CPU logits)."""
    from src.model_utils import forward_with_native_hooks
    dev = model_device(raw_model)
    s_all, p_all, logits_all = [], [], []
    for b in batches:
        logits = forward_with_native_hooks(raw_model, to_device(b, dev), bool(contract["is_layer_output_tuple"]),
                                           mode, vec, target_layer, alpha)
        s, p = log_odds_and_p_abs(logits, contract["unknown_token_ids"])
        assert torch.isfinite(s).all(), f"Non-finite log-odds under mode={mode}"
        s_all.append(s.double().cpu())
        p_all.append(p.double().cpu())
        if return_logits:
            logits_all.append(logits.float().cpu())
    out = {"log_odds": torch.cat(s_all).numpy(), "p_abs": torch.cat(p_all).numpy()}
    if return_logits:
        out["logits"] = torch.cat(logits_all)
    return out


@torch.no_grad()
def capture_residuals_and_logits(raw_model: Any, batches: List[Dict[str, Any]], contract: Dict[str, Any],
                                 layers: Sequence[int], pos: int) -> Tuple[Any, Any]:
    """Unsteered pass: ([N, len(layers), d_model] float32 residuals at `pos`, [N, vocab] float32 logits)."""
    from src.model_utils import last_token_logits_kwargs, native_intervention_hooks
    dev = model_device(raw_model)
    acts, logits_all = [], []
    for b in batches:
        captured: Dict[int, Any] = {}
        with native_intervention_hooks(raw_model, bool(contract["is_layer_output_tuple"]), "none",
                                       capture_layers=list(layers), capture_pos=int(pos), captured=captured):
            out = raw_model(**to_device(b, dev), use_cache=False, **last_token_logits_kwargs(raw_model))
        assert set(captured) == set(layers), "Residual capture hook did not fire on every layer"
        acts.append(torch.stack([captured[l].cpu() for l in layers], dim=1))
        logits_all.append(out.logits[:, -1, :].float().cpu())
    result = torch.cat(acts), torch.cat(logits_all)
    assert torch.isfinite(result[0]).all() and torch.isfinite(result[1]).all()
    return result


@torch.no_grad()
def generate_with_hooks(raw_model: Any, tokenizer: Any, batches: List[Dict[str, Any]], contract: Dict[str, Any],
                        mode: str = "none", vec: Any = None, target_layer: Optional[int] = None,
                        alpha: float = 0.0, max_new_tokens: int = CFG.max_new_tokens) -> List[str]:
    """Greedy decoding with the intervention active at every position (prompt and generated tokens)."""
    from src.model_utils import native_intervention_hooks
    dev = model_device(raw_model)
    texts: List[str] = []
    for b in batches:
        enc = {k: b[k].to(dev) for k in ("input_ids", "attention_mask")}
        with native_intervention_hooks(raw_model, bool(contract["is_layer_output_tuple"]), mode, vec,
                                       target_layer, alpha):
            ids = raw_model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, num_beams=1,
                                     temperature=None, top_p=None, top_k=None, repetition_penalty=1.0,
                                     pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
        texts.extend(tokenizer.batch_decode(ids[:, enc["input_ids"].shape[1]:], skip_special_tokens=True))
    return texts


@torch.no_grad()
def verify_native_hooks_match_nnsight(raw_model: Any, tokenizer: Any, batch: Dict[str, Any],
                                      contract: Dict[str, Any], unit_vec: Any, steer_vec: Any,
                                      target_layer: int, atol: float = 0.05) -> Dict[str, float]:
    """Cross-check native hooks against the Stage 0-verified nnsight path on one batch.

    `atol` allows a few fp16 ULPs because native passes run lm_head on the final position only.
    """
    from nnsight import LanguageModel
    from src.model_utils import forward_with_intervention, forward_with_native_hooks
    llm = LanguageModel(raw_model, tokenizer=tokenizer, dispatch=True)
    inputs = to_device(batch, model_device(raw_model))
    is_tuple = bool(contract["is_layer_output_tuple"])
    diffs = {}
    for mode, vec, layer, alpha in (("none", None, None, 0.0), ("ablate_all", unit_vec, None, 0.0),
                                    ("steer_layer", steer_vec, target_layer, 1.0)):
        a = forward_with_native_hooks(raw_model, inputs, is_tuple, mode, vec, layer, alpha).float()
        b = forward_with_intervention(llm, is_tuple, inputs, mode, vec, layer, alpha).float()
        diffs[mode] = float((a - b).abs().max().item())
        assert diffs[mode] <= atol, f"Native hook {mode} differs from nnsight by {diffs[mode]:.3e}"
    del llm
    return diffs


# ----------------------------- checkpoints -----------------------------------
class ConditionCheckpoint:
    """Resumable dict of finished condition results, invalidated when the fingerprint changes."""

    def __init__(self, path: Optional[Path], fingerprint: str, force: bool = False, every: int = 10):
        self.path = Path(path) if path else None
        self.fingerprint = fingerprint
        self.results: Dict[str, Any] = {}
        self.every = max(1, int(every))
        self._dirty = 0
        if self.path and self.path.exists() and not force:
            cp = torch.load(self.path, map_location="cpu", weights_only=False)
            if cp.get("fingerprint") == fingerprint:
                self.results = dict(cp.get("results", {}))
                print(f"[Checkpoint] Resuming {len(self.results)} finished conditions from {self.path}", flush=True)

    def __contains__(self, key: str) -> bool:
        return key in self.results

    def get(self, key: str) -> Any:
        return self.results[key]

    def put(self, key: str, value: Any) -> None:
        self.results[key] = value
        self._dirty += 1
        if self._dirty >= self.every:
            self.flush()

    def flush(self) -> None:
        if not self.path or not self._dirty:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        torch.save({"fingerprint": self.fingerprint, "results": self.results}, tmp)
        tmp.replace(self.path)
        self._dirty = 0

    def remove(self) -> None:
        if self.path and self.path.exists():
            self.path.unlink()
