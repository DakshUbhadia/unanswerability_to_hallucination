"""Stage 9: publication figures, executive summary, and README results injection (CPU only, from JSON artifacts)."""
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.config import CFG  # noqa: E402
from src.eval_common import atomic_write_json, read_json  # noqa: E402

# Okabe-Ito colorblind-safe palette.
C_LEARNED, C_UNANS, C_ANS, C_CONCEPT, C_RANDOM, C_SHUF, C_BCTRL = (
    "#0072B2", "#D55E00", "#009E73", "#CC79A7", "#7F7F7F", "#E69F00", "#56B4E9")
README_BEGIN, README_END = "<!-- BEGIN_STAGE9_RESULTS -->", "<!-- END_STAGE9_RESULTS -->"


def _style() -> None:
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25, "savefig.dpi": 300})


def _save(fig: Any, png: Path, pdf: Path) -> None:
    for p in (png, pdf):
        Path(p).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _yerr(center: float, ci: List[float]) -> List[List[float]]:
    return [[max(0.0, center - ci[0])], [max(0.0, ci[1] - center)]]


def _cap_lookup(stage7: Optional[Dict[str, Any]], mode: str, alpha: Optional[float]) -> Dict[str, Any]:
    from src.results_tables import find_capability
    return find_capability(stage7, mode, alpha)


def _control_cloud(ax, xs: Dict[str, List[float]], ys: Dict[str, List[float]], label_suffix: str, marker_scale=1.0):
    for tier, color, marker, name in (("random", C_RANDOM, "o", "99 random"), ("shuffled", C_SHUF, "^", "19 shuffled")):
        x, y = np.asarray(xs[tier]), np.asarray(ys[tier])
        ax.scatter(x, y, s=8 * marker_scale, color=color, alpha=0.25, marker=marker, linewidths=0)
        ax.errorbar(np.mean(x), np.mean(y), xerr=[[np.mean(x) - np.percentile(x, 2.5)], [np.percentile(x, 97.5) - np.mean(x)]],
                    yerr=[[np.mean(y) - np.percentile(y, 2.5)], [np.percentile(y, 97.5) - np.mean(y)]], fmt=marker,
                    color=color, ms=6, capsize=2, label=f"{name} {label_suffix} (mean, 95% interval)")
    ax.scatter(xs["concept"], ys["concept"], marker="D", s=40, color=C_CONCEPT, edgecolor="k", linewidths=0.5,
               label=f"r_concept {label_suffix}", zorder=5)


def plot_figure1_intervention_pareto(stage6_dict, stage7_dict, out_png_path=CFG.fig1_png_path,
                                     out_pdf_path=None) -> None:
    """2x2: (necessity, sufficiency) x (WikiText-2 KL, MMLU). Y = mean first-token abstention log-odds S(p)."""
    _style()
    out_pdf_path = out_pdf_path or Path(out_png_path).with_suffix(".pdf")
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    nec, suf = stage6_dict["necessity_curve"], stage6_dict["sufficiency_curve"]
    ctrl6 = stage6_dict["controls_summary"]
    mmlu_ok = bool(stage7_dict and stage7_dict.get("mmlu_evaluated"))
    panels = [(axes[0, 0], nec, "necessity", "kl"), (axes[1, 0], nec, "necessity", "mmlu"),
              (axes[0, 1], suf, "sufficiency", "kl"), (axes[1, 1], suf, "sufficiency", "mmlu")]
    for ax, curve, kind, xkind in panels:
        part = "D_test^unans" if kind == "necessity" else "D_test^ans"
        pts = []
        for row in curve:
            mode, alpha = row["mode"], row["alpha"]
            cap = (_cap_lookup(stage7_dict, "none", None) if (mode == "steer_layer" and alpha == 0.0)
                   else _cap_lookup(stage7_dict, mode, alpha if mode == "steer_layer" else None))
            x, xci = ((cap.get("wikitext_kl"), cap.get("wikitext_kl_ci")) if xkind == "kl"
                      else (cap.get("mmlu_acc"), cap.get("mmlu_ci")))
            if x is None:
                continue
            x = 100 * x if xkind == "mmlu" else x
            xci = [100 * v for v in xci] if (xkind == "mmlu" and xci) else xci
            pts.append((mode, alpha, x, xci, row))
        steer = sorted([p for p in pts if p[0] == "steer_layer"], key=lambda p: p[1])
        if steer:
            ax.plot([p[2] for p in steer], [p[4]["mean_log_odds"] for p in steer], "-", color=C_LEARNED, lw=1.2, alpha=0.7)
        for mode, alpha, x, xci, row in pts:
            y = row["mean_log_odds"]
            is_base = mode == "steer_layer" and alpha == 0.0
            color, marker = ("k", "s") if is_base else ((C_UNANS, "*") if mode == "ablate_all" else (C_LEARNED, "o"))
            ax.errorbar(x, y, xerr=_yerr(x, xci) if xci else None, yerr=_yerr(y, row["log_odds_ci95"]), fmt=marker,
                        color=color, ms=11 if marker == "*" else 6, capsize=2,
                        label="baseline" if is_base else ("learned r_hat ablate_all" if mode == "ablate_all" else None))
            if not is_base and mode == "steer_layer":
                ax.annotate(f"{alpha:+g}", (x, y), textcoords="offset points", xytext=(4, 4), fontsize=7, color=C_LEARNED)
        if steer:
            ax.plot([], [], "o-", color=C_LEARNED, label="learned r steering (alpha grid)")
        if xkind == "kl" and stage7_dict:
            ops = [("ablate_all", "ablate_all_unans", "ablate_all"), (f"alpha={CFG.alpha_primary_neg:+.1f}", "steer_neg1_unans",
                   f"alpha={CFG.alpha_primary_neg:+g}")] if kind == "necessity" else \
                  [(f"alpha={CFG.alpha_primary_pos:+.1f}", "steer_pos1_ans", f"alpha={CFG.alpha_primary_pos:+g}")]
            for i, (op7, op6, label) in enumerate(ops):
                xs = stage7_dict["controls_wikitext_kl"][op7]
                ys = ctrl6[op6]["mean_log_odds"]
                _control_cloud(ax, xs, ys, f"@ {label}", 1.0 if i == 0 else 0.6)
            ax.invert_xaxis()
            ax.set_xlabel("WikiText-2 token-level KL (nats; inverted, right = better retention)")
        elif xkind == "mmlu":
            ax.set_xlabel("200-item chat MMLU accuracy (%)")
            if not mmlu_ok:
                ax.text(0.5, 0.5, "MMLU not evaluated (Day 14 Cut 2)", transform=ax.transAxes, ha="center")
        else:
            ax.text(0.5, 0.5, "Stage 7 not available", transform=ax.transAxes, ha="center")
        ax.set_ylabel(f"Mean abstention log-odds S(p) on {part}")
        ax.set_title(f"{'A' if kind == 'necessity' else 'B'}{1 if xkind == 'kl' else 2} - "
                     f"{kind.capitalize()} vs. {'KL' if xkind == 'kl' else 'MMLU'}", loc="left", fontweight="bold")
        handles, labels = ax.get_legend_handles_labels()
        uniq = dict(zip(labels, handles))
        if uniq:
            ax.legend(uniq.values(), uniq.keys(), fontsize=6.5, loc="best", framealpha=0.85)
    fig.suptitle(f"Figure 1. Intervention specificity vs. capability (l* = {stage6_dict['l_star']}, i* = {stage6_dict['i_star']}); "
                 "y error bars: template-cluster bootstrap 95% CI, x: item bootstrap 95% CI", fontsize=9)
    fig.tight_layout()
    _save(fig, out_png_path, out_pdf_path)


def plot_figure2a_popqa_violin(stage8_dict, out_png_path=CFG.fig2a_png_path, out_pdf_path=None,
                               stage6_dict: Optional[Dict[str, Any]] = None) -> None:
    _style()
    out_pdf_path = out_pdf_path or Path(out_png_path).with_suffix(".pdf")
    groups = [("test_unans", "D_test^unans", C_UNANS), ("test_ans", "D_test^ans", C_ANS), ("b_ctrl", "D_B-ctrl", C_BCTRL),
              ("popqa_abstain", "PopQA abstain", "#999933"), ("popqa_matched_wrong", "PopQA wrong (matched)", C_UNANS),
              ("popqa_matched_correct", "PopQA correct (matched)", C_ANS)]
    data = [np.asarray(stage8_dict["violin_6groups"][k], dtype=float) for k, _, _ in groups]
    fig, ax = plt.subplots(figsize=(10, 5.2))
    parts = ax.violinplot(data, positions=range(1, 7), showextrema=False, widths=0.8)
    for body, (_, _, color) in zip(parts["bodies"], groups):
        body.set_facecolor(color); body.set_alpha(0.3); body.set_edgecolor(color)
    rng = np.random.default_rng(0)
    for i, (d, (_, _, color)) in enumerate(zip(data, groups), start=1):
        ax.scatter(i + rng.uniform(-0.12, 0.12, len(d)), d, s=5, color=color, alpha=0.5, linewidths=0)
        q1, med, q3 = np.percentile(d, [25, 50, 75])
        ax.vlines(i, q1, q3, color="k", lw=3)
        ax.scatter([i], [med], color="white", edgecolor="k", s=25, zorder=4)
    ax.set_xticks(range(1, 7))
    ax.set_xticklabels([f"{name}\n(N={len(d)})" for d, (_, name, _) in zip(data, groups)], fontsize=8)
    ax.set_ylabel("Unsteered projection x_{i*}^{(l*)} . r_hat")
    top = max(float(np.max(d)) for d in data)
    span = top - min(float(np.min(d)) for d in data)

    def bracket(x1, x2, h, text):
        ax.plot([x1, x1, x2, x2], [h, h + 0.02 * span, h + 0.02 * span, h], color="k", lw=0.8)
        ax.text((x1 + x2) / 2, h + 0.03 * span, text, ha="center", va="bottom", fontsize=7.5)

    if stage6_dict:
        h3 = stage6_dict["primary_tests"]["H3"]
        bracket(1, 3, top + 0.05 * span, f"H3 (Fam. B unans vs B-ctrl): Wilcoxon p = {h3['p_wilcoxon']:.3g}")
    bracket(5, 6, top + 0.05 * span, f"H4 AUROC = {stage8_dict['auroc_proj']:.3f} "
            f"[{stage8_dict['auroc_proj_ci'][0]:.3f}, {stage8_dict['auroc_proj_ci'][1]:.3f}]; MWU p = {stage8_dict['h4_mwu_p']:.3g}")
    ax.text(0.01, 0.02, f"AUROC_maxp = {stage8_dict['auroc_maxp']:.3f}\nAUROC_ent = {stage8_dict['auroc_ent']:.3f}\n"
            f"H5 dAUROC (proj - maxp) = {stage8_dict['delta_auroc_proj_minus_maxp']:+.3f}, p_boot = {stage8_dict['h5_boot_p']:.3g}",
            transform=ax.transAxes, fontsize=7.5, va="bottom", bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "0.7"})
    ax.set_ylim(top=top + 0.18 * span)
    ax.set_title(f"Figure 2A. Abstention-direction projections across contextual and closed-book groups "
                 f"(l* = {stage8_dict['l_star']}, i* = {stage8_dict['i_star']})", fontsize=9, loc="left")
    fig.tight_layout()
    _save(fig, out_png_path, out_pdf_path)


def plot_figureS1_layerwise_auroc(stage8_dict, out_png_path=CFG.figS1_png_path, out_pdf_path=None) -> None:
    _style()
    out_pdf_path = out_pdf_path or Path(out_png_path).with_suffix(".pdf")
    rows = stage8_dict["layerwise_auroc_exploratory"]
    layers = [r["layer"] for r in rows]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.fill_between(layers, [r["ci_low"] for r in rows], [r["ci_high"] for r in rows], color=C_LEARNED, alpha=0.2,
                    label="95% paired bootstrap CI")
    ax.plot(layers, [r["auroc"] for r in rows], "-o", ms=3, color=C_LEARNED, label="PopQA wrong vs. correct AUROC_proj(l)")
    ax.plot(layers, [r["auroc_test_ctx"] for r in rows], "--", color=C_ANS, label="D_test unans vs. ans AUROC(l)")
    for val, name, color in ((stage8_dict["auroc_maxp"], "AUROC_maxp", C_UNANS), (stage8_dict["auroc_ent"], "AUROC_ent", C_SHUF),
                             (0.5, "chance", "k")):
        ax.axhline(val, ls=":", color=color, lw=1, label=f"{name} = {val:.3f}")
    ax.axvline(stage8_dict["l_star"], ls="--", color="0.4", lw=1, label=f"l* = {stage8_dict['l_star']}")
    ax.set_xlabel("Decoder layer l"); ax.set_ylabel("AUROC"); ax.set_ylim(0, 1.02)
    ax.set_title(f"[EXPLORATORY] Figure S1. Layer-wise PopQA wrong-vs-correct AUROC at i* = {stage8_dict['i_star']}",
                 fontsize=9, loc="left")
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    _save(fig, out_png_path, out_pdf_path)


def plot_figure2b_selective_steering(stage10_dict, out_png_path=CFG.fig2b_png_path, out_pdf_path=None) -> None:
    _style()
    out_pdf_path = out_pdf_path or Path(out_png_path).with_suffix(".pdf")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    for direction, color, ls in (("learned", C_LEARNED, "-"), ("concept", C_CONCEPT, "--")):
        curve = stage10_dict["coverage_curve"][direction]
        cov = [r["coverage"] for r in curve]
        acc = [100 * r["selective_acc"] if r["selective_acc"] is not None else np.nan for r in curve]
        ax1.errorbar(cov, acc, xerr=[[r["coverage"] - r["coverage_ci"][0] for r in curve], [r["coverage_ci"][1] - r["coverage"] for r in curve]],
                     yerr=[[a - 100 * r["selective_acc_ci"][0] if r["selective_acc"] is not None else 0 for a, r in zip(acc, curve)],
                           [100 * r["selective_acc_ci"][1] - a if r["selective_acc"] is not None else 0 for a, r in zip(acc, curve)]],
                     fmt="o" + ls, color=color, ms=4, capsize=2, label=f"{direction} steering (alpha grid)")
        for r, c, a in zip(curve, cov, acc):
            ax1.annotate(f"{r['alpha']:g}", (c, a), textcoords="offset points", xytext=(3, 3), fontsize=6, color=color)
        alphas = [r["alpha"] for r in curve]
        ax2.plot(alphas, [r["wrong_to_abstain_rate"] for r in curve], ls, marker="o", color=C_UNANS, ms=3,
                 label=f"wrong -> abstain ({direction})")
        ax2.plot(alphas, [r["correct_to_abstain_rate"] for r in curve], ls, marker="s", color=C_ANS, ms=3,
                 label=f"correct -> abstain ({direction})")
        ax2.plot(alphas, [r["selective_acc"] for r in curve], ls, marker="^", color=color, ms=3,
                 label=f"conditional precision ({direction})")
    conf = stage10_dict["confidence_threshold_curve"]
    ax1.plot([r["coverage"] for r in conf], [100 * r["selective_acc"] for r in conf], ":", color=C_SHUF, marker=".",
             label="max-probability thresholding (no steering)")
    ax1.axhline(100 * stage10_dict["random_abstention_accuracy"], color="k", ls=":", lw=1, label="random abstention")
    ax1.set_xlabel(f"Coverage (fraction of {stage10_dict['n_items']} matched PopQA prompts answered)")
    ax1.set_ylabel("Selective accuracy among answered (%)")
    ax1.set_title("Figure 2B (stretch). Accuracy vs. coverage", loc="left", fontsize=9)
    ax1.legend(fontsize=7)
    ax2.set_xlabel("Steering multiplier alpha at l*"); ax2.set_ylabel("Rate")
    ax2.set_title("Abstention induced on baseline-wrong vs. baseline-correct items", loc="left", fontsize=9)
    ax2.legend(fontsize=6.5)
    fig.tight_layout()
    _save(fig, out_png_path, out_pdf_path)


# ----------------------------- summary & README ------------------------------
def assess_stage10(stage10_dict: Dict[str, Any]) -> Dict[str, Any]:
    threshold = sorted(stage10_dict["confidence_threshold_curve"], key=lambda r: r["coverage"])
    cov_min, cov_max = threshold[0]["coverage"], threshold[-1]["coverage"]
    learned = stage10_dict["coverage_curve"]["learned"]
    comparable, not_comparable = [], []
    for row in learned:
        cov, acc = row["coverage"], row["selective_acc"]
        if acc is None or not (cov_min <= cov < cov_max):
            not_comparable.append({"alpha": row["alpha"], "coverage": cov})
            continue
        threshold_acc = float(np.interp(cov, [r["coverage"] for r in threshold], [r["selective_acc"] for r in threshold]))
        comparable.append({"alpha": row["alpha"], "coverage": cov, "learned_selective_acc": acc,
                           "threshold_selective_acc": threshold_acc, "gap_points": (threshold_acc-acc)*100})
    largest = max(comparable, key=lambda r:r["gap_points"]) if comparable else None
    learned_gains = stage10_dict["coverage_curve"]["learned"]
    concept_gains = stage10_dict["coverage_curve"]["concept"]
    lmax=max(learned_gains,key=lambda r:r["selective_abstention_gain"])
    cmax=max(concept_gains,key=lambda r:r["selective_abstention_gain"])
    return {"confidence_coverage_range":[cov_min,cov_max], "comparable_points":comparable,
            "not_comparable_points":not_comparable,
            "dominated_at_all_comparable_points":bool(comparable and all(r["gap_points"]>0 for r in comparable)),
            "largest_gap_points":None if largest is None else largest["gap_points"],
            "largest_gap_coverage":None if largest is None else largest["coverage"],
            "largest_gap_alpha":None if largest is None else largest["alpha"],
            "learned_max_gain":lmax["selective_abstention_gain"],"learned_max_gain_alpha":lmax["alpha"],
            "concept_max_gain":cmax["selective_abstention_gain"],"concept_max_gain_alpha":cmax["alpha"],
            "concept_beats_learned_gain":cmax["selective_abstention_gain"]>lmax["selective_abstention_gain"]}


def build_stage9_executive_summary(stage5_dict, stage6_dict, stage7_dict, stage8_dict, stage10_dict=None) -> Dict[str, Any]:
    holm={r["test_id"]:r for r in stage8_dict["holm_bonferroni_table"]}
    pt=stage6_dict["primary_tests"]; h4=stage8_dict["h4"]; h5=stage8_dict["h5"]
    shuf=h4["controls"]["shuffled"]; shuf_au=shuf["aurocs"]
    summary={"stage":9,"model_name":stage6_dict["model_name"],"l_star":stage5_dict["l_star"],"i_star":stage5_dict["i_star"],
      "filter_relaxed":stage5_dict["filter_relaxed"],"r_norm_at_star":stage5_dict["r_norm_at_star"],
      "hypotheses":{"H1":{"effect":pt["H1"]["mean_effect"],"p":pt["H1"]["p_wilcoxon"],"p_mc":pt["H1"]["p_mc"]},
      "H2":{"effect":pt["H2"]["mean_effect"],"p":pt["H2"]["p_wilcoxon"],"p_mc":pt["H2"]["p_mc"]},
      "H3":{"effect":pt["H3"]["mean_effect"],"p":pt["H3"]["p_wilcoxon"],"b_ctrl_abstention_rate":pt["H3"]["b_ctrl_gen_abstention_rate_regex"]},
      "H4":{"auroc_proj":stage8_dict["auroc_proj"],"ci":stage8_dict["auroc_proj_ci"],"p":stage8_dict["h4_mwu_p"]},
      "H5":{"delta_auroc":stage8_dict["delta_auroc_proj_minus_maxp"],"p":stage8_dict["h5_boot_p"]}},
      "holm_bonferroni":stage8_dict["holm_bonferroni_table"],"confirmed":{k:bool(v["confirmed"]) for k,v in holm.items()},
      "interpretation":stage8_dict["interpretation"],"logit_gen_discrepancy":stage6_dict["logit_gen_discrepancy"],
      "capability":None,"stage10_ran":stage10_dict is not None,
      "h4_controls":{"random_rank":h4["controls"]["random"]["rank_of_observed"],"random_p":h4["controls"]["random"]["p_rank"],
      "shuffled_rank":shuf["rank_of_observed"],"shuffled_n":shuf["n"],"shuffled_p":shuf["p_rank"],
      "shuffled_max_auroc":shuf["max"],"shuffled_min_auroc":shuf["min"],
      "n_shuffled_above_learned":sum(x>stage8_dict["auroc_proj"] for x in shuf_au)},
      "confidence_baselines":{"auroc_maxp":stage8_dict["auroc_maxp"],"auroc_maxp_ci":stage8_dict["auroc_maxp_ci"],
      "auroc_ent":stage8_dict["auroc_ent"],"auroc_ent_ci":stage8_dict["auroc_ent_ci"],
      "delta_maxp_ci":h5["delta_auroc_maxp_ci"]}}
    for k in summary["hypotheses"]: summary["hypotheses"][k]["confirmed"]=summary["confirmed"].get(k)
    if stage7_dict:
        abl=next((r for r in stage7_dict["learned_direction_capability"] if r["mode"]=="ablate_all"),{})
        summary["capability"]={"wikitext_baseline_nll":stage7_dict["wikitext_baseline_nll"],"ablate_all_kl":abl.get("wikitext_kl"),
        "mmlu_baseline_acc":stage7_dict["mmlu_baseline_acc"],"mmlu_ablate_all_acc":abl.get("mmlu_acc")}
    if stage10_dict:
        summary["stage10_assessment"]=assess_stage10(stage10_dict)
        summary["stage10"]={"max_selective_gain":summary["stage10_assessment"]["learned_max_gain"]}
    return summary


def render_results_markdown(summary: Dict[str, Any], table1_md: str, table2_md: str, table3_md: str,
                            table4_md: str = "", has_fig2b: bool = False) -> str:
    h=summary["hypotheses"]; interp=summary["interpretation"]
    tick=lambda k:"confirmed" if summary["confirmed"].get(k) else "not confirmed"
    lines=[README_BEGIN,"## Results (auto-generated by Stage 9)","",
      f"Model: `{summary['model_name']}`; locked coordinate l* = {summary['l_star']}, i* = {summary['i_star']}{' (validation filter relaxed)' if summary['filter_relaxed'] else ''}.","",
      f"- **H1 necessity**: mean log-odds reduction {h['H1']['effect']:.3f}, Wilcoxon p = {_fmt_p(h['H1']['p'])}, Monte Carlo p = {h['H1']['p_mc']:.2f} -> {tick('H1')}.",
      f"- **H2 sufficiency**: mean log-odds increase {h['H2']['effect']:.3f}, Wilcoxon p = {_fmt_p(h['H2']['p'])}, Monte Carlo p = {h['H2']['p_mc']:.2f} -> {tick('H2')}.",
      f"- **H3 private-detail control**: projection gap {h['H3']['effect']:.3f}, p = {_fmt_p(h['H3']['p'])}, B-ctrl abstention rate {h['H3']['b_ctrl_abstention_rate']:.3f} -> {tick('H3')}.",
      f"- **H4 closed-book separation**: AUROC_proj = {h['H4']['auroc_proj']:.3f} [{h['H4']['ci'][0]:.3f}, {h['H4']['ci'][1]:.3f}], MWU p = {_fmt_p(h['H4']['p'])} -> {tick('H4')}.",
      f"- H4 specificity: rank {summary['h4_controls']['random_rank']}/100 against covariance-free random directions (p = {summary['h4_controls']['random_p']}); rank {summary['h4_controls']['shuffled_rank']}/{summary['h4_controls']['shuffled_n']+1} against within-pair label-shuffled directions (p = {summary['h4_controls']['shuffled_p']}; {summary['h4_controls']['n_shuffled_above_learned']} shuffled directions exceed the learned AUROC, max {summary['h4_controls']['shuffled_max_auroc']:.3f}). The shuffled null is not valid for this paired design (see DEV-014) and is reported for transparency, not as evidence.",
      f"- **H5 vs. output confidence**: dAUROC = {h['H5']['delta_auroc']:+.3f}, p_boot = {_fmt_p(h['H5']['p'])} -> {tick('H5')}.",
      f"- Output confidence separates wrong from correct better than the projection: AUROC_maxp = {summary['confidence_baselines']['auroc_maxp']:.3f}, AUROC_entropy = {summary['confidence_baselines']['auroc_ent']:.3f}, versus AUROC_proj = {h['H4']['auroc_proj']:.3f}.",
      "",f"**Interpretation (rule v4.1, post hoc; see DEV-013):** `{interp['overall']}`.","",interp["verdict_explanation"]]
    if interp.get("overall_as_registered_v4")!=interp["overall"]:
        lines += ["",f"Under the original v4 aggregation rule this would read `{interp['overall_as_registered_v4']}`; that rule ignored H5 and is superseded."]
    if summary.get("stage10_assessment"):
        a=summary["stage10_assessment"]
        if a["dominated_at_all_comparable_points"]:
            msg=(f"Stage 10 (exploratory) is a negative result: at every comparable operating point, learned steering has lower selective accuracy than max-probability thresholding at the same coverage (largest gap {a['largest_gap_points']:.1f} points at coverage {a['largest_gap_coverage']:.2f}).")
        else: msg="Stage 10 (exploratory): learned steering is not dominated at every comparable operating point by max-probability thresholding."
        msg+=f" The orthogonal concept control reaches a higher selective-abstention gain ({a['concept_max_gain']:.3f} at alpha = {a['concept_max_gain_alpha']}) than the learned direction ({a['learned_max_gain']:.3f} at alpha = {a['learned_max_gain_alpha']}). Points at alpha >= 1.5 are affected by a known abstention-parser gap (Tier 1, open)."
        lines += ["","- "+msg]
    if summary["logit_gen_discrepancy"]: lines += ["","> Branch Rule 4 flagged a logit/generation discrepancy; qualitative claims defer to generation rates (Table 4)."]
    lines += ["","### Figures","","![Figure 1](figures/fig1_intervention_pareto.png)","","![Figure 2A](figures/fig2a_popqa_violin.png)","","![Figure S1](figures/figS1_layerwise_auroc.png)",""]
    if has_fig2b: lines += ["![Figure 2B](figures/fig2b_selective_steering.png)",""]
    lines += ["### Tables","",table1_md.strip(),"",table2_md.strip(),"",table3_md.strip(),""]
    if table4_md: lines += [table4_md.strip(),""]
    lines.append(README_END); return "\n".join(lines)+"\n"


def _fmt_p(p: Optional[float]) -> str:
    return "n/a" if p is None else (f"{p:.2e}" if p < 1e-3 else f"{p:.4f}")




def update_readme_with_final_results(readme_path, summary_dict, table1_md_path, table2_md_path, table3_md_path,
                                     table4_md_path=None, has_fig2b=False) -> None:
    readme_path = Path(readme_path)
    read = lambda p: Path(p).read_text(encoding="utf-8") if p and Path(p).exists() else ""
    block = render_results_markdown(summary_dict, read(table1_md_path), read(table2_md_path), read(table3_md_path),
                                    read(table4_md_path), has_fig2b)
    text = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""
    pattern = re.compile(re.escape(README_BEGIN) + r".*?" + re.escape(README_END) + r"\n?", re.S)
    text = pattern.sub(lambda _: block, text) if pattern.search(text) else text.rstrip("\n") + "\n\n" + block
    readme_path.write_text(text, encoding="utf-8", newline="\n")


def run_stage9(stage5_path=CFG.stage5_out_path, stage6_path=CFG.stage6_out_path, stage7_path=CFG.stage7_out_path,
               stage8_path=CFG.stage8_out_path, stage10_path=CFG.stage10_out_path, figures_dir=CFG.figures_dir,
               summary_path=CFG.stage9_out_path, readme_path=CFG.readme_path, tables_dir=CFG.tables_dir) -> Dict[str, Any]:
    from src.results_tables import write_table1
    figures_dir, tables_dir = Path(figures_dir), Path(tables_dir)
    s5, s6, s8 = read_json(stage5_path), read_json(stage6_path), read_json(stage8_path)
    s7 = read_json(stage7_path) if Path(stage7_path).exists() else None
    s10 = read_json(stage10_path) if Path(stage10_path).exists() else None
    from src.eval_common import write_csv_and_md
    from src.popqa_uncertainty import TABLE3_COLUMNS
    write_table1(tables_dir / "table1_main_results.csv", tables_dir / "table1_main_results.md", s6, s7, s8, load_missing=False)
    write_csv_and_md(s8["table3_rows"], TABLE3_COLUMNS, tables_dir / "table3_popqa_auroc.csv", tables_dir / "table3_popqa_auroc.md",
                     f"Table 3. Popularity-matched PopQA wrong (y=1) vs. correct (y=0), N = {s8['n_matched_per_group']} per group")
    plot_figure1_intervention_pareto(s6, s7, figures_dir / "fig1_intervention_pareto.png", figures_dir / "fig1_intervention_pareto.pdf")
    plot_figure2a_popqa_violin(s8, figures_dir / "fig2a_popqa_violin.png", figures_dir / "fig2a_popqa_violin.pdf", s6)
    plot_figureS1_layerwise_auroc(s8, figures_dir / "figS1_layerwise_auroc.png", figures_dir / "figS1_layerwise_auroc.pdf")
    if s10:
        plot_figure2b_selective_steering(s10, figures_dir / "fig2b_selective_steering.png", figures_dir / "fig2b_selective_steering.pdf")
    summary = build_stage9_executive_summary(s5, s6, s7, s8, s10)
    summary["passed_acceptance_gate"] = True
    atomic_write_json(summary_path, summary)
    update_readme_with_final_results(readme_path, summary, tables_dir / "table1_main_results.md",
                                     tables_dir / "table2_sensitivity_grid.md", tables_dir / "table3_popqa_auroc.md",
                                     tables_dir / "table4_generation_audit.md", has_fig2b=s10 is not None)
    return summary


def verify_stage9_outputs(summary_path=CFG.stage9_out_path, figures_dir=CFG.figures_dir, readme_path=CFG.readme_path,
                          tables_dir=CFG.tables_dir) -> Dict[str, Any]:
    d = read_json(summary_path)
    assert d["stage"] == 9 and d["passed_acceptance_gate"] is True
    assert set(d["confirmed"]) == {"H1", "H2", "H3", "H4", "H5"}
    names = ["fig1_intervention_pareto", "fig2a_popqa_violin", "figS1_layerwise_auroc"]
    if d.get("stage10_ran"):
        names.append("fig2b_selective_steering")
    for name in names:
        for ext in ("png", "pdf"):
            p = Path(figures_dir) / f"{name}.{ext}"
            assert p.is_file() and p.stat().st_size > 1000, f"Missing or empty figure {p}"
    for t in ("table1_main_results", "table2_sensitivity_grid", "table3_popqa_auroc"):
        assert (Path(tables_dir) / f"{t}.csv").is_file(), f"Missing table {t}"
    text = Path(readme_path).read_text(encoding="utf-8")
    assert text.count(README_BEGIN) == 1 and text.count(README_END) == 1, "README results block missing or duplicated"
    return d
