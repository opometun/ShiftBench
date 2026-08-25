"""Correlate pre-training shift metrics against downstream model performance.

This is the study's headline analysis: for each shift metric, how well does the
value computed *before* training predict the mIoU obtained *after* training?

Inputs
    results/shift/distances.json                     from run_shift_metrics.py
    output/<mix>/<model>/test_eval/summary.json      from evaluate_test.sbatch

Sign convention matters and is handled explicitly. Every metric here except
SADGE is a *distance*: larger means the training mixture is further from the
inference distribution, so a metric that predicts utility well should correlate
NEGATIVELY with mIoU. SADGE is a *similarity* (higher means more alike), so it
should correlate positively. The 'predictive' column applies the expected sign
so that metrics can be ranked on a single scale, and 'raw_spearman' keeps the
unadjusted value.

Spearman is the primary statistic: the deck asks which metrics *rank* mixtures
correctly, and rank correlation does not assume the shift/performance
relationship is linear. Pearson is reported alongside.

Caveat, printed with the results: n = 9 mixtures. Correlations from nine points
have wide confidence intervals, and p-values should be read as weak evidence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scipy import stats

# Metrics where a HIGHER value means MORE similar (so positive correlation with
# mIoU is the good outcome). Everything else is a distance.
SIMILARITY_METRICS = {"sadge", "sadge_appearance", "sadge_geometry"}

# Matched-ratio pairs for the within-ratio test. Each entry is
# (synthetic share, gta mixture, synscapes mixture): the same amount of
# synthetic data from two different sources, so a metric's job is to say which
# source is closer to the inference distribution.
MATCHED_PAIRS = [
    ("25%", "cityscapes75_gta25", "cityscapes75_synscapes25"),
    ("50%", "cityscapes50_gta50", "cityscapes50_synscapes50"),
    ("75%", "cityscapes25_gta75", "cityscapes25_synscapes75"),
    ("100%", "gta100", "synscapes100"),
]

# mIoU differences below this are treated as ties rather than as a direction a
# metric can get right or wrong. The 50% pair differs by 0.0009, well inside
# run-to-run variation, so scoring metrics on it would be noise.
TIE_THRESHOLD = 0.005

# Synthetic proportion of each mixture, known from the recipe. Used as the
# baseline predictor: if a metric cannot beat simply counting synthetic images,
# it has not been shown to contribute anything beyond dataset composition.
SYNTHETIC_FRACTION = {
    "cityscapes100": 0,
    "cityscapes75_gta25": 25, "cityscapes75_synscapes25": 25,
    "cityscapes50_gta50": 50, "cityscapes50_synscapes50": 50,
    "cityscapes25_gta75": 75, "cityscapes25_synscapes75": 75,
    "gta100": 100, "synscapes100": 100,
}

FRIENDLY_NAMES = {
    "frechet_dinov2": "FID (DINOv2)",
    "frechet_streetclip": "FCD (StreetCLIP)",
    "centroid_dinov2": "centroid (DINOv2)",
    "centroid_streetclip": "centroid (StreetCLIP)",
    "color_js": "colour histogram JS",
    "texture_js": "LBP texture JS",
    "class_frequency_js": "class frequency JS",
    "class_presence_js": "class presence JS",
    "scene_complexity_js": "scene complexity JS",
    "sadge": "SADGE fused (similarity)",
    "sadge_appearance": "SADGE app. DINOv3 (sim)",
    "sadge_geometry": "SADGE geom. MASt3R (sim)",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distances", default="results/shift/distances.json")
    parser.add_argument("--output-root", default="output")
    parser.add_argument("--model", default="segformer")
    parser.add_argument(
        "--performance-source",
        choices=("test", "validation"),
        default="test",
        help=(
            "'test' reads test_eval/summary.json (recommended: matches the "
            "split the shift metrics use). 'validation' falls back to the "
            "training history's best val mIoU, which is optimistically biased "
            "because it drove early stopping and checkpoint selection."
        ),
    )
    parser.add_argument(
        "--target",
        choices=("miou", "ece"),
        default="miou",
        help=(
            "Which downstream quantity the shift metrics are asked to predict. "
            "'miou' is segmentation accuracy. 'ece' is expected calibration "
            "error, already recorded for every run and never yet analysed: it "
            "asks whether shift predicts how badly a model misjudges its own "
            "confidence, not just how often it is wrong. ECE is an error, so "
            "the expected signs are reversed relative to mIoU: a distance "
            "should correlate POSITIVELY with it."
        ),
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help=(
            "Significance level for the matched-ratio tests and for the "
            "Holm-adjusted correlation p-values."
        ),
    )
    parser.add_argument(
        "--seeds",
        nargs="*",
        default=None,
        help=(
            "Seeds to average over, e.g. --seeds 42 43 44. Seed 42 lives in "
            "output/<mix>/<model>/ and others in output/<mix>/<model>_seed<N>/, "
            "matching train_array.sbatch. With more than one seed the tie "
            "threshold is MEASURED from the observed spread instead of assumed."
        ),
    )
    parser.add_argument(
        "--out",
        default=None,
        help=(
            "Defaults to results/shift/correlation_<model>.json. The model name "
            "is in the filename deliberately: a fixed default silently "
            "overwrites one architecture's analysis when the other is run."
        ),
    )
    args = parser.parse_args(argv)
    if args.out is None:
        suffix = "" if args.target == "miou" else f"_{args.target}"
        args.out = f"results/shift/correlation_{args.model}{suffix}.json"
    return args


def run_subdir(model: str, seed: str) -> str:
    """Directory name for a (model, seed) run, mirroring train_array.sbatch."""
    return model if str(seed) == "42" else f"{model}_seed{seed}"


def load_performance_all_seeds(
    root: Path, mix: str, model: str, source: str, seeds: list[str] | None,
    target: str = "miou",
) -> list[float]:
    """Every available per-seed score for one mixture, in seed order."""
    values = []
    for seed in (seeds or ["42"]):
        value = load_performance(
            root, mix, model, source, run_subdir(model, seed), target
        )
        if value is not None:
            values.append(value)
    return values


def load_performance(
    root: Path, mix: str, model: str, source: str, subdir: str | None = None,
    target: str = "miou",
) -> float | None:
    subdir = subdir or model
    if source == "test":
        path = root / mix / subdir / "test_eval" / "summary.json"
        if not path.is_file():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        metrics = payload.get("metrics")
        if not metrics:
            return None
        return float(metrics["ece"] if target == "ece" else metrics["mIoU"])

    if target == "ece":
        # Per-epoch history records val mIoU but not ECE, so there is nothing
        # to fall back on here.
        raise SystemExit(
            "--target ece requires --performance-source test; the training "
            "history does not record calibration error."
        )

    path = root / mix / subdir / f"training_history_{model}.json"
    if not path.is_file():
        return None
    history = json.loads(path.read_text(encoding="utf-8"))
    return max(float(e["val_mIoU"]) for e in history) if history else None


def measure_tie_threshold(per_seed: dict[str, list[float]]) -> tuple[float, dict]:
    """Derive a tie threshold from the observed seed-to-seed spread.

    With one seed per mixture there is no way to tell a real 0.4-point
    difference between synthetic sources from run-to-run noise, so TIE_THRESHOLD
    had to be assumed. Repeat seeds make it measurable.

    The threshold is 2 * the pooled per-mixture standard deviation, a rough
    stand-in for the standard error of a difference between two independent
    means. Differences smaller than that are indistinguishable from optimisation
    noise -- initialisation, batch ordering, GPU nondeterminism -- and are
    scored as ties rather than as calls a metric can get right or wrong.

    Returns (threshold, diagnostics). Falls back to the assumed constant when
    fewer than two seeds are available for any mixture.
    """
    spreads = {m: v for m, v in per_seed.items() if len(v) >= 2}
    if not spreads:
        return TIE_THRESHOLD, {"source": "assumed", "value": TIE_THRESHOLD}

    # Population std per mixture, pooled by quadrature.
    stds = {}
    for mixture, values in spreads.items():
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        stds[mixture] = variance ** 0.5
    pooled = (sum(s ** 2 for s in stds.values()) / len(stds)) ** 0.5
    threshold = 2.0 * pooled
    return threshold, {
        "source": "measured",
        "value": threshold,
        "pooled_std": pooled,
        "seeds_per_mixture": {m: len(v) for m, v in per_seed.items()},
        "per_mixture_std": stds,
        "assumed_fallback": TIE_THRESHOLD,
    }


def compare_pair(
    gta_values: list[float],
    syn_values: list[float],
    alpha: float = 0.05,
    fallback_threshold: float = TIE_THRESHOLD,
    lower_is_better: bool = False,
) -> tuple[str, dict]:
    """Decide which synthetic source performed better at one matched ratio.

    With repeat seeds the primary test is a Welch two-sample t-test on the
    per-seed values, which does not assume the two mixtures have matching
    variances.

    Seed k for GTA-V and seed k for Synscapes are not fully independent:
    seed_everything() fixes decoder initialisation and the DataLoader
    generator, so both runs start from the same RNG state and differ only in
    the data they see. That partial pairing makes a paired test defensible, so
    it is computed and reported alongside rather than dismissed. Welch remains
    primary because the pairing is only partial, and the paired result is there
    to show whether the verdict depends on that choice.

    A difference that is not significant at alpha is reported as a tie: it
    cannot be distinguished from run-to-run noise, so no metric can be scored
    right or wrong on it. This replaces the earlier fixed threshold, which
    required assuming a noise level rather than measuring one.

    With fewer than two seeds per mixture there is nothing to test, so the
    fixed threshold is used and flagged as such.

    Note the sample is tiny (typically n = 3 per group), so power is low --
    a real difference of a few tenths of a point will not reach significance.
    Read a tie as "not resolvable at this sample size", not "no difference".
    """
    mean_gta = sum(gta_values) / len(gta_values)
    mean_syn = sum(syn_values) / len(syn_values)
    delta = mean_syn - mean_gta

    # With mIoU a positive delta means Synscapes did better. With ECE, which is
    # an error, the sense reverses.
    def _winner_from(sign_positive: bool) -> str:
        favours_syn = (not sign_positive) if lower_is_better else sign_positive
        return "synscapes" if favours_syn else "gta"

    if len(gta_values) < 2 or len(syn_values) < 2:
        winner = ("tie" if abs(delta) < fallback_threshold
                  else _winner_from(delta > 0))
        return winner, {
            "test": "fixed_threshold",
            "threshold": fallback_threshold,
            "delta": delta,
            "n_gta": len(gta_values), "n_syn": len(syn_values),
        }

    result = stats.ttest_ind(syn_values, gta_values, equal_var=False)
    var_gta = stats.tvar(gta_values)
    var_syn = stats.tvar(syn_values)
    standard_error = (var_gta / len(gta_values) + var_syn / len(syn_values)) ** 0.5
    degrees = float(getattr(result, "df", float("nan")))
    if standard_error > 0 and degrees == degrees:  # df not NaN
        critical = float(stats.t.ppf(1 - alpha / 2, degrees))
        interval = (delta - critical * standard_error, delta + critical * standard_error)
    else:
        interval = (float("nan"), float("nan"))

    significant = float(result.pvalue) < alpha
    winner = _winner_from(delta > 0) if significant else "tie"
    payload = {
        "test": "welch_t",
        "delta": delta,
        "ci95": list(interval),
        "t": float(result.statistic),
        "p": float(result.pvalue),
        "df": degrees,
        "alpha": alpha,
        "significant": bool(significant),
        "n_gta": len(gta_values), "n_syn": len(syn_values),
    }

    # Sensitivity analysis. Runs sharing a seed index also share controlled
    # randomness: seed_everything() fixes decoder initialisation and the
    # DataLoader generator, so seed k for one mixture and seed k for the other
    # start from the same RNG state. That makes a paired test defensible even
    # though the two mixtures are different datasets. Welch stays primary
    # because the pairing is only partial (the data itself differs); this is
    # reported alongside so a reader can see whether the verdict depends on it.
    if len(gta_values) == len(syn_values) and len(gta_values) > 1:
        paired = stats.ttest_rel(syn_values, gta_values)
        paired_significant = float(paired.pvalue) < alpha
        payload["paired"] = {
            "t": float(paired.statistic),
            "p": float(paired.pvalue),
            "significant": bool(paired_significant),
            "winner": _winner_from(delta > 0) if paired_significant else "tie",
            "agrees_with_welch": bool(paired_significant == significant),
        }

    return winner, payload


def within_ratio_analysis(
    distances: dict,
    performance: dict,
    metric_names: list[str],
    tie_threshold: float = TIE_THRESHOLD,
    per_seed: dict[str, list[float]] | None = None,
    alpha: float = 0.05,
    lower_is_better: bool = False,
) -> dict:
    """Score each metric on the comparison the study is actually about.

    The nine mixtures are not nine independent samples: they are two arms of
    five points sharing a common origin, and every quantity moves monotonically
    with synthetic fraction. A metric that merely rises with synthetic content
    therefore earns a high global Spearman whether or not it captures anything
    about domain gap.

    Holding the synthetic fraction fixed removes that shared trend. At a given
    ratio the only thing that differs is WHICH synthetic source was used, so the
    question becomes: does this metric identify the source that produces the
    better model? That is the practical question a practitioner would ask, and
    it discriminates between metrics that the global correlation does not.
    """
    results: dict[str, dict] = {}
    pairs_used = []

    per_seed = per_seed or {}
    for label, gta_mix, syn_mix in MATCHED_PAIRS:
        if gta_mix not in performance or syn_mix not in performance:
            continue
        gta_values = per_seed.get(gta_mix, [performance[gta_mix]])
        syn_values = per_seed.get(syn_mix, [performance[syn_mix]])
        better, test = compare_pair(
            gta_values, syn_values, alpha=alpha, fallback_threshold=tie_threshold,
            lower_is_better=lower_is_better,
        )
        pairs_used.append({
            "ratio": label,
            "gta_mixture": gta_mix,
            "synscapes_mixture": syn_mix,
            "gta_miou": performance[gta_mix],
            "synscapes_miou": performance[syn_mix],
            "miou_delta": test["delta"],
            "better": better,
            "test": test,
        })

    for metric in metric_names:
        calls, correct, scored = [], 0, 0
        for pair in pairs_used:
            gta_value = distances[pair["gta_mixture"]].get(metric)
            syn_value = distances[pair["synscapes_mixture"]].get(metric)
            if gta_value is None or syn_value is None:
                # The metric has no value for one side of this pair, which is
                # different from the pair being a statistical tie. Record it
                # explicitly so the printed table keeps one cell per ratio --
                # silently skipping shifts every later column left and makes
                # the row impossible to read against the others.
                calls.append({
                    "ratio": pair["ratio"], "predicted": None,
                    "actual": pair["better"], "correct": None,
                    "unavailable": True,
                })
                continue
            # A distance says a source is closer when its value is SMALLER;
            # a similarity says so when its value is LARGER.
            if metric in SIMILARITY_METRICS:
                predicted = "synscapes" if syn_value > gta_value else "gta"
            else:
                predicted = "synscapes" if syn_value < gta_value else "gta"
            is_tie = pair["better"] == "tie"
            hit = None if is_tie else (predicted == pair["better"])
            if hit is not None:
                scored += 1
                correct += int(hit)
            calls.append({
                "ratio": pair["ratio"], "predicted": predicted,
                "actual": pair["better"], "correct": hit,
            })
        results[metric] = {
            "label": FRIENDLY_NAMES.get(metric, metric),
            "correct": correct,
            "scored": scored,
            "accuracy": (correct / scored) if scored else None,
            "calls": calls,
        }

    return {"pairs": pairs_used, "metrics": results,
            "tie_threshold": tie_threshold}


def print_within_ratio(within: dict, target: str = "miou") -> None:
    pairs = within["pairs"]
    print("\n\n=== within-ratio test: which synthetic source is closer? ===")
    print("Holds synthetic fraction fixed, so the shared monotonic trend that")
    print("inflates the global correlation cannot contribute.\n")
    label = "test ECE, lower is better" if target == "ece" else "test mIoU"
    print(f"ground truth ({label}):")
    for p in pairs:
        t = p.get("test", {})
        if t.get("test") == "welch_t":
            lo, hi = t["ci95"]
            verdict = p["better"] if t["significant"] else "tie (n.s.)"
            print(f"  {p['ratio']:>5s} synthetic: GTA-V {p['gta_miou']:.4f} vs "
                  f"Synscapes {p['synscapes_miou']:.4f}")
            print(f"         delta {t['delta']:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  "
                  f"t={t['t']:.2f}  p={t['p']:.4f}  ->  {verdict}")
        else:
            note = "  (tie)" if p["better"] == "tie" else ""
            print(f"  {p['ratio']:>5s} synthetic: GTA-V {p['gta_miou']:.4f} vs "
                  f"Synscapes {p['synscapes_miou']:.4f}  ->  "
                  f"{p['better']}{note}   [fixed threshold, no repeat seeds]")

    ratios = [p["ratio"] for p in pairs]
    header = "".join(f"{r:>7s}" for r in ratios)
    print(f"\n{'metric':26s}{header}   correct")
    print("-" * (26 + 7 * len(ratios) + 10))
    ordered = sorted(
        within["metrics"].items(),
        key=lambda kv: (kv[1]["accuracy"] is None, -(kv[1]["accuracy"] or 0)),
    )
    any_unavailable = False
    for _, entry in ordered:
        cells = ""
        for call in entry["calls"]:
            if call.get("unavailable"):
                mark = "?"
                any_unavailable = True
            elif call["correct"] is None:
                mark = "-"
            else:
                mark = "y" if call["correct"] else "N"
            cells += f"{mark:>7s}"
        score = (f"{entry['correct']}/{entry['scored']}"
                 if entry["scored"] else "n/a")
        print(f"{entry['label']:26s}{cells}   {score}")
    print("\ny = ranked the sources correctly, N = ranked them backwards,")
    print(f"- = not scored, the {'ECE' if target == 'ece' else 'mIoU'} "
          "difference was not significant.")
    if any_unavailable:
        print("? = the metric has no value for one mixture in that pair, so it")
        print("    could not be scored (distinct from a statistical tie).")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    distances_path = Path(args.distances)
    if not distances_path.is_file():
        raise SystemExit(
            f"Missing {distances_path}. Run scripts/run_shift_metrics.py first."
        )
    distances = json.loads(distances_path.read_text(encoding="utf-8"))
    root = Path(args.output_root)

    # Per-seed scores drive both the mean used for correlation and the measured
    # spread used for the tie threshold.
    per_seed: dict[str, list[float]] = {}
    performance: dict[str, float] = {}
    missing: list[str] = []
    for mix in distances:
        values = load_performance_all_seeds(
            root, mix, args.model, args.performance_source, args.seeds,
            args.target,
        )
        if not values:
            missing.append(mix)
        else:
            per_seed[mix] = values
            performance[mix] = sum(values) / len(values)

    if missing:
        print(f"WARNING: no {args.performance_source} performance for: "
              f"{', '.join(sorted(missing))}")
        if args.performance_source == "test":
            print("         Run scripts/hpc/evaluate_test.sbatch to produce it.\n")
    if len(performance) < 3:
        raise SystemExit(
            f"Only {len(performance)} mixtures have both a shift score and a "
            "performance value; correlation needs at least 3."
        )

    mixes = sorted(performance)
    miou = [performance[m] for m in mixes]

    lower_is_better = args.target == "ece"

    metric_names = sorted({k for m in mixes for k in distances[m]})
    rows = []
    for metric in metric_names:
        if not all(metric in distances[m] for m in mixes):
            print(f"WARNING: {metric} missing for some mixtures; skipped")
            continue
        values = [distances[m][metric] for m in mixes]
        spearman = stats.spearmanr(values, miou)
        pearson = stats.pearsonr(values, miou)
        # A distance should anti-correlate with mIoU; a similarity should
        # correlate. Flip distances so bigger 'predictive' = better predictor.
        # A distance should anti-correlate with mIoU and a similarity should
        # correlate with it. ECE is an error rather than a score, so both
        # expectations invert.
        expected_sign = 1.0 if metric in SIMILARITY_METRICS else -1.0
        if lower_is_better:
            expected_sign = -expected_sign
        rows.append({
            "metric": metric,
            "label": FRIENDLY_NAMES.get(metric, metric),
            "kind": "similarity" if metric in SIMILARITY_METRICS else "distance",
            "raw_spearman": float(spearman.statistic),
            "spearman_p": float(spearman.pvalue),
            "raw_pearson": float(pearson.statistic),
            "pearson_p": float(pearson.pvalue),
            "predictive": float(spearman.statistic) * expected_sign,
        })

    # Holm-Bonferroni within this family. One family is the twelve metrics for
    # one architecture and one downstream target, so a separate correction runs
    # for each combination rather than once across all of them. Sort ascending,
    # multiply the i-th smallest by (m - i), and enforce monotonicity with a
    # running maximum so an adjusted value never falls below a smaller one.
    ordered = sorted(rows, key=lambda r: r["spearman_p"])
    running = 0.0
    for index, row in enumerate(ordered):
        adjusted = min(1.0, row["spearman_p"] * (len(ordered) - index))
        running = max(running, adjusted)
        row["spearman_p_holm"] = running
        row["survives_holm"] = bool(running < args.alpha)

    rows.sort(key=lambda r: r["predictive"], reverse=True)

    # Baseline. Every mixture's shift score and its downstream score both move
    # with the synthetic proportion, which is known from the recipe without
    # measuring anything. A metric that does not beat this baseline has not been
    # shown to add information. Reported here so the comparison is part of the
    # pipeline rather than an afterthought.
    unknown = [m for m in mixes if m not in SYNTHETIC_FRACTION]
    if unknown:
        print(f"WARNING: no synthetic fraction recorded for {unknown}; "
              f"baseline comparison skipped")
        baseline = {"predictor": "synthetic_fraction", "unavailable": unknown}
    else:
        fractions = [SYNTHETIC_FRACTION[m] for m in mixes]
        baseline_spearman = stats.spearmanr(fractions, miou)
        baseline_expected_sign = 1.0 if lower_is_better else -1.0
        baseline = {
            "predictor": "synthetic_fraction",
            "raw_spearman": float(baseline_spearman.statistic),
            "spearman_p": float(baseline_spearman.pvalue),
            "predictive": float(baseline_spearman.statistic) * baseline_expected_sign,
            "best_metric": rows[0]["metric"] if rows else None,
            "best_metric_predictive": rows[0]["predictive"] if rows else None,
        }
    if rows and "raw_spearman" in baseline:
        baseline["margin_over_baseline"] = (
            rows[0]["predictive"] - baseline["predictive"]
        )
        # At n = 9 one adjacent rank swap moves Spearman by 6/(n(n^2-1)) * 2.
        # A margin at or below that is a single swap in the ordering, which is
        # the smallest difference the statistic can express and not evidence of
        # a better predictor. Reported in units of swaps rather than as a
        # boolean, because the comparison sits exactly on the boundary here and
        # a strict inequality would flip on floating-point noise.
        swap = 12.0 / (len(mixes) * (len(mixes) ** 2 - 1))
        in_swaps = baseline["margin_over_baseline"] / swap
        baseline["one_swap_is"] = swap
        baseline["margin_in_swaps"] = in_swaps
        # Spearman is quantised at this n: without ties, achievable values are
        # multiples of 1/60 apart, so margins come in whole swaps. The baseline
        # has tied ranks, which shifts it slightly off that grid. The boundary
        # therefore sits between one swap and two rather than just above one,
        # so the classification does not hinge on that offset.
        if in_swaps <= 1.5:
            baseline["verdict"] = "within one rank swap of the baseline"
        elif in_swaps <= 2.5:
            baseline["verdict"] = "about two rank swaps above the baseline"
        else:
            baseline["verdict"] = "more than two rank swaps above the baseline"

    label = "ECE (lower is better)" if lower_is_better else "mIoU"
    print(f"n = {len(mixes)} mixtures | target = {args.performance_source} "
          f"{label} | model = {args.model}\n")
    print(f"{'metric':26s} {'kind':11s} {'spearman':>9s} {'p':>7s} "
          f"{'pearson':>8s} {'predictive':>11s}")
    print("-" * 78)
    for r in rows:
        print(f"{r['label']:26s} {r['kind']:11s} {r['raw_spearman']:9.3f} "
              f"{r['spearman_p']:7.3f} {r['raw_pearson']:8.3f} "
              f"{r['predictive']:11.3f}")

    print("\n'predictive' = Spearman with the expected sign applied, so +1.0 is a")
    print("perfect predictor of utility and -1.0 predicts exactly backwards.")
    print(f"\nWith n = {len(mixes)}, these correlations carry wide uncertainty;")
    print("treat p-values as weak evidence, not confirmation.")

    threshold, tie_info = measure_tie_threshold(per_seed)
    counts = sorted({len(v) for v in per_seed.values()})
    if tie_info["source"] == "measured":
        unit = "ECE" if lower_is_better else "mIoU"
        print(f"\nSeeds per mixture: {counts}. Observed per-mixture std "
              f"{tie_info['pooled_std']:.4f} {unit} (pooled).")
        print("Matched-ratio pairs are decided by a Welch t-test on the per-seed")
        print("values rather than a fixed threshold, so 'tie' means the")
        print("difference is not resolvable at this sample size.")
    else:
        print(f"\nOne seed per mixture: no variance to test against, so pairs "
              f"fall back to the assumed {TIE_THRESHOLD} threshold.")
        print("Run repeats with SEED=43/44 for a statistical test instead.")

    within = within_ratio_analysis(
        distances, performance, metric_names,
        tie_threshold=threshold, per_seed=per_seed,
        alpha=args.alpha, lower_is_better=lower_is_better,
    )
    print_within_ratio(within, target=args.target)

    print("\n\n=== baseline: does any metric beat the mixing ratio? ===")
    print("Synthetic proportion is known from the recipe without measuring")
    print("anything, and both the shift scores and performance move with it.\n")
    if "predictive" in baseline:
        print(f"  synthetic fraction     predictive = {baseline['predictive']:+.3f}")
    if baseline.get("best_metric"):
        print(f"  best metric ({baseline['best_metric']:<20s}) "
              f"= {baseline['best_metric_predictive']:+.3f}")
        print(f"  margin = {baseline['margin_over_baseline']:+.4f}, "
              f"one adjacent rank swap = {baseline['one_swap_is']:.4f}, "
              f"so {baseline['margin_in_swaps']:.2f} swaps")
        print(f"  -> the best metric is {baseline['verdict']}")

    survivors = sum(1 for r in rows if r.get("survives_holm"))
    print(f"\nHolm correction within this family of {len(rows)} metrics: "
          f"{survivors}/{len(rows)} survive at alpha = {args.alpha}.")

    payload = {
        "n": len(mixes),
        "model": args.model,
        "performance_source": args.performance_source,
        "target": args.target,
        "alpha": args.alpha,
        "mixtures": mixes,
        "performance": performance,
        "performance_per_seed": per_seed,
        "seeds_requested": args.seeds,
        "tie_threshold": tie_info,
        "correlations": rows,
        "fraction_baseline": baseline,
        "within_ratio": within,
    }
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
