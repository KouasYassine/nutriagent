"""Split conformal prediction intervals for every nutrient.

Turns a point prediction (e.g. 420 kcal) into a range (e.g. 300 - 560 kcal)
that contains the true value with probability >= 1 - alpha, with no assumption
on the model or the error distribution, only exchangeability of dishes
(Vovk et al. 2005; Lei et al. 2018).

Three variants are compared:
  absolute   s = |y - y_hat|                  one width for every dish
  relative   s = |y - y_hat| / (y_hat + c)     width grows with the predicted portion
  binned     absolute score, calibrated separately in small / medium / large
             bins of the PREDICTED value ("Mondrian" conformal prediction,
             Vovk et al. 2005). Coverage is then guaranteed within each bin,
             not only on average, so neither small nor large plates are
             systematically under-covered.

Usage
    python src/uncertainty.py                 # calibrate on val, evaluate on test
    python src/uncertainty.py --alpha 0.2     # 80 % intervals

If predictions_val.csv does not exist yet, the script falls back to repeated
random 50/50 splits of the test predictions (calibrate on one half, measure
coverage on the other). This is statistically valid, just noisier.
The interval parameters used by the app are written to
results/uncertainty/conformal.json.
"""
import argparse
import json

import numpy as np
import pandas as pd

from config import RESULTS_DIR, TARGETS, UNITS

EVAL_DIR = RESULTS_DIR / "evaluation"
OUT_DIR = RESULTS_DIR / "uncertainty"


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Finite-sample corrected (1 - alpha) quantile of calibration scores."""
    n = len(scores)
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(scores, level, method="higher"))


def offsets(cal: pd.DataFrame) -> dict:
    """c in the relative score: 10 % of the mean prediction, so tiny predictions
    (an almost empty plate) do not get zero-width intervals."""
    return {t: 0.1 * float(cal[f"{t}_predicted"].mean()) for t in TARGETS}


def scores(df, target, mode, c):
    y, y_hat = df[f"{target}_actual"].to_numpy(), df[f"{target}_predicted"].to_numpy()
    resid = np.abs(y - y_hat)
    return resid if mode == "absolute" else resid / (y_hat + c[target])


N_BINS = 3
MODES = ("absolute", "relative", "binned")


def fit_binned(cal, target, alpha):
    """Bin edges = terciles of calibration predictions; one quantile per bin."""
    y_hat = cal[f"{target}_predicted"].to_numpy()
    resid = np.abs(cal[f"{target}_actual"].to_numpy() - y_hat)
    edges = np.quantile(y_hat, np.linspace(0, 1, N_BINS + 1)[1:-1]).tolist()
    bins = np.digitize(y_hat, edges)
    return {"edges": edges, "q": [conformal_quantile(resid[bins == b], alpha) for b in range(N_BINS)]}


def interval(y_hat, q, mode, c):
    y_hat = np.asarray(y_hat, dtype=float)
    if mode == "absolute":
        half = q
    elif mode == "relative":
        half = q * (y_hat + c)
    else:  # binned: q = {"edges": [...], "q": [...]}
        half = np.asarray(q["q"])[np.digitize(y_hat, q["edges"])]
    return np.clip(y_hat - half, 0, None), y_hat + half


def fit(cal, alpha):
    c = offsets(cal)
    params = {mode: {t: conformal_quantile(scores(cal, t, mode, c), alpha) for t in TARGETS}
              for mode in ("absolute", "relative")}
    params["binned"] = {t: fit_binned(cal, t, alpha) for t in TARGETS}
    return params, c


def evaluate(test, params, c):
    """Coverage and mean width per target and mode, plus coverage by calorie size."""
    rows = []
    size_bin = pd.qcut(test["calories_actual"], 3, labels=["small", "medium", "large"])
    for mode, qs in params.items():
        for t in TARGETS:
            y, y_hat = test[f"{t}_actual"].to_numpy(), test[f"{t}_predicted"].to_numpy()
            lo, hi = interval(y_hat, qs[t], mode, c[t])
            covered = (y >= lo) & (y <= hi)
            row = {"mode": mode, "target": t, "coverage": covered.mean(), "mean_width": (hi - lo).mean()}
            for b in ["small", "medium", "large"]:
                m = (size_bin == b).to_numpy()
                row[f"coverage_{b}"] = covered[m].mean()
                row[f"width_{b}"] = (hi - lo)[m].mean()
            rows.append(row)
    return pd.DataFrame(rows)


def load(split, model):
    path = EVAL_DIR / f"predictions_{split}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df = df[df["model"] == model].reset_index(drop=True)
    for t in TARGETS:
        df[f"{t}_predicted"] = df[f"{t}_predicted"].clip(lower=0)
    return df


def plot(test, params, c, alpha, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = test.sort_values("calories_predicted").reset_index(drop=True)
    y_hat, y = df["calories_predicted"].to_numpy(), df["calories_actual"].to_numpy()
    x = np.arange(len(df))
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.2), sharey=True)
    for ax, mode in zip(axes, MODES):
        lo, hi = interval(y_hat, params[mode]["calories"], mode, c["calories"])
        inside = (y >= lo) & (y <= hi)
        ax.fill_between(x, lo, hi, color="#9ec5e8", alpha=0.6, label=f"{int(100 * (1 - alpha))}% interval")
        ax.plot(x, y_hat, color="#1f4e79", lw=1.2, label="prediction")
        ax.scatter(x[inside], y[inside], s=7, color="#2b8a3e", label="true (inside)")
        ax.scatter(x[~inside], y[~inside], s=12, color="#c92a2a", label="true (outside)")
        ax.set_title(f"{mode} | coverage {inside.mean():.1%} | mean width {(hi - lo).mean():.0f} kcal", fontsize=10)
        ax.set_xlabel("test dishes, sorted by predicted calories")
        ax.set_ylim(0, np.percentile(y, 99) * 1.3)
    axes[0].set_ylabel("calories (kcal)")
    axes[0].legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="A2_Mass2")
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--repeats", type=int, default=500, help="random splits in fallback mode")
    args = p.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    test = load("test", args.model)
    if test is None or test.empty:
        raise SystemExit(f"No test predictions for {args.model}. Run src/evaluate.py first.")
    cal = load("val", args.model)

    if cal is not None and not cal.empty:
        protocol = "calibrated on validation split, evaluated on test split"
        params, c = fit(cal, args.alpha)
        table = evaluate(test, params, c)
    else:
        protocol = (f"no validation predictions: {args.repeats} random 50/50 splits of the test set "
                    "(calibrate on one half, evaluate on the other); numbers are averages")
        rng = np.random.default_rng(0)
        tables = []
        for _ in range(args.repeats):
            idx = rng.permutation(len(test))
            half = len(test) // 2
            prm, cc = fit(test.iloc[idx[:half]], args.alpha)
            tables.append(evaluate(test.iloc[idx[half:]].reset_index(drop=True), prm, cc))
        allt = pd.concat(tables)
        table = allt.groupby(["mode", "target"], sort=False).mean().reset_index()
        table["coverage_std"] = allt.groupby(["mode", "target"], sort=False)["coverage"].std().to_numpy()
        # Parameters for the app: use every available test dish for calibration.
        params, c = fit(test, args.alpha)

    print(f"Model {args.model} | target coverage {1 - args.alpha:.0%} | {protocol}\n")
    show = table.copy()
    for col in show.columns:
        if col.startswith("coverage"):
            show[col] = (100 * show[col]).round(1)
        elif col.startswith("width") or col == "mean_width":
            show[col] = show[col].round(1)
    print(show.to_string(index=False))

    table.to_csv(OUT_DIR / "coverage.csv", index=False)
    md = ["| Target | Method | Coverage | Small plates | Medium | Large | Mean width |", "|---|---|---|---|---|---|---|"]
    for _, r in table.iterrows():
        md.append(f"| {r['target']} | {r['mode']} | {100 * r['coverage']:.1f}% | {100 * r['coverage_small']:.1f}% | "
                  f"{100 * r['coverage_medium']:.1f}% | {100 * r['coverage_large']:.1f}% | "
                  f"{r['mean_width']:.1f} {UNITS[r['target']]} |")
    (OUT_DIR / "coverage.md").write_text("\n".join(md) + "\n")
    plot(test, params, c, args.alpha, OUT_DIR / "calorie_intervals.png")
    (OUT_DIR / "conformal.json").write_text(json.dumps({
        "model": args.model, "alpha": args.alpha, "protocol": protocol,
        "recommended_mode": "binned", "offsets": c, "quantiles": params,
        "units": UNITS,
    }, indent=2))
    print(f"\nSaved {OUT_DIR / 'coverage.csv'}, calorie_intervals.png and conformal.json")


if __name__ == "__main__":
    main()
