"""Metrics table and figures from saved predictions (no GPU or PyTorch needed).

    python src/report.py

Reads results/evaluation/predictions_{val,test}.csv and writes
    results/evaluation/metrics.csv        MAE / RMSE / %MAE per model and nutrient
    results/evaluation/metrics.md         the same table, ready for the README
    results/evaluation/pred_vs_true.png   predicted vs true, best model
"""
import numpy as np
import pandas as pd

from config import RESULTS_DIR, TARGETS, UNITS, load_labels, make_splits

EVAL_DIR = RESULTS_DIR / "evaluation"


def metrics_for(name, y_true, y_pred):
    err = y_pred - y_true
    return {"model": name,
            **{f"{t}_MAE": np.abs(err[:, i]).mean() for i, t in enumerate(TARGETS)},
            **{f"{t}_RMSE": np.sqrt((err[:, i] ** 2).mean()) for i, t in enumerate(TARGETS)},
            **{f"{t}_pctMAE": 100 * np.abs(err[:, i]).mean() / y_true[:, i].mean() for i, t in enumerate(TARGETS)}}


def main():
    preds = pd.read_csv(EVAL_DIR / "predictions_test.csv")
    train_df, _, test_df = make_splits(load_labels())
    rows = []

    # Reference: always predict the training-set mean. Any useful model must beat this.
    y_test = test_df[TARGETS].to_numpy()
    rows.append(metrics_for("Baseline (train mean)", y_test, np.tile(train_df[TARGETS].mean().to_numpy(), (len(y_test), 1))))

    for name, df in preds.groupby("model", sort=False):
        y_true = df[[f"{t}_actual" for t in TARGETS]].to_numpy()
        y_pred = df[[f"{t}_predicted" for t in TARGETS]].clip(lower=0).to_numpy()
        rows.append(metrics_for(name, y_true, y_pred))

    table = pd.DataFrame(rows)

    # Validation calorie MAE: the ONLY number used to choose the best model.
    val_path = EVAL_DIR / "predictions_val.csv"
    table["val_calories_MAE"] = np.nan
    if val_path.exists():
        val = pd.read_csv(val_path)
        for name, df in val.groupby("model", sort=False):
            mae = (df["calories_predicted"].clip(lower=0) - df["calories_actual"]).abs().mean()
            table.loc[table["model"] == name, "val_calories_MAE"] = mae
    table.to_csv(EVAL_DIR / "metrics.csv", index=False)

    header = ("| Model | Val calories (kcal) | " + " | ".join(f"Test {t} ({UNITS[t]})" for t in TARGETS) + " |")
    lines = [header, "|" + "---|" * (len(TARGETS) + 2)]
    for _, r in table.iterrows():
        val_cell = "–" if np.isnan(r["val_calories_MAE"]) else f"{r['val_calories_MAE']:.1f}"
        cells = [f"{r[f'{t}_MAE']:.1f} ({r[f'{t}_pctMAE']:.0f}%)" for t in TARGETS]
        lines.append(f"| {r['model']} | {val_cell} | " + " | ".join(cells) + " |")
    md = "\n".join(lines)
    (EVAL_DIR / "metrics.md").write_text(md + "\n\nMAE, with MAE as % of the mean true value in brackets. "
                                          "The best model is chosen on the validation column only.\n")
    print(md)

    candidates = table.iloc[1:]
    key = "val_calories_MAE" if candidates["val_calories_MAE"].notna().any() else "calories_MAE"
    best = candidates.sort_values(key).iloc[0]["model"]
    base = table.iloc[0]["calories_MAE"]
    print(f"\nBest model (chosen on validation): {best} | test calorie MAE {table.set_index('model').loc[best, 'calories_MAE']:.1f} kcal "
          f"vs {base:.1f} kcal for the mean baseline")
    plot(preds[preds["model"] == best], best)


def plot(df, name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(TARGETS), figsize=(18, 3.8))
    for ax, t in zip(axes, TARGETS):
        y, y_hat = df[f"{t}_actual"], df[f"{t}_predicted"].clip(lower=0)
        lim = np.percentile(np.r_[y, y_hat], 99.5) * 1.05
        ax.scatter(y, y_hat, s=8, alpha=0.5, color="#1f4e79")
        ax.plot([0, lim], [0, lim], color="#c92a2a", lw=1, ls="--")
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_title(f"{t} | MAE {np.abs(y - y_hat).mean():.1f} {UNITS[t]}")
        ax.set_xlabel("true")
    axes[0].set_ylabel("predicted")
    fig.suptitle(f"{name}: predicted vs true on the test set", y=1.02)
    fig.tight_layout()
    fig.savefig(EVAL_DIR / "pred_vs_true.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
