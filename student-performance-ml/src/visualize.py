import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
SCREENSHOTS_DIR = PROJECT_ROOT.parent / "screenshots"


def set_plot_style():
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "figure.titlesize": 15,
        "figure.autolayout": True,
    })


def plot_actual_vs_predicted(y_true, y_pred, r2, mae, model_name="Linear Regression", output_filename="actual_vs_predicted.png"):
    set_plot_style()
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)

    # Scatter of actual vs predicted
    scatter = ax.scatter(
        y_true,
        y_pred,
        color="#4338ca",
        alpha=0.65,
        edgecolors="white",
        s=60,
        label="Test Students",
    )

    # Reference diagonal line (Perfect prediction: y = x)
    min_val = min(min(y_true), min(y_pred)) - 1
    max_val = max(max(y_true), max(y_pred)) + 1
    ax.plot([min_val, max_val], [min_val, max_val], color="#ef4444", linestyle="--", linewidth=2, label="Ideal (Actual = Predicted)")

    # Best-fit trendline
    z = np.polyfit(y_true, y_pred, 1)
    p = np.poly1d(z)
    ax.plot(np.unique(y_true), p(np.unique(y_true)), color="#059669", linewidth=2, label=f"Model Trend (Slope = {z[0]:.2f})")

    ax.set_title(f"{model_name}: Actual vs Predicted Final Grade (G3)", fontweight="bold", pad=12)
    ax.set_xlabel("Actual Final Grade (G3)")
    ax.set_ylabel("Predicted Final Grade (G3)")
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)

    # Add metrics text box
    metrics_text = f"$R^2$ Score: {r2:.4f}\nMAE: {mae:.4f}"
    ax.text(
        0.05,
        0.92,
        metrics_text,
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#f3f4f6", edgecolor="#cbd5e1", alpha=0.9),
    )

    ax.legend(loc="lower right", frameon=True)

    # Save to both results and screenshots
    for target_dir in [RESULTS_DIR, SCREENSHOTS_DIR]:
        target_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(target_dir / output_filename, bbox_inches="tight")

    plt.close(fig)
    return RESULTS_DIR / output_filename


def plot_feature_impacts(df, output_filename="feature_impacts.png"):
    set_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=150)
    fig.suptitle("Impact of Key Student Factors on Final Grade (G3)", fontsize=16, fontweight="bold", y=1.02)

    # 1. Study Time impact
    ax1 = axes[0, 0]
    study_means = df.groupby("studytime")["G3"].mean()
    study_counts = df.groupby("studytime")["G3"].count()
    bars = ax1.bar(
        [f"{x} (<{(x*2.5):.0f}h)" for x in study_means.index],
        study_means.values,
        color="#3b82f6",
        edgecolor="#1d4ed8",
        alpha=0.85,
    )
    for bar in bars:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, h + 0.2, f"{h:.1f}", ha="center", va="bottom", fontsize=10)
    ax1.set_title("Weekly Study Time vs Average G3", fontweight="bold")
    ax1.set_xlabel("Study Time Category")
    ax1.set_ylabel("Mean Final Grade (0-20)")
    ax1.set_ylim(0, 20)

    # 2. Previous Grades correlation (G1 and G2 vs G3)
    ax2 = axes[0, 1]
    ax2.scatter(df["G2"], df["G3"], color="#10b981", alpha=0.6, edgecolors="white", s=50, label="Period 2 Grade (G2)")
    ax2.scatter(df["G1"], df["G3"], color="#f59e0b", alpha=0.4, edgecolors="white", s=40, label="Period 1 Grade (G1)")
    ax2.plot([0, 20], [0, 20], color="#6b7280", linestyle="--", alpha=0.7)
    ax2.set_title("Interim Grades (G1, G2) vs Final Grade (G3)", fontweight="bold")
    ax2.set_xlabel("Interim Grade")
    ax2.set_ylabel("Final Grade (G3)")
    ax2.legend(loc="upper left")

    # 3. Past Class Failures
    ax3 = axes[1, 0]
    failure_means = df.groupby("failures")["G3"].mean()
    bars = ax3.bar(
        [f"{x} failures" for x in failure_means.index],
        failure_means.values,
        color="#ef4444",
        edgecolor="#b91c1c",
        alpha=0.85,
    )
    for bar in bars:
        h = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width() / 2, h + 0.2, f"{h:.1f}", ha="center", va="bottom", fontsize=10)
    ax3.set_title("Past Failures vs Average G3", fontweight="bold")
    ax3.set_xlabel("Number of Past Failures")
    ax3.set_ylabel("Mean Final Grade (0-20)")
    ax3.set_ylim(0, 20)

    # 4. Absences vs G3 with trend
    ax4 = axes[1, 1]
    ax4.scatter(df["absences"], df["G3"], color="#8b5cf6", alpha=0.6, edgecolors="white", s=45)
    z = np.polyfit(df["absences"], df["G3"], 1)
    p = np.poly1d(z)
    ax4.plot(np.unique(df["absences"]), p(np.unique(df["absences"])), color="#dc2626", linewidth=2, label=f"Trend (Slope = {z[0]:.3f})")
    ax4.set_title("School Absences vs Final Grade (G3)", fontweight="bold")
    ax4.set_xlabel("Number of Absences")
    ax4.set_ylabel("Final Grade (G3)")
    ax4.legend(loc="upper right")

    # Save to results and screenshots
    for target_dir in [RESULTS_DIR, SCREENSHOTS_DIR]:
        target_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(target_dir / output_filename, bbox_inches="tight")

    plt.close(fig)
    return RESULTS_DIR / output_filename


def plot_linear_coefficients(feature_names, coefficients, intercept, output_filename="linear_coefficients.png", top_n=10):
    set_plot_style()
    coef_df = pd.DataFrame({"feature": feature_names, "coefficient": coefficients})
    coef_df["abs_coef"] = coef_df["coefficient"].abs()
    top_features = coef_df.sort_values(by="abs_coef", ascending=False).head(top_n)
    top_features = top_features.sort_values(by="coefficient", ascending=True)

    colors = ["#10b981" if c >= 0 else "#ef4444" for c in top_features["coefficient"]]

    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    bars = ax.barh(top_features["feature"], top_features["coefficient"], color=colors, edgecolor="black", linewidth=0.5)

    for bar in bars:
        width = bar.get_width()
        x_pos = width + (0.05 if width >= 0 else -0.05)
        ha = "left" if width >= 0 else "right"
        ax.text(x_pos, bar.get_y() + bar.get_height() / 2, f"{width:+.3f}", va="center", ha=ha, fontsize=10, fontweight="bold")

    ax.axvline(0, color="#6b7280", linewidth=1)
    ax.set_title(f"Linear Regression: Top {top_n} Feature Coefficients (Intercept = {intercept:.3f})", fontweight="bold", pad=12)
    ax.set_xlabel("Coefficient Impact on Final Grade (G3)")
    ax.set_ylabel("Model Feature")

    # Add legend
    import matplotlib.patches as mpatches
    pos_patch = mpatches.Patch(color="#10b981", label="Positive Impact (Increases G3)")
    neg_patch = mpatches.Patch(color="#ef4444", label="Negative Impact (Decreases G3)")
    ax.legend(handles=[pos_patch, neg_patch], loc="lower right")

    for target_dir in [RESULTS_DIR, SCREENSHOTS_DIR]:
        target_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(target_dir / output_filename, bbox_inches="tight")

    plt.close(fig)
    return RESULTS_DIR / output_filename
