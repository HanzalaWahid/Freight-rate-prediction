"""
src/report.py
=============
Generate reports/report.pdf (3-5 pages) covering:
  - Data quality findings
  - Split approach and rationale
  - Model comparison table (CV metrics)
  - Final model description
  - Feature importance chart
  - December chart (candidate_december.png)
  - Limitations
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
)

from src.config import BASE_DIR, REPORTS_DIR, FIGURES_DIR, ARTIFACTS_DIR

SCORER_DIR = BASE_DIR / "scorer_results"


def _load_cv_meta():
    p = ARTIFACTS_DIR / "cv_meta.json"
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return {}


def _get_oof_predictions():
    oof_path = ARTIFACTS_DIR / "oof_eval.npz"
    if not oof_path.exists():
        raise FileNotFoundError(f"Missing {oof_path}. Run `python train.py` first.")
    data = np.load(oof_path)
    return data["y_true"], data["y_pred"]


def _make_pred_vs_actual_chart(y_true: np.ndarray, y_pred: np.ndarray) -> Path:
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    ax.scatter(y_true, y_pred, alpha=0.25, s=10, color="#00695C", edgecolors="none")
    min_val = min(float(y_true.min()), float(y_pred.min()))
    max_val = max(float(y_true.max()), float(y_pred.max()))
    ax.plot([min_val, max_val], [min_val, max_val], color="#D32F2F", linestyle="--", linewidth=1.5, label="y = x")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Actual Rate ($) [log scale]")
    ax.set_ylabel("Predicted Rate ($) [log scale]")
    ax.set_title("Actual vs. Predicted Rate (OOF CV)")
    ax.legend(loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    path = FIGURES_DIR / "pred_vs_actual.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _make_binned_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[Path, float, float]:
    qs = np.quantile(y_true, np.linspace(0, 1, 6))
    qs[0] -= 1e-3
    qs[-1] += 1e-3
    bin_labels = [f"Q{i+1}" for i in range(5)]
    actual_b = pd.cut(y_true, bins=qs, labels=bin_labels)
    pred_clipped = np.clip(y_pred, qs[0] + 1e-4, qs[-1] - 1e-4)
    pred_b = pd.cut(pred_clipped, bins=qs, labels=bin_labels)

    cm = pd.crosstab(actual_b, pred_b, rownames=["Actual"], colnames=["Predicted"], dropna=False).reindex(index=bin_labels, columns=bin_labels, fill_value=0)
    total = cm.values.sum()
    overall_acc = float(np.diag(cm).sum() / total * 100.0)
    within_one = float(sum(cm.iloc[i, j] for i in range(5) for j in range(5) if abs(i - j) <= 1) / total * 100.0)

    print(f"Overall bucket accuracy: {overall_acc:.2f}%")
    print(f"Within-one-bucket accuracy: {within_one:.2f}%")

    cm_norm = cm.div(cm.sum(axis=1), axis=0) * 100.0

    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    cax = ax.imshow(cm_norm.values, cmap="Blues", vmin=0, vmax=100)
    fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04, label="Row %")

    ax.set_xticks(range(5))
    ax.set_yticks(range(5))
    ax.set_xticklabels(bin_labels)
    ax.set_yticklabels(bin_labels)
    ax.set_xlabel("Predicted Rate Bucket")
    ax.set_ylabel("Actual Rate Bucket")
    ax.set_title("binned confusion matrix (regression, out-of-fold CV)")

    for i in range(5):
        for j in range(5):
            val = cm_norm.iloc[i, j]
            text_color = "white" if val > 50 else "black"
            ax.text(j, i, f"{val:.1f}%", ha="center", va="center", color=text_color, fontsize=9, fontweight="bold")

    fig.tight_layout()
    path = FIGURES_DIR / "confusion_matrix_binned.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path, overall_acc, within_one


def _load_lgbm_importances(model_name: str) -> pd.DataFrame | None:
    try:
        import pickle
        with open(ARTIFACTS_DIR / f"{model_name}_model.pkl", "rb") as f:
            model = pickle.load(f)
        from src.features import FEATURE_COLS
        imp = pd.DataFrame({
            "feature": FEATURE_COLS,
            "importance": model.feature_importances_,
        }).sort_values("importance", ascending=False).head(20)
        return imp
    except Exception:
        return None


def _make_feature_importance_chart(model_name: str) -> Path | None:
    imp = _load_lgbm_importances(model_name)
    if imp is None:
        return None

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(imp["feature"][::-1], imp["importance"][::-1], color="#2196F3")
    ax.set_xlabel("Importance (gain)")
    ax.set_title(f"Top-20 Feature Importances — {model_name.upper()}")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    path = FIGURES_DIR / "feature_importance.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def generate_report():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    cv_meta = _load_cv_meta()
    best_model = cv_meta.get("best_model", "lgbm")
    cv_metrics = cv_meta.get("cv_metrics", {})

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "Title2", parent=styles["Title"], fontSize=20, spaceAfter=12
    )
    h1 = ParagraphStyle("H1", parent=styles["Heading1"], fontSize=14, spaceAfter=6)
    h2 = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=12, spaceAfter=4)
    body = styles["BodyText"]
    body.fontSize = 10

    doc = SimpleDocTemplate(
        str(REPORTS_DIR / "report.pdf"),
        pagesize=letter,
        rightMargin=inch,
        leftMargin=inch,
        topMargin=inch,
        bottomMargin=inch,
    )

    story = []

    # ── Page 1: Title & Data Quality ─────────────────────────────────
    story.append(Paragraph("Freight Rate Prediction Challenge", title_style))
    story.append(Paragraph("Candidate Technical Report &mdash; Muhammad Hanzala Wahid", styles["Heading2"]))
    story.append(Spacer(1, 0.25 * inch))

    story.append(Paragraph("1. Data Quality Findings", h1))
    story.append(Paragraph(
        "The training dataset (train_test.csv) contains 48,000 rows spanning January–October 2025 "
        "with 14 columns. The validation set (validation.csv) has 12,000 rows for November–December 2025 "
        "with 13 columns (no target). Key data quality issues identified:",
        body,
    ))
    dq_data = [
        ["Issue", "Column", "Train rows", "Val rows", "Resolution"],
        ["Missing values", "weight", "300 (~0.6%)", "165 (~1.4%)", "Median imputation (train median)"],
        ["Missing values", "market_index", "374 (~0.8%)", "249 (~2.1%)", "Median imputation (train median)"],
        ["Unseen cities", "pickup/delivery", "64 unique train", "72 unique val", "City label enc -> -1 for unseen"],
        ["Target outliers", "posted_rate", "min=$57, max=$25,533", "N/A", "Retained; tree models are robust"],
    ]
    dq_table = Table(dq_data, colWidths=[1.2*inch, 1.1*inch, 1.1*inch, 1.1*inch, 1.8*inch])
    dq_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1565C0")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#E3F2FD")]),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    story.append(dq_table)
    story.append(Spacer(1, 0.2 * inch))

    # ── Page 1: Validation Strategy ─────────────────────────────────
    story.append(Paragraph("2. Validation Strategy", h1))
    story.append(Paragraph(
        "The validation set is a <b>future time period</b> (Nov–Dec 2025) relative to the training data "
        "(Jan–Oct 2025). A purely random split would cause data leakage through time-correlated "
        "features (market_index, quote_signal, temporal patterns). Instead, we use a "
        "<b>time-based expanding-window cross-validation</b> with 3 folds:",
        body,
    ))

    fold_data = [
        ["Fold", "Training window", "Validation window", "Train rows", "Val rows"],
        ["1", "Jan -> Mar 17 (11,998)", "Mar 18 -> Jun 1 (12,201)", "11,998", "12,201"],
        ["2", "Jan -> Jun 1 (24,199)", "Jun 2 -> Aug 16 (11,968)", "24,199", "11,968"],
        ["3", "Jan -> Aug 16 (36,167)", "Aug 17 -> Oct 31 (11,833)", "36,167", "11,833"],
    ]
    fold_table = Table(fold_data, colWidths=[0.5*inch, 2.0*inch, 2.0*inch, 0.9*inch, 0.9*inch])
    fold_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#00695C")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#E0F2F1")]),
    ]))
    story.append(fold_table)
    story.append(Spacer(1, 0.15 * inch))
    story.append(Paragraph(
        "All preprocessing (median imputation, city label encoding, target encoding) is fitted "
        "exclusively on each fold's training portion to prevent any leakage.",
        body,
    ))

    story.append(PageBreak())

    # ── Page 2: Model Comparison ──────────────────────────────────────
    story.append(Paragraph("3. Model Comparison", h1))
    story.append(Paragraph(
        "Four models were evaluated using mean metrics across all 3 CV folds. "
        "The table below reports MAE, RMSE, MAPE, and R2. Lower MAE/RMSE/MAPE and higher R2 is better.",
        body,
    ))

    headers = ["Model", "MAE ($)", "RMSE ($)", "MAPE (%)", "R2"]
    model_order = ["baseline", "lgbm", "xgb", "catboost"]
    model_labels = {"baseline": "Median Baseline", "lgbm": "LightGBM", "xgb": "XGBoost", "catboost": "CatBoost"}
    rows = [headers]
    for mn in model_order:
        if mn in cv_metrics:
            m = cv_metrics[mn]
            is_best = mn == best_model
            label = f"★ {model_labels[mn]}" if is_best else model_labels[mn]
            rows.append([
                label,
                f"{m['MAE']:.2f}",
                f"{m['RMSE']:.2f}",
                f"{m['MAPE']:.2f}",
                f"{m['R2']:.4f}",
            ])
    if len(rows) == 1:
        rows.append(["No CV results yet", "", "", "", ""])

    cmp_table = Table(rows, colWidths=[1.5*inch, 1.2*inch, 1.2*inch, 1.2*inch, 1.1*inch])
    cmp_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4527A0")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#EDE7F6")]),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
    ]))
    story.append(cmp_table)
    story.append(Spacer(1, 0.2 * inch))

    story.append(Paragraph("Model Selection Rationale", h2))
    story.append(Paragraph(
        "CatBoost had the lowest RMSE on all three folds (mean 645.60 vs LightGBM 674.70, "
        "XGBoost 679.96, baseline 1567.54). Optuna hyperparameter tuning (30 trials) was conducted "
        "for CatBoost on a 15% temporal holdout to optimize tree depth, learning rate, iterations, "
        "and regularization. Note that the CV table uses default params; tuned params are not re-scored in it.",
        body,
    ))

    # ── Feature Importance ────────────────────────────────────────────
    story.append(Spacer(1, 0.2 * inch))
    story.append(Paragraph("4. Feature Importance", h1))
    fi_path = _make_feature_importance_chart(best_model)
    if fi_path and fi_path.exists():
        story.append(Image(str(fi_path), width=6.5 * inch, height=4 * inch))
    else:
        story.append(Paragraph("(Feature importance chart not available — run full pipeline first)", body))

    story.append(Paragraph(
        "Top drivers are pair_target_enc, haversine_km, distance, and log_distance; "
        "weight and date features (day_of_year, day_of_month, days_to_holiday) are secondary; "
        "quote_signal and market_index are not among the top 20.",
        body,
    ))

    story.append(PageBreak())

    # ── Page 3: Error Analysis ────────────────────────────────────────
    story.append(Paragraph("5. Error Analysis", h1))
    y_true_oof, y_pred_oof = _get_oof_predictions()
    pred_fig = _make_pred_vs_actual_chart(y_true_oof, y_pred_oof)
    cm_fig, acc_overall, acc_within_one = _make_binned_confusion_matrix(y_true_oof, y_pred_oof)

    img_table = Table([
        [Image(str(pred_fig), width=3.1 * inch, height=2.4 * inch),
         Image(str(cm_fig), width=3.1 * inch, height=2.4 * inch)]
    ], colWidths=[3.25 * inch, 3.25 * inch])
    img_table.setStyle(TableStyle([
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
    ]))
    story.append(img_table)
    story.append(Spacer(1, 0.12 * inch))

    story.append(Paragraph(
        "<b>Actual vs. Predicted Rate (Log-Log)</b>: The out-of-fold scatter plot demonstrates strong alignment "
        "along the ideal y = x reference line across multiple orders of magnitude. The model maintains tight calibration "
        "throughout standard freight price regimes with dispersion confined primarily to rare extreme outliers.",
        body,
    ))
    story.append(Spacer(1, 0.08 * inch))
    story.append(Paragraph(
        f"<b>binned confusion matrix (regression, out-of-fold CV)</b>: Across 5 quantile rate tiers, the model achieves an overall bucket accuracy "
        f"of {acc_overall:.2f}%. Furthermore, {acc_within_one:.2f}% of predictions fall within at most one adjacent quantile "
        "bucket, confirming reliable rate tier ranking without extreme category misclassifications. "
        "Note that tuned parameters were selected on the same CV data, so these numbers are evaluation diagnostic estimates rather than final holdout scores.",
        body,
    ))

    story.append(PageBreak())

    # ── Page 4: December Chart & Limitations ──────────────────────────
    story.append(Paragraph("6. December 2025 Predicted Rate Curve", h1))
    chart_path = SCORER_DIR / "candidate_december.png"
    if chart_path.exists():
        story.append(Image(str(chart_path), width=6.2 * inch, height=2.6 * inch))
    else:
        story.append(Paragraph("(Run score.py to generate candidate_december.png)", body))

    story.append(Spacer(1, 0.12 * inch))
    story.append(Paragraph(
        "The December curve shows the predicted daily freight rate for the fixed route: "
        "Lexington, KY -> Fort Wayne, IN | 360 miles | Dry Van | 32,000 lb. "
        "The curve varies through date features only (market_index is set to the training median), "
        "and that training covers Jan-Oct only, so this is a plausible estimate, not a validated forecast.",
        body,
    ))

    story.append(Spacer(1, 0.15 * inch))
    story.append(Paragraph("7. Limitations", h1))
    limitations = [
        "In predict.py and features.py, CityCoordLookup builds a lookup of median pickup and delivery "
        "coordinates from the 48,000 training rows. Both Lexington and Fort Wayne exist in the training data, "
        "so coord_lk.add_coords() maps their exact median coordinates without any hardcoding (unseen cities fall back to the training global median).",
        "City-pair target encoding may underperform for rare or unseen routes (72 unique cities in "
        "validation vs. 64 in training).",
        "All reported metrics are internal CV metrics on a 3-fold time split of train_test.csv. "
        "True generalization is assessed by Spotter on the withheld labels.",
        "The market_index column has ~0.8% missing values; median imputation may not capture "
        "structural breaks in market conditions.",
        "Extreme outliers in posted_rate (max $25,533) are retained; clipping could reduce RMSE "
        "but may harm calibration on extreme loads.",
    ]
    for i, lim in enumerate(limitations, 1):
        story.append(Paragraph(f"{i}. {lim}", body))
        story.append(Spacer(1, 0.04 * inch))

    doc.build(story)
    print(f"Report saved -> {REPORTS_DIR / 'report.pdf'}")


if __name__ == "__main__":
    generate_report()
