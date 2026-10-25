"""
main.py
=======
Inference and evaluation pipeline only (no training, no CV):
  1. Check required artifacts exist
  2. Predict validation.csv -> validation_predictions.csv and fill data/december_chart_inputs.csv
  3. Run score.py
  4. Generate reports/report.pdf (reads oof_eval.npz and cv_results.csv)
  5. Print final checklist
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from src.config import BASE_DIR, VALIDATION_PREDICTIONS_PATH, DECEMBER_INPUTS_PATH, ARTIFACTS_DIR
from src.predict import make_predictions
from src.report import generate_report


def check_artifacts() -> bool:
    meta_path = ARTIFACTS_DIR / "cv_meta.json"
    if not meta_path.exists():
        return False
    try:
        with open(meta_path) as f:
            meta = json.load(f)
    except Exception:
        return False

    best_model = meta.get("best_model", "catboost")
    required = [
        ARTIFACTS_DIR / f"{best_model}_model.pkl",
        ARTIFACTS_DIR / "city_encoder.pkl",
        ARTIFACTS_DIR / "pair_encoder.pkl",
        ARTIFACTS_DIR / "coord_lookup.pkl",
        ARTIFACTS_DIR / "imp_stats.pkl",
        ARTIFACTS_DIR / "cv_meta.json",
        ARTIFACTS_DIR / "oof_eval.npz",
    ]
    return all(p.exists() for p in required)


def main():
    if not check_artifacts():
        print("Run `python train.py` first.")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("STEP: Generating Predictions")
    print("=" * 60)
    make_predictions()

    print("\n" + "=" * 60)
    print("STEP: Running score.py")
    print("=" * 60)
    result = subprocess.run(
        [
            sys.executable,
            "score.py",
            "--predictions", str(VALIDATION_PREDICTIONS_PATH),
            "--december-predictions", str(DECEMBER_INPUTS_PATH),
        ],
        cwd=str(BASE_DIR),
        capture_output=True,
        text=True,
    )
    print(result.stdout)
    if result.returncode != 0:
        print(f"ERROR: score.py failed!\n{result.stderr}")
        sys.exit(1)
    print("scorer passed")

    print("\n" + "=" * 60)
    print("STEP: Generating PDF Report")
    print("=" * 60)
    generate_report()

    # Final checklist
    print("\n" + "=" * 60)
    print("FINAL CHECKLIST")
    print("=" * 60)

    checks = []
    checks.append(("Scorer passes (score.py exits 0)", result.returncode == 0))

    vp = VALIDATION_PREDICTIONS_PATH
    if vp.exists():
        df = pd.read_csv(vp)
        row_ok = len(df) == 12_000
        id_ok = set(df["load_id"]) == {f"TE-{i:06d}" for i in range(1, 12_001)}
        nan_ok = df["predicted_rate"].isna().sum() == 0
        pos_ok = (df["predicted_rate"] > 0).all()
        checks.append(("validation_predictions.csv -- 12,000 rows", row_ok))
        checks.append(("validation_predictions.csv -- IDs match template", id_ok))
        checks.append(("validation_predictions.csv -- no NaN predictions", nan_ok))
        checks.append(("validation_predictions.csv -- all positive predictions", pos_ok))
    else:
        checks.append(("validation_predictions.csv -- EXISTS", False))

    dec_path = DECEMBER_INPUTS_PATH
    if dec_path.exists():
        dec = pd.read_csv(dec_path)
        dec_ok = (dec["predicted_rate"] > 0).all() and len(dec) == 31
        checks.append(("december_chart_inputs.csv -- 31 rows, all positive", dec_ok))
    else:
        checks.append(("december_chart_inputs.csv -- EXISTS", False))

    chart_path = BASE_DIR / "scorer_results" / "candidate_december.png"
    checks.append(("scorer_results/candidate_december.png -- EXISTS", chart_path.exists()))

    report_path = BASE_DIR / "reports" / "report.pdf"
    checks.append(("reports/report.pdf -- EXISTS", report_path.exists()))

    readme_path = BASE_DIR / "README.md"
    checks.append(("README.md -- EXISTS", readme_path.exists()))

    for name, ok in checks:
        icon = "[OK]" if ok else "[FAIL]"
        print(f"  {icon} {name}")

    if not all(ok for _, ok in checks):
        sys.exit(1)


if __name__ == "__main__":
    main()
