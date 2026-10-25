"""
main.py
=======
Single entry point: raw data -> all outputs.

Usage:
    python main.py
    python main.py --skip-cv
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

from src.config import BASE_DIR, VALIDATION_PREDICTIONS_PATH, DECEMBER_INPUTS_PATH, ARTIFACTS_DIR


def _run_step(step_name: str, fn):
    print(f"\n{'='*60}")
    print(f"STEP: {step_name}")
    print("=" * 60)
    fn()


def main():
    parser = argparse.ArgumentParser(description="Freight Rate Prediction Pipeline")
    parser.add_argument(
        "--skip-cv",
        action="store_true",
        help="Skip cross-validation if cv_results.csv and cv_meta.json exist",
    )
    args = parser.parse_args()

    # Step a: Cross-Validation
    skip_cv = args.skip_cv and (ARTIFACTS_DIR / "cv_results.csv").exists() and (ARTIFACTS_DIR / "cv_meta.json").exists()
    if skip_cv:
        print(f"\n{'='*60}")
        print("STEP: Cross-Validation (SKIPPED via --skip-cv)")
        print("=" * 60)
        if not (ARTIFACTS_DIR / "oof_eval.npz").exists():
            from src.report import _get_or_compute_oof_predictions
            _get_or_compute_oof_predictions()
    else:
        from src.train_cv import run_cv
        _run_step("Cross-Validation + Optuna Tuning", run_cv)
        if not (ARTIFACTS_DIR / "oof_eval.npz").exists():
            from src.report import _get_or_compute_oof_predictions
            _get_or_compute_oof_predictions()

    # Step b: Final model training on all data
    from src.train_final import run_final_training
    _run_step("Final Model Training (all train data)", run_final_training)

    # Step c: Generate predictions
    from src.predict import make_predictions
    _run_step("Generating Predictions", make_predictions)

    # Step d: Run score.py
    print(f"\n{'='*60}")
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

    # Step e: Generate PDF report
    from src.report import generate_report
    _run_step("Generating PDF Report", generate_report)

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

    loom_path = BASE_DIR / "LOOM_SCRIPT.md"
    checks.append(("LOOM_SCRIPT.md -- EXISTS", loom_path.exists()))

    readme_path = BASE_DIR / "README.md"
    checks.append(("README.md -- EXISTS", readme_path.exists()))

    for name, ok in checks:
        icon = "[OK]" if ok else "[FAIL]"
        print(f"  {icon} {name}")

    all_ok = all(ok for _, ok in checks)
    if all_ok:
        print("\n[PASS] All checks passed! Ready to submit.")
    else:
        print("\n[WARN] Some checks failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
