# LOOM_SCRIPT.md — 2–3 Minute Walkthrough Script

**Total time: ~2.5 minutes**

---

## [0:00 – 0:15] Opening

> "Hi, I'm Muhammad Hanzala Wahid. This is my submission for the Freight Rate Prediction Challenge. I'll walk you through my end-to-end approach in about two and a half minutes."

---

## [0:15 – 0:45] Problem & Data

> "The task: predict freight rates for 12,000 loads in a validation set covering November–December 2025, using a development dataset from January–October 2025.
>
> The training set has 48,000 loads across 14 columns — pickup/delivery cities with coordinates, equipment type, weight, distance, and two market signals: market_index and quote_signal.
>
> Key data quality issues: 300 rows had missing weight (~0.6%) and 374 had missing market_index (~0.8%). I imputed these with training medians fitted strictly per CV fold to avoid lookahead leakage."

---

## [0:45 – 1:15] Validation Strategy

> "Because the validation set is a future time window, a random K-fold split would leak time-correlated market signals. Instead, I used a 3-fold expanding-window cross-validation:
> - Fold 1: Train Jan to Mar 17 (11,998 rows) -> Validate Mar 18 to Jun 1 (12,201 rows)
> - Fold 2: Train Jan to Jun 1 (24,199 rows) -> Validate Jun 2 to Aug 16 (11,968 rows)
> - Fold 3: Train Jan to Aug 16 (36,167 rows) -> Validate Aug 17 to Oct 31 (11,833 rows)
>
> All imputers, label encoders, and smoothed route target encodings are fitted solely on each fold's training portion."

---

## [1:15 – 1:50] Features & Models

> "I engineered 27 features across 5 groups:
> - Route length: Haversine distance and bearing computed from city coordinates, plus log distance
> - Route pricing: Smoothed city-pair target encoding capturing route-level price dynamics
> - Equipment: One-hot flags and ordinal encoding for Dry Van, Flatbed, and Reefer
> - Temporal: Day of week, month, cyclic sin/cos transforms, and distance to US holidays
>
> On 3-fold CV, CatBoost achieved the lowest RMSE across all three folds — mean RMSE 645.60, beating LightGBM at 674.70, XGBoost at 679.96, and the baseline at 1567.54. I tuned CatBoost with 30 Optuna trials and retrained the final model on all 48,000 rows."

---

## [1:50 – 2:20] Results & Predictions

> "Here is our output verification:
> - On 3-fold CV, CatBoost achieved the best performance with an RMSE of $645.60, an MAE of $175.62, and an R2 of 0.82.
> - In our out-of-fold error analysis, binned into 5 quantile tiers, the model achieves 88.75% exact bucket accuracy and 99.14% within-one-bucket accuracy.
> - validation_predictions.csv contains all 12,000 predictions, matching the template IDs exactly with no missing values and all positive rates.
> - In scorer_results/candidate_december.png, the December curve provides a realistic daily trajectory for the Lexington to Fort Wayne route, varying cleanly with temporal features."

---

## [2:20 – 2:30] Closing

> "The full pipeline runs end-to-end with `python main.py`, or `python main.py --skip-cv` for a fast rerun. Code, artifacts, and this report are all in the GitHub repository. Thanks for the opportunity — happy to discuss any aspect of the approach."

---

**Notes for recording:**
- Show: terminal running `python main.py`, then `artifacts/cv_results.csv`, then `validation_predictions.csv` (first 10 rows), then `scorer_results/candidate_december.png`
- Keep screen font large enough to read
- No need to show full training output — skip ahead with a cut
