from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

TRAIN_TEST_PATH = DATA_DIR / "train_test.csv"
VALIDATION_PATH = DATA_DIR / "validation.csv"
VALIDATION_TEMPLATE_PATH = DATA_DIR / "validation_predictions_template.csv"
DECEMBER_INPUTS_PATH = DATA_DIR / "december_chart_inputs.csv"

VALIDATION_PREDICTIONS_PATH = BASE_DIR / "validation_predictions.csv"
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
SCORER_RESULTS_DIR = BASE_DIR / "scorer_results"
ARTIFACTS_DIR = BASE_DIR / "artifacts"

# Modeling & Evaluation Constants
TARGET_COL = "posted_rate"
RANDOM_STATE = 42

# Ensure required directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
SCORER_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
