import os
import logging
from pathlib import Path
from dotenv import load_dotenv

# Load env variables from .env if it exists
load_dotenv()

# Base project directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Dataset paths
DATA_DIR = BASE_DIR / "data"
RAW_DATA_PATH = os.getenv("RAW_DATA_PATH", str(DATA_DIR / "WA_Fn-UseC_-HR-Employee-Attrition.csv"))

# Model artifacts
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = os.getenv("MODEL_PATH", str(MODEL_DIR / "best_model.joblib"))

# Reports and figures
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
METRICS_PATH = REPORTS_DIR / "metrics.json"

# Settings
RANDOM_STATE = int(os.getenv("RANDOM_STATE", "42"))
TEST_SIZE = float(os.getenv("TEST_SIZE", "0.2"))

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Setup logging
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(BASE_DIR / "project.log", mode="a")
        ]
    )
