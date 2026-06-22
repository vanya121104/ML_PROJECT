# HR Employee Attrition Prediction & Insights

An end-to-end Machine Learning pipeline to predict employee attrition risk, identify core drivers of turnover, and explain model decisions using SHAP values. Includes a reproducible scikit-learn preprocessing and modeling pipeline, a FastAPI service layer, and an interactive Streamlit dashboard.

---

## 📂 Project Structure

```
ml-project/
├── data/
│   └── WA_Fn-UseC_-HR-Employee-Attrition.csv  # Raw dataset
├── notebooks/
│   └── eda_and_data_dictionary.ipynb           # Jupyter EDA Notebook
├── src/
│   ├── config.py                              # Configuration parameters
│   ├── preprocessing.py                       # Ingestion, validation & cleaning
│   ├── pipeline.py                            # Scikit-learn/Imblearn pipelines
│   ├── train.py                               # Training & hyperparameter tuning
│   ├── evaluate.py                            # Metrics report and curve plotting
│   ├── explain.py                             # SHAP explanations generator
│   ├── api.py                                 # FastAPI Application
│   └── dashboard.py                           # Streamlit Dashboard
├── tests/
│   ├── test_preprocessing.py                  # Unit tests for preprocessing
│   └── test_prediction.py                     # Unit tests for model predictions
├── reports/
│   ├── figures/                               # Diagnostic & SHAP summary plots
│   ├── data_dictionary.md                     # Data definitions and types
│   └── business_attrition_report.md           # Business insights & policy recommendations
├── Dockerfile.api                             # Dockerfile for FastAPI
├── Dockerfile.dashboard                       # Dockerfile for Streamlit
├── docker-compose.yml                         # Compose configuration for local services
├── .github/workflows/ci.yml                   # GitHub Actions pipeline config
├── requirements.txt                           # Third-party dependencies
├── README.md                                  # Setup & user guide
└── .env.example                               # Template env configuration
```

---

## 🚀 Local Quickstart

### 1. Environment Setup
Clone this repository to your workspace, create a virtual environment, and install dependencies:

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

Create an active configuration file:
```bash
cp .env.example .env
```

### 2. Ingest and Train Model
Running the training script will:
* Load the raw dataset from `data/`.
* Export a data dictionary to `reports/data_dictionary.md`.
* Split data, apply SMOTE to resolve class imbalance, and tune multiple models (Logistic Regression, Random Forest, XGBoost, LightGBM, CatBoost, Gradient Boosting, SVM).
* Select the best classifier based on F1-Score & ROC-AUC.
* Generate diagnostic plots under `reports/figures/`.
* Save the serialized model artifact to `models/best_model.joblib`.

```bash
python src/train.py
```

### 3. Run Unit Tests
Run standard automated validation checks:
```bash
pytest tests/
```

### 4. Launch FastAPI Server
Run the API to expose prediction and explanation interfaces:
```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
```
* Access interactive Swagger documentation at: **`http://localhost:8000/docs`**

#### Endpoint API Details:
* **`GET /health`**: Returns system & model status.
* **`POST /predict`**: Takes an employee's attributes and returns attrition probability (0 to 1) and predicted class label.
* **`POST /explain`**: Takes an employee's attributes and returns a sorted list of SHAP explanation impacts.

### 5. Launch Streamlit Dashboard
Run the dashboard to explore analytics and evaluate risk profiles:
```bash
streamlit run src/dashboard.py
```
* Access the UI at: **`http://localhost:8501`**

---

## 🐳 Docker Deployment

To build and run both the API and Streamlit Dashboard concurrently in lightweight containers:

```bash
# Build and run containers
docker-compose up --build
```
* API runs on: **`http://localhost:8000`**
* Dashboard runs on: **`http://localhost:8501`**

To stop the services:
```bash
docker-compose down
```
