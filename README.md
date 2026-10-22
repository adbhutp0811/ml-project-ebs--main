# Student Performance Indicator — End-to-End ML Project

A complete end-to-end machine learning project that predicts a student's **Maths score (out of 100)** from demographic and academic-profile features (gender, ethnicity, parental education, lunch type, test preparation course, reading score, writing score), and serves the trained model through a small Flask web app.

The project follows the standard ML life cycle:

> Problem Statement → Data Collection → Data Checks → Exploratory Data Analysis → Data Pre-processing → Model Training → Model Evaluation → Model Deployment (Flask)

---

## 1. Problem Statement

Understand how a student's test performance is affected by other variables — Gender, Race/Ethnicity, Parental Level of Education, Lunch Type and Test Preparation Course — and build a regression model that predicts the Maths score from those inputs plus the reading and writing scores.

## 2. Dataset

| Property | Value |
| --- | --- |
| Source | [Kaggle — Students Performance in Exams](https://www.kaggle.com/datasets/spscientist/students-performance-in-exams?datasetId=74977) |
| Location | `notebook/data/stud.csv` |
| Rows | 1000 students |
| Columns | 8 (5 categorical, 3 numerical) |
| Missing values | None |
| Duplicates | None |

### Feature description

| Column | Type | Description |
| --- | --- | --- |
| `gender` | categorical | male / female |
| `race_ethnicity` | categorical | group A – group E |
| `parental_level_of_education` | categorical | some college, associate's degree, bachelor's degree, master's degree, high school, some high school |
| `lunch` | categorical | standard / free-reduced |
| `test_preparation_course` | categorical | completed / none |
| `math_score` | numerical | **Target variable** (0–100) |
| `reading_score` | numerical | 0–100 |
| `writing_score` | numerical | 0–100 |

## 3. Tech Stack

- **Language:** Python 3.8+ (developed on 3.14)
- **Data & EDA:** NumPy, Pandas, Matplotlib, Seaborn
- **ML / Modelling:** scikit-learn, XGBoost, CatBoost
- **Serving:** Flask + Jinja2 templates
- **Serialization:** Pickle (via `src/utils.py`)
- **Packaging:** `setuptools` (`setup.py`), `pip`/`requirements.txt`

## 4. Project Structure

```
mlproject/
├── app.py                          # Flask application entry point
├── setup.py                        # Package metadata (pip install -e .)
├── requirements.txt                # Runtime dependencies
├── artifacts/                      # Serialized model + processed datasets
│   ├── model.pkl                   # Trained LinearRegression estimator
│   ├── preprocessor.pkl            # Fitted ColumnTransformer (OneHot + StandardScaler)
│   ├── data.csv                    # Full cleaned dataset
│   ├── train.csv                   # Training split
│   └── test.csv                    # Test split
├── notebook/
│   ├── 1 . EDA STUDENT PERFORMANCE .ipynb   # Full EDA & visualisation
│   ├── 2. MODEL TRAINING.ipynb               # Model comparison & evaluation
│   └── data/stud.csv                         # Raw source dataset
├── src/
│   ├── __init__.py
│   ├── exception.py                # CustomException with file/line details
│   ├── logger.py                   # Timestamped log files in logs/
│   ├── utils.py                    # save_object, load_object, evaluate_models
│   └── pipeline/
│       ├── train_pipeline.py       # (placeholder — not implemented yet)
│       └── predict_pipeline.py     # CustomData + PredictPipeline
├── templates/
│   ├── index.html                  # Landing page
│   └── home.html                   # Prediction form + result card
├── logs/                           # Runtime logs (timestamped .log files)
└── catboost_info/                  # CatBoost training artefacts
```

## 5. Setup

### 5.1 Prerequisites

- Python 3.8 or newer
- pip

### 5.2 Create a virtual environment

**Windows (PowerShell)**

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**macOS / Linux**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 5.3 Install dependencies

```bash
pip install -r requirements.txt
```

Optionally install the project itself in editable mode:

```bash
pip install -e .
```

## 6. How to Run

Start the Flask server from the **project root** (the app resolves `artifacts/` relative to the current working directory):

```bash
python app.py
```

Then open <http://127.0.0.1:5000>.

### Routes

| Method | Route | Description |
| --- | --- | --- |
| `GET` | `/` | Renders the welcome landing page (`templates/index.html`) |
| `GET` | `/predictdata` | Renders the prediction form (`templates/home.html`) |
| `POST` | `/predictdata` | Reads the submitted form, builds a feature DataFrame, runs the pipeline and re-renders the form with the predicted Maths score |

### Example request

```bash
curl -X POST http://127.0.0.1:5000/predictdata \
  -d "gender=male" \
  -d "ethnicity=group B" \
  -d "parental_level_of_education=bachelor's degree" \
  -d "lunch=standard" \
  -d "test_preparation_course=none" \
  -d "reading_score=72" \
  -d "writing_score=74"
```

## 7. How It Works

1. **`app.py`** collects the form fields and builds a `CustomData` object.
2. **`CustomData.get_data_as_data_frame()`** (`src/pipeline/predict_pipeline.py:53`) converts it into a one-row Pandas DataFrame with the same column names/order used at training time.
3. **`PredictPipeline.predict()`** (`src/pipeline/predict_pipeline.py:12`) loads `artifacts/preprocessor.pkl` and `artifacts/model.pkl` via `load_object`, applies `preprocessor.transform(...)` and then `model.predict(...)`.
4. The predicted Maths score is passed back to `templates/home.html` and displayed in the result card.

### Pre-processing (training & inference)

A `sklearn.compose.ColumnTransformer` (serialized as `artifacts/preprocessor.pkl`) applies:

- `OneHotEncoder` on the 5 categorical columns
- `StandardScaler` on the 2 numerical input columns (`reading_score`, `writing_score`)

### Model evaluation (notebook `2. MODEL TRAINING.ipynb`)

Nine regressors were trained on an 80/20 split (`test_size=0.2`, `random_state=42`) and compared with MAE, RMSE and R²:

| Model | Test R² |
| --- | --- |
| **Ridge** | **0.8806** |
| **Linear Regression** (served) | **0.8803** |
| CatBoosting Regressor | 0.8516 |
| AdaBoost Regressor | 0.8498 |
| Random Forest Regressor | 0.8473 |
| Lasso | 0.8253 |
| XGBRegressor | 0.8216 |
| K-Neighbors Regressor | 0.7838 |
| Decision Tree | 0.7603 |

The deployed estimator is a **Linear Regression** (`artifacts/model.pkl`) with test R² ≈ 0.88, RMSE ≈ 5.40 and MAE ≈ 4.22 — the decision tree's near-perfect training score (R² 0.9997) is a clear overfitting case.

## 8. Retraining the Model

Training currently lives in the notebook. To refresh the artifacts:

1. Open `notebook/2. MODEL TRAINING.ipynb` and run it top to bottom against `notebook/data/stud.csv`.
2. Persist the fitted objects into `artifacts/` using the helper in `src/utils.py`:

```python
from src.utils import save_object

save_object("artifacts/preprocessor.pkl", preprocessor)
save_object("artifacts/model.pkl", model)
```

3. Restart the Flask app so the new artefacts are picked up.

`src/utils.py` also provides `evaluate_models(...)`, which runs a `GridSearchCV` over a dictionary of models and parameter grids and returns a `{model_name: test_r2_score}` report.

## 9. Key Insights from EDA

- Female students have a higher overall/average score, while male students score higher in Maths.
- Students receiving **standard lunch** perform better than those on free/reduced lunch — across both genders.
- **Group E** students score highest and **Group A** lowest, indicating a socioeconomic gradient.
- Parental education shows a mild effect: master's/bachelor's degree parents tend to have higher-scoring children.
- Students who **completed the test preparation course** score higher across all three subjects.
- All three scores increase roughly linearly with one another; Maths is the weakest subject overall.

## 10. Important Notes & Known Issues

- **Run from the project root.** `PredictPipeline` loads `artifacts/model.pkl` using a relative path, so starting the app from another directory will raise `FileNotFoundError`.
- **Form field naming.** The ethnicity dropdown posts `ethnicity` (not `race_ethnicity`); `app.py:25` maps it to the `race_ethnicity` feature. Renaming the field requires updating both the template and the mapping.
- **Sklearn version pinning.** `artifacts/*.pkl` were pickled with scikit-learn 1.3.2. Running against a much newer scikit-learn raises `InconsistentVersionWarning` and may break at inference — reinstall scikit-learn 1.3.2 or retrain the artifacts.
- **`src/pipeline/train_pipeline.py` is empty.** Training is notebook-driven; there is no scripted, reproducible retraining entry point yet.
- **Unpinned dependencies.** `requirements.txt` lists packages without version constraints, so a fresh install can pull incompatible versions.
- **No automated tests** exist for the pipeline or the Flask routes.
- **Debug mode** is enabled (`app.run(debug=True)`) — disable it before any non-local deployment.

## 11. Future Improvements

- Implement `src/pipeline/train_pipeline.py` as a scriptable retraining pipeline (load → preprocess → train → evaluate → save artifacts).
- Pin dependency versions in `requirements.txt` and rebuild the artifacts against them.
- Add a `tests/` suite covering the pipeline and the HTTP routes.
- Compare gradient-boosting models (CatBoost/XGBoost) against the linear baseline; CatBoost was within 0.03 R² and handles categoricals natively.
- Add cross-validated metrics and persist an evaluation report alongside the model.
- Replace the ad-hoc HTML with a templated layout and add input validation for score ranges.

## 12. License

Released for educational and portfolio purposes. The dataset originates from Kaggle and remains subject to its original license.
