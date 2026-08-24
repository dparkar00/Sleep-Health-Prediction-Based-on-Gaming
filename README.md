# Sleep-Health-Prediction-Based-on-Gaming
# Sleep Disruption Prediction - AI/ML Ops Project

## Overview

This project implements an end-to-end machine learning pipeline to predict sleep disruption risk based on gaming behavior patterns. The model classifies users as having healthy or disrupted sleep patterns using demographic data, gaming behavior metrics, and sleep-quality indicators.

The project demonstrates full MLOps principles including:
- Modular code architecture
- Model versioning and persistence
- Interactive web deployment
- Reproducible pipeline

## Live Demo

Try the application here: [https://sleepdisruption-6tmjvvykhxZjwzwnisbwnq.streamlit.app/](https://sleepdisruption-6tmjvvykhxZjwzwnisbwnq.streamlit.app/)

## Problem Statement

Long or late gaming sessions can disrupt circadian rhythms and reduce overall sleep quality, leading to:
- Higher stress levels
- Weakened emotional regulation
- Increased risk of anxiety and depressive symptoms

Our goal is to provide a data-driven tool that helps users understand their personal sleep-health risk based on their gaming habits.

## Dataset

- **Source**: Public Kaggle dataset containing demographic data, gaming behavior metrics, and sleep-quality indicators
- **Size**: 1,000 samples
- **Features**: Age, gender, daily gaming hours, session timing, genre preferences
- **Target**: Binary classification (Healthy vs At Risk sleep disruption)
- **Preprocessing**: Applied RobustScaler for feature scaling
- **Train/Test Split**: 80% (800 samples) / 20% (200 samples)

## Model

### Algorithm: Random Forest Classifier

**Why Random Forest?**
- Handles mixed numerical/categorical features
- Captures complex interactions (e.g., age × gaming hours × genres)
- Robust to extreme values
- Resists overfitting compared to single decision trees

### Hyperparameter Tuning

Optimal parameters found via Grid Search with 5-Fold Cross Validation:

| Parameter | Optimal Value | Rationale |
|-----------|---------------|-----------|
| n_estimators | 300 | More trees = lower variance, stable predictions |
| max_depth | 10 | Limits tree depth to prevent overfitting |
| min_samples_split | 10 | More samples to split → more generalized rules |
| min_samples_leaf | 2 | Smoother decision boundaries |
| Threshold | 0.30 | Tuned to prioritize recall |

### Final Performance Metrics

- **F1 Score**: 0.851
- **ROC-AUC**: 0.765
- **PR-AUC**: 0.786
- **Accuracy**: 79%

### Feature Importance (Top Features)

1. Daily Gaming Hours
2. Age
3. Primary Game Genre

## Project Structure

```
├── data/
│   └── (dataset files)
├── notebooks/
│   └── exploratory_analysis.ipynb
├── models/
│   ├── random_forest_model.pkl
│   └── preprocessing_artifacts.pkl
├── src/
│   ├── data_preprocessing.py
│   ├── model_training.py
│   ├── model_evaluation.py
│   └── utils.py
├── app/
│   └── streamlit_app.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation & Setup

### Prerequisites

- Python 3.8+
- pip

### Local Development

1. Clone the repository:
```bash
git clone https://github.com/yourusername/sleep-disruption-prediction.git
cd sleep-disruption-prediction
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the Streamlit app locally:
```bash
streamlit run app/streamlit_app.py
```

## Deployment Architecture

The model is deployed as a cloud-hosted interactive web application using:

- **Python + scikit-learn**: Model training and inference
- **Streamlit**: Interactive UI and deployment layer
- **joblib**: Model persistence (saved .pkl files)
- **GitLab**: Version control and CI/CD

### How Predictions Are Exposed

1. User inputs are collected via Streamlit UI components (sliders, dropdowns)
2. Inputs are converted to structured data format
3. Data passes through the same preprocessing pipeline as training
4. Model generates prediction with probability score
5. Results are displayed instantly to the user

### REST API Integration

The application supports both UI-based and programmatic access through REST API endpoints:

**Streamlit UI Service Layer:**
- Provides a front-end interface for the ML model
- User inputs captured through UI components
- Inputs programmatically converted into structured data
- Data processed using the same pipeline as training
- Predictions returned instantly to the user

**FastAPI REST Endpoint Integration:**
The same prediction logic can be wrapped in a FastAPI REST API:

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class UserFeatures(BaseModel):
    age: int
    gender: str
    daily_gaming_hours: float
    session_timing: str
    primary_genre: str

@app.post("/predict")
async def predict(user: UserFeatures):
    # Preprocess input
    # Run model inference
    # Return prediction
    return {"prediction": prediction, "probability": probability}
```

**Consumption Methods:**

| Method | REST API | Streamlit UI |
|--------|----------|--------------|
| Request Format | JSON request (POST) | Form inputs (UI) |
| Processing | Endpoint handles request | Python function handles input |
| Response | Returns JSON response | Displays result on screen |
| Use Case | Used by systems/apps | Used by end users |

This dual approach ensures the model is accessible both through a user-friendly interface and as a scalable microservice for integration with other applications.
