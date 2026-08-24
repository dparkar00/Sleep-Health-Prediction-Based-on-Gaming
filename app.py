import streamlit as st
import joblib
import numpy as np
from pathlib import Path


st.set_page_config(
    page_title="Gaming Sleep Predictor",
    page_icon="🎮",
    layout="wide",
)


MODELS_DIR = Path("models")

MODEL_PATH = MODELS_DIR / "random_forest_model.pkl"
SCALER_PATH = MODELS_DIR / "scaler.pkl"
MAPPINGS_PATH = MODELS_DIR / "category_mappings.pkl"
THRESHOLD_PATH = MODELS_DIR / "best_threshold.pkl"


@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    mappings = joblib.load(MAPPINGS_PATH)

    if THRESHOLD_PATH.exists():
        threshold = joblib.load(THRESHOLD_PATH)
    else:
        threshold = 0.50

    return model, scaler, mappings, threshold


def encode(mappings, col, value):
    reverse = {v: k for k, v in mappings[col].items()}
    return reverse[value]


st.title("🎮 Gaming Sleep Predictor")
st.write(
    "Enter gaming and demographic information to predict whether someone may be at risk "
    "for poor or disrupted sleep."
)

if not MODEL_PATH.exists() or not SCALER_PATH.exists() or not MAPPINGS_PATH.exists():
    st.error(
        "Model artifacts were not found. Run `python src/train.py` first, then restart the app."
    )
    st.stop()


model, scaler, mappings, threshold = load_artifacts()

left_col, right_col = st.columns(2)

with left_col:
    age = st.slider("Age", 0, 100, 25, 1)
    daily_gaming_hours = st.slider(
        "Daily Gaming Hours",
        0.0,
        24.0,
        2.0,
        0.5,
    )

    gender = st.radio(
        "Gender",
        list(mappings["gender"].values()),
        horizontal=True,
    )

with right_col:
    platform = st.selectbox(
        "What Gaming Platform Do You Use?",
        list(mappings["gaming_platform"].values()),
    )

    genre = st.selectbox(
        "What Game Genre Do You Most Often Play?",
        list(mappings["game_genre"].values()),
    )

    game = st.selectbox(
        "What Is Your Primary Game?",
        list(mappings["primary_game"].values()),
    )


st.divider()

if st.button("Predict", type="primary"):
    input_data = np.array([[
        age,
        encode(mappings, "gender", gender),
        daily_gaming_hours,
        encode(mappings, "game_genre", genre),
        encode(mappings, "primary_game", game),
        encode(mappings, "gaming_platform", platform),
    ]])

    input_scaled = scaler.transform(input_data)

    probability_at_risk = model.predict_proba(input_scaled)[:, 1][0]
    pred = int(probability_at_risk >= threshold)

    st.subheader("Prediction Result")

    if pred == 0:
        st.success("✅ Low Risk: You are not at risk for poor sleep.")
    else:
        st.error("⚠️ High Risk: You may be at risk for poor sleep.")

    st.metric(
        label="Probability of Poor Sleep Risk",
        value=f"{probability_at_risk * 100:.1f}%",
    )

    st.caption(f"Decision threshold used by the model: {threshold:.2f}")

with st.expander("About this ML Ops workflow"):
    st.write(
        """
        This app uses saved artifacts from the training pipeline:
        - Random Forest model
        - RobustScaler
        - category mappings
        - tuned decision threshold

        The training pipeline loads the CSV, preprocesses the data, trains the model,
        tunes the threshold, runs GridSearchCV, and saves the final model artifacts.
        """
    )
