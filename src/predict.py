from pathlib import Path
import joblib

from preprocess import preprocess_single_input


def predict_sleep_disruption(input_data: dict, models_dir: str = "models") -> dict:
    """
    Predicts sleep disruption risk for one input row.

    Returns:
    - prediction: 0 or 1
    - label: Healthy Sleep or At Risk
    - probability_at_risk
    - threshold
    """
    models_path = Path(models_dir)

    model = joblib.load(models_path / "random_forest_model.pkl")
    threshold = joblib.load(models_path / "best_threshold.pkl")

    X_scaled = preprocess_single_input(input_data, models_dir=models_dir)

    probability_at_risk = model.predict_proba(X_scaled)[:, 1][0]
    prediction = int(probability_at_risk >= threshold)

    label = "At Risk" if prediction == 1 else "Healthy Sleep"

    return {
        "prediction": prediction,
        "label": label,
        "probability_at_risk": float(probability_at_risk),
        "threshold": float(threshold),
    }


if __name__ == "__main__":
    sample_input = {
        "age": 22,
        "gender": "Male",
        "daily_gaming_hours": 4,
        "game_genre": "Action",
        "primary_game": "Fortnite",
        "gaming_platform": "PC",
    }

    result = predict_sleep_disruption(sample_input)
    print(result)
