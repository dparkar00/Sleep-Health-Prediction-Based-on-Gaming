from pathlib import Path
import joblib
import pandas as pd
from sklearn.preprocessing import RobustScaler


FEATURE_COLUMNS = [
    "age",
    "gender",
    "daily_gaming_hours",
    "game_genre",
    "primary_game",
    "gaming_platform",
]


def create_binary_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create binary target variable based on sleep quality indicators.

    Target:
    0 = Healthy Sleep
    1 = At Risk / Sleep Disrupted

    This follows the notebook logic:
    - sleep_hours < 7
    - sleep_quality is poor or very poor
    - sleep_disruption_frequency is often, always, or frequently
    - target is 1 if the participant meets at least 2 conditions
    """
    df = df.copy()
    df["sleep_disrupted"] = 0

    if "sleep_hours" in df.columns:
        condition1 = df["sleep_hours"] < 7
        print(f"Condition 1 (sleep_hours < 7): {condition1.sum()} participants")
    else:
        condition1 = pd.Series([False] * len(df), index=df.index)
        print("Condition 1: 'sleep_hours' column not found")

    if "sleep_quality" in df.columns:
        condition2 = (
            df["sleep_quality"]
            .astype(str)
            .str.lower()
            .isin(["poor", "very poor"])
        )
        print(f"Condition 2 (poor/very poor sleep quality): {condition2.sum()} participants")
    else:
        condition2 = pd.Series([False] * len(df), index=df.index)
        print("Condition 2: 'sleep_quality' column not found")

    if "sleep_disruption_frequency" in df.columns:
        condition3 = (
            df["sleep_disruption_frequency"]
            .astype(str)
            .str.lower()
            .isin(["often", "always", "frequently"])
        )
        print(f"Condition 3 (frequent disruption): {condition3.sum()} participants")
    else:
        condition3 = pd.Series([False] * len(df), index=df.index)
        print("Condition 3: 'sleep_disruption_frequency' column not found")

    disruption_score = (
        condition1.astype(int)
        + condition2.astype(int)
        + condition3.astype(int)
    )

    df["sleep_disrupted"] = (disruption_score >= 2).astype(int)

    print("\n--- Disruption Score Distribution ---")
    print(f"Score 0: {(disruption_score == 0).sum()} participants")
    print(f"Score 1: {(disruption_score == 1).sum()} participants")
    print(f"Score 2: {(disruption_score == 2).sum()} participants")
    print(f"Score 3: {(disruption_score == 3).sum()} participants")

    return df


def preprocess_training_data(
    df: pd.DataFrame,
    models_dir: str = "models",
):
    """
    Preprocesses the training data following the notebook's steps:
    - select available feature columns
    - separate X and y
    - fill missing numeric values with median
    - fill missing categorical values with mode
    - encode categorical values with category codes
    - save category mappings
    - fit RobustScaler
    - save scaler
    """
    models_path = Path(models_dir)
    models_path.mkdir(parents=True, exist_ok=True)

    df = create_binary_target(df)

    target_counts = df["sleep_disrupted"].value_counts().sort_index()

    print("\n--- Final Target Distribution ---")
    print(
        f"Healthy sleep (0): {target_counts.get(0, 0)} participants "
        f"({target_counts.get(0, 0) / len(df) * 100:.1f}%)"
    )
    print(
        f"Disrupted sleep (1): {target_counts.get(1, 0)} participants "
        f"({target_counts.get(1, 0) / len(df) * 100:.1f}%)"
    )

    available_features = [col for col in FEATURE_COLUMNS if col in df.columns]
    missing_features = [col for col in FEATURE_COLUMNS if col not in df.columns]

    print(f"\n✓ Available features: {available_features}")
    if missing_features:
        print(f"⚠️ Missing features: {missing_features}")

    if not available_features:
        raise ValueError("None of the expected feature columns were found in the CSV.")

    X = df[available_features].copy()
    y = df["sleep_disrupted"].copy()

    print(f"\nFeatures (X) shape: {X.shape}")
    print(f"Target (y) shape: {y.shape}")
    print(f"Feature columns: {X.columns.tolist()}")

    print("\nMissing values before imputation:")
    print(X.isnull().sum())

    imputation_values = {}

    if X.isnull().sum().sum() > 0:
        for col in X.columns:
            if X[col].dtype in ["int64", "float64"]:
                fill_value = X[col].median()
                X[col] = X[col].fillna(fill_value)
                imputation_values[col] = fill_value
                print(f"  {col}: filled with median ({fill_value})")
            else:
                fill_value = X[col].mode()[0] if not X[col].mode().empty else "unknown"
                X[col] = X[col].fillna(fill_value)
                imputation_values[col] = fill_value
                print(f"  {col}: filled with mode ('{fill_value}')")
    else:
        print("✓ No missing values found")
        for col in X.columns:
            if X[col].dtype in ["int64", "float64"]:
                imputation_values[col] = X[col].median()
            else:
                imputation_values[col] = X[col].mode()[0] if not X[col].mode().empty else "unknown"

    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()
    category_mappings = {}

    print(f"\nCategorical features to encode: {categorical_features}")

    for col in categorical_features:
        cat = X[col].astype("category")
        X[col] = cat.cat.codes
        category_mappings[col] = dict(enumerate(cat.cat.categories))
        print(f"  {col}: encoded to numeric codes")

    print("\n✓ All features are now numeric")
    print(f"Final feature shape: {X.shape}")

    robust_scaler = RobustScaler()
    X_scaled = robust_scaler.fit_transform(X)

    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

    print("\nBefore RobustScaler:")
    print(f"  Value range: [{X.values.min():.2f}, {X.values.max():.2f}]")
    print(f"  Mean: {X.values.mean():.3f}")
    print(f"  Std: {X.values.std():.3f}")

    print("\nAfter RobustScaler:")
    print(f"  Value range: [{X_scaled.min():.2f}, {X_scaled.max():.2f}]")
    print(f"  Mean: {X_scaled.mean():.3f}")
    print(f"  Std: {X_scaled.std():.3f}")

    print("\nFirst 5 rows after RobustScaler:")
    print(X_scaled_df.head())

    print("\nFinal dataset info:")
    print(f"  - Total samples: {len(df)}")
    print(f"  - Features: {X.shape[1]} columns")
    print("  - Target: Binary (0=Healthy, 1=At Risk)")
    print(
        f"  - Target distribution: {target_counts.get(0, 0)}/"
        f"{target_counts.get(1, 0)} "
        f"({target_counts.get(1, 0) / len(df) * 100:.1f}% at risk)"
    )

    joblib.dump(category_mappings, models_path / "category_mappings.pkl")
    joblib.dump(robust_scaler, models_path / "scaler.pkl")
    joblib.dump(imputation_values, models_path / "imputation_values.pkl")
    joblib.dump(X.columns.tolist(), models_path / "feature_columns.pkl")

    print("\n✓ Preprocessing complete! Data ready for model training.")

    return X_scaled, y, X.columns.tolist(), robust_scaler, category_mappings


def preprocess_single_input(input_data: dict, models_dir: str = "models"):
    """
    Preprocess one user input row for prediction using saved artifacts.
    """
    models_path = Path(models_dir)

    scaler = joblib.load(models_path / "scaler.pkl")
    category_mappings = joblib.load(models_path / "category_mappings.pkl")
    imputation_values = joblib.load(models_path / "imputation_values.pkl")
    feature_columns = joblib.load(models_path / "feature_columns.pkl")

    X = pd.DataFrame([input_data])

    for col in feature_columns:
        if col not in X.columns:
            X[col] = imputation_values.get(col, 0)

    X = X[feature_columns].copy()

    for col in X.columns:
        if pd.isna(X.loc[0, col]):
            X.loc[0, col] = imputation_values.get(col, 0)

    for col, mapping in category_mappings.items():
        if col in X.columns:
            reverse_mapping = {str(v): k for k, v in mapping.items()}
            raw_value = str(X.loc[0, col])
            X.loc[0, col] = reverse_mapping.get(raw_value, -1)

    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0)

    X_scaled = scaler.transform(X)

    return X_scaled
