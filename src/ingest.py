from pathlib import Path
import pandas as pd


def load_dataset(csv_path: str = "data/gaming_health_data.csv") -> pd.DataFrame:
    """
    Loads the local CSV dataset.
    """
    path = Path(csv_path)

    if not path.exists():
        raise FileNotFoundError(
            f"CSV file not found at {csv_path}. "
        )

    df = pd.read_csv(path)

    print("\n✓ Dataset loaded successfully!")
    print(f"Shape: {df.shape}")
    print("\nFirst 5 rows:")
    print(df.head())

    print("\nData Types and Non-null Counts:")
    print(df.info())

    print("\nMissing Values Per Column:")
    print(df.isnull().sum())

    print("\nStatistical Summary:")
    print(df.describe(include="all"))

    return df
