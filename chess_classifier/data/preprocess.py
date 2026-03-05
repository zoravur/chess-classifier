# This code does preprocessing of the dataset for feeding into the ML model.

# It requires pandas.
# TODO: make it so that the whole dataset doesn't have to fit in memory
import pandas as pd
from sklearn.model_selection import train_test_split

RESULT_MAP = {"1-0": 0, "0-1": 1, "1/2-1/2": 2}
PIECES = list("prnbqkPRNBQK")
FILES = (chr(file + ord("a")) for file in range(8))
RANKS = (str(i) for i in range(1, 9))
SQUARES = [f + r for r in RANKS for f in FILES]


def preprocess_features(df: pd.DataFrame) -> pd.DataFrame:
    # Select out the features and labels
    features = ["white_rating", "black_rating", "ply"]

    one_hot_df = pd.DataFrame({f"{col}_{piece}": df[col] == piece for piece in PIECES for col in SQUARES})
    X_df = pd.concat([df[features], one_hot_df], axis="columns")
    X_df["ply"] = X_df["ply"] / 80  # average game length
    X_df["to_move"] = X_df["ply"] % 2  # turn (black or white)

    # Scale ratings to make it easier for the optimizer. Note that this is not strictly required for gradient boosting.
    X_df["white_rating"] = (X_df["white_rating"] - 1500) / 400
    X_df["black_rating"] = (X_df["black_rating"] - 1500) / 400

    return X_df


def preprocess_df(df: pd.DataFrame):
    """Extract features and labels from raw game dataframe."""

    X_df = preprocess_features(df)
    y_df = df["result"].map(RESULT_MAP)
    return X_df, y_df


def to_dataset_arrays(X_df: pd.DataFrame, y_df: pd.Series, test_size: float, random_state: int):
    """Convert to numpy arrays and split into train/test."""
    X, y = X_df.to_numpy(), y_df.to_numpy()
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
