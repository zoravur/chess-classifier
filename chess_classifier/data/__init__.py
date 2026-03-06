"""
chess_classifier.data - Dataset preprocessing for the chess outcome classifier.

Usage:
    from chess_classifier.data import load_df_from_parquet, preprocess_df, to_dataset_arrays

    df = load_df_from_parquet("path/to/games.parquet", n_positions=100_000, shuffle_seed=42)
    X_df, y_df = preprocess_df(df)
    X_train, X_test, y_train, y_test = to_dataset_arrays(X_df, y_df, test_size=0.2, random_state=42)

Features:
    - white_rating, black_rating: normalized to (rating - 1500) / 400
    - ply: normalized to ply / 80, plus to_move derived from ply % 2
    - 768 one-hot features for piece positions (12 pieces × 64 squares)

Labels:
    - 0: white wins (1-0)
    - 1: black wins (0-1)
    - 2: draw (1/2-1/2)
"""

from .df_loader import download_dataset_from_huggingface, load_df_from_parquet
from .preprocess import (
    PIECES,
    SQUARES,
    preprocess_df,
    preprocess_features,
)

__all__ = [
    "load_df_from_parquet",
    "download_dataset_from_huggingface",
    "preprocess_df",
    "preprocess_features",
    "fen_to_features" "to_dataset_arrays",
    "SQUARES",
    "PIECES",
]
