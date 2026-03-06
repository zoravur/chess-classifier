import chess
import numpy as np
import pandas as pd

from chess_classifier.data import preprocess_features


def fen_to_features(fen: str, white_rating: int, black_rating: int) -> np.ndarray:
    board = chess.Board(fen)

    square_feats = {}

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            square_feats[chess.square_name(square)] = piece.symbol()
        else:
            square_feats[chess.square_name(square)] = ""

    # Ply from fullmove counter
    ply = 2 * board.fullmove_number - (1 if board.turn == chess.WHITE else 0)
    to_move = board.turn = chess.WHITE

    df = pd.DataFrame(
        [
            dict(
                ply=ply,
                to_move=to_move,
                white_rating=white_rating,
                black_rating=black_rating,
                **square_feats,
            )
        ]
    )
    X_df = preprocess_features(df)
    return X_df.to_numpy()
