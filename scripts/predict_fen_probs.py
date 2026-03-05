import chess
import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig

from chess_classifier.data import preprocess_features
from chess_classifier.models import Model


def fen_to_features(fen: str, white_rating: int, black_rating: int) -> np.ndarray:
    board = chess.Board(fen)

    square_feats = {}

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            square_feats[chess.square_name(square)] = piece.symbol()
        else:
            square_feats[chess.square_name(square)] = ""

    # square_feats = {f'{sq}_{p}': 0 for p in PIECES for sq in SQUARES}

    # for square in chess.SQUARES:
    #     sq_name = chess.square_name(square)
    #     piece = board.piece_at(square)
    #     if piece:
    #         square_feats[f'{sq_name}_{piece.symbol()}'] = 1

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


@hydra.main(version_base=None, config_path="../configs", config_name="predict")
def main(cfg: DictConfig):
    model = Model.load(cfg.predict.model_path)

    fen = input("FEN: ")
    white_rating = int(input("White Elo: "))
    black_rating = int(input("Black Elo: "))

    inp = fen_to_features(fen, white_rating=white_rating, black_rating=black_rating)

    ((white_win_prob, black_win_prob, draw_prob),) = model.predict_proba(inp)

    print(
        ", ".join(
            f"{k}: {float(v*100):.2f}%"
            for k, v in dict(White=white_win_prob, Black=black_win_prob, Draw=draw_prob).items()
        )
    )


if __name__ == "__main__":
    main()
