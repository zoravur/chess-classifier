import hydra
from omegaconf import DictConfig

from chess_classifier.data import fen_to_features
from chess_classifier.models import Model


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
