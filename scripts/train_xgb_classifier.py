import hydra
from omegaconf import DictConfig

from chess_classifier.data import (
    download_dataset_from_huggingface,
    load_df_from_parquet,
    preprocess_df,
    to_dataset_arrays,
)
from chess_classifier.evaluation import evaluate_model
from chess_classifier.models import Model


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg: DictConfig):
    hf_url = f"{cfg.data.hf_repo}/{cfg.data.hf_file}"
    download_dataset_from_huggingface(hf_url=hf_url, save_path=cfg.data.save_path, limit=cfg.data.limit)
    df = load_df_from_parquet(
        parquet_path=cfg.data.save_path,
        n_positions=cfg.data.limit,
        shuffle_seed=cfg.training.seed,
    )
    X_df, y_df = preprocess_df(df)
    X_train, X_test, y_train, y_test = to_dataset_arrays(X_df, y_df, test_size=0.2, random_state=cfg.training.seed)

    model = Model(seed=cfg.training.seed, learning_rate=cfg.training.lr, **cfg[cfg.training.model])
    model.fit(X_train, y_train)

    model.save(cfg.training.model_path)
    print("Train:", evaluate_model(model, X_train, y_train))
    print("Test:", evaluate_model(model, X_test, y_test))


if __name__ == "__main__":
    main()
