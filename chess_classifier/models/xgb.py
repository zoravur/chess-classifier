# The model architecture
import numpy as np
from xgboost import XGBClassifier


class Model:  # maybe rename, I want to save `Model` for the name of an abstract class.
    def __init__(self, n_estimators, max_depth, learning_rate, seed):
        self.xgb = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            seed=seed,
        )

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        return self.xgb.fit(X_train, y_train)

    def predict(self, X_test: np.ndarray):
        y_preds = self.xgb.predict(X_test)
        return y_preds

    def predict_proba(self, X_test: np.ndarray):
        y_probs = self.xgb.predict_proba(X_test)
        return y_probs

    def save(self, fname: str):
        self.xgb.save_model(fname=fname)

    @classmethod
    def load(cls, fname: str) -> "Model":
        model = cls.__new__(cls)
        model.xgb = XGBClassifier()
        model.xgb.load_model(fname=fname)
        return model
