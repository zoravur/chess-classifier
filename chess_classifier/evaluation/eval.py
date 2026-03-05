import numpy as np
from sklearn.metrics import log_loss

# should be abstract class, currently just points to XGB
from chess_classifier.models import Model


def evaluate_model(model: Model, X_test: np.ndarray, y_test: np.ndarray):
    # get calibrated probabilities
    y_probs = model.predict_proba(X_test)
    ll = log_loss(y_test, y_probs)

    # final predictions
    y_preds = y_probs.argmax(axis=-1)
    correct_count = (y_preds == y_test).sum()
    total_count = y_test.shape[0]
    accuracy = correct_count / total_count

    return dict(
        log_loss=ll,
        correct_count=correct_count,
        total_count=total_count,
        accuracy=accuracy,
    )
