"""Pure-numpy value model: P(win | state features).

Loads weights saved by scripts/phase8_train.py and exposes a cheap
predict() usable inside the bot's decision loop. Supports a linear
(logistic-regression) model and an optional 1-hidden-layer MLP, both
stored as plain arrays so no torch/sklearn is needed at runtime.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[3] / ".data" / "phase8_value_model.npz"


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


class ValueModel:
    """Win-probability estimator over the feature vector from
    ``ml.features.features_from_state``."""

    def __init__(self, mean, std, kind, params):
        self.mean = np.asarray(mean, dtype=np.float64)
        self.std = np.asarray(std, dtype=np.float64)
        self.std[self.std == 0] = 1.0
        self.kind = str(kind)
        self.params = params

    @classmethod
    def load(cls, path: str | Path = DEFAULT_MODEL_PATH) -> "ValueModel":
        data = np.load(path, allow_pickle=False)
        kind = str(data["kind"]) if "kind" in data else "logreg"
        if kind == "mlp":
            params = {"W1": data["W1"], "b1": data["b1"], "W2": data["W2"], "b2": float(data["b2"])}
        else:
            params = {"w": data["w"], "b": float(data["b"])}
        return cls(data["mean"], data["std"], kind, params)

    def predict(self, features: Sequence[float]) -> float:
        x = (np.asarray(features, dtype=np.float64) - self.mean) / self.std
        if self.kind == "mlp":
            h = np.tanh(x @ self.params["W1"] + self.params["b1"])
            z = float(h @ self.params["W2"] + self.params["b2"])
        else:
            z = float(x @ self.params["w"] + self.params["b"])
        return float(_sigmoid(z))

    def predict_batch(self, X) -> np.ndarray:
        Xn = (np.asarray(X, dtype=np.float64) - self.mean) / self.std
        if self.kind == "mlp":
            H = np.tanh(Xn @ self.params["W1"] + self.params["b1"])
            Z = H @ self.params["W2"] + self.params["b2"]
        else:
            Z = Xn @ self.params["w"] + self.params["b"]
        return _sigmoid(Z)


_CACHED: ValueModel | None = None


def get_value_model(path: str | Path = DEFAULT_MODEL_PATH) -> ValueModel | None:
    """Process-cached model load; returns None if no model file exists."""
    global _CACHED
    if _CACHED is not None:
        return _CACHED
    try:
        _CACHED = ValueModel.load(path)
    except Exception:  # noqa: BLE001 — no model yet / unreadable
        return None
    return _CACHED
