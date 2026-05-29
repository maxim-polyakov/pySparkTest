"""PyFunc ALS для MLflow serve (без Spark). Подключается через code_paths при log_model."""
from __future__ import annotations

from typing import Any

import mlflow.pyfunc
import numpy as np


def _vector_to_numpy(vec: Any) -> np.ndarray:
    if hasattr(vec, "toArray"):
        return np.asarray(vec.toArray(), dtype=np.float32)
    return np.asarray(vec, dtype=np.float32)


class MovielensAlsPyFunc(mlflow.pyfunc.PythonModel):
    """Топ-K рекомендаций по user/item factors."""

    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        import pandas as pd

        user = pd.read_parquet(context.artifacts["user_factors"])
        item = pd.read_parquet(context.artifacts["item_factors"])
        self.user_ids = user["id"].astype(int).values
        self.item_ids = item["id"].astype(int).values
        self.user_feats = np.stack(
            [_vector_to_numpy(x) for x in user["features"]], dtype=np.float32
        )
        self.item_feats = np.stack(
            [_vector_to_numpy(x) for x in item["features"]], dtype=np.float32
        )
        self.user_index = {int(u): i for i, u in enumerate(self.user_ids)}

    def predict(self, context: mlflow.pyfunc.PythonModelContext, model_input: Any):
        import pandas as pd

        df = (
            model_input
            if isinstance(model_input, pd.DataFrame)
            else pd.DataFrame(model_input)
        )
        if "userId" not in df.columns:
            raise ValueError("Ожидается колонка userId")

        default_k = 10
        if "k" in df.columns and len(df):
            default_k = int(df["k"].iloc[0])

        rows: list[dict[str, Any]] = []
        for _, row in df.iterrows():
            uid = int(row["userId"])
            k = (
                int(row["k"])
                if "k" in df.columns and pd.notna(row.get("k"))
                else default_k
            )
            idx = self.user_index.get(uid)
            if idx is None:
                rows.append({"userId": uid, "recommendations": []})
                continue
            scores = self.user_feats[idx] @ self.item_feats.T
            k_eff = min(k, len(scores))
            top_idx = np.argpartition(-scores, k_eff - 1)[:k_eff]
            top_idx = top_idx[np.argsort(-scores[top_idx])]
            recs = [
                {"movieId": int(self.item_ids[i]), "score": float(scores[i])}
                for i in top_idx
            ]
            rows.append({"userId": uid, "recommendations": recs})
        return pd.DataFrame(rows)
