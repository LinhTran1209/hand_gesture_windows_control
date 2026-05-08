import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np


def load_model_predictor(
    model_path: Path,
    meta_path: Path,
    name: str,
) -> tuple[Any, dict[str, Any]]:
    if not model_path.exists():
        raise FileNotFoundError(f"Missing {name} model: {model_path}")

    payload = joblib.load(model_path)
    meta: dict[str, Any] = {}

    if isinstance(payload, dict):
        meta.update(
            {
                "feature_columns": payload.get("feature_columns"),
                "model_name": payload.get("model_name"),
            }
        )

        label_encoder = payload.get("label_encoder")
        if label_encoder is not None and hasattr(label_encoder, "classes_"):
            meta["labels"] = list(label_encoder.classes_)
            meta["label_encoder"] = label_encoder

        model = None
        for key in ("pipeline", "model", "best_model", "estimator", "classifier", "clf"):
            candidate = payload.get(key)
            if candidate is not None and hasattr(candidate, "predict"):
                model = candidate
                print(f"[INFO] {name} predictor from payload['{key}']")
                break

        if model is None and hasattr(payload, "predict"):
            model = payload

        if model is None:
            raise RuntimeError(
                f"No predictor with .predict() in {name} payload. Keys: {list(payload.keys())}"
            )
    else:
        model = payload

    if not hasattr(model, "predict"):
        raise RuntimeError(f"{name} model has no predict(): {type(model).__name__}")

    if meta_path.exists():
        try:
            meta.update(json.loads(meta_path.read_text(encoding="utf-8")))
        except Exception as exc:
            print(f"[WARN] Cannot read {name} metadata: {type(exc).__name__}")

    return model, meta


def patch_simple_imputer_compat(
    estimator: Any,
    sample_feature: np.ndarray | None = None,
) -> None:
    seen: set[int] = set()

    def walk(obj: Any) -> list[Any]:
        if obj is None or id(obj) in seen:
            return []
        seen.add(id(obj))
        found = [obj]

        steps = getattr(obj, "steps", None)
        if isinstance(steps, list):
            for _, step in steps:
                found.extend(walk(step))

        transformers = getattr(obj, "transformers", None)
        if isinstance(transformers, list):
            for item in transformers:
                if isinstance(item, tuple) and len(item) >= 2:
                    found.extend(walk(item[1]))

        transformer_list = getattr(obj, "transformer_list", None)
        if isinstance(transformer_list, list):
            for _, step in transformer_list:
                found.extend(walk(step))

        return found

    for obj in walk(estimator):
        if obj.__class__.__name__ != "SimpleImputer" or hasattr(obj, "_fill_dtype"):
            continue

        fill_dtype = None
        statistics = getattr(obj, "statistics_", None)
        if statistics is not None:
            try:
                fill_dtype = np.asarray(statistics).dtype
            except Exception:
                fill_dtype = None

        if fill_dtype is None and sample_feature is not None:
            try:
                fill_dtype = np.asarray(sample_feature).dtype
            except Exception:
                fill_dtype = np.float64

        obj._fill_dtype = fill_dtype or np.float64


def predict_one_label(model: Any, feature: np.ndarray, meta: dict[str, Any]) -> str:
    patch_simple_imputer_compat(model, feature)
    pred = model.predict(feature)[0]
    label_encoder = meta.get("label_encoder")

    if label_encoder is not None and not isinstance(pred, str):
        try:
            pred = label_encoder.inverse_transform([pred])[0]
        except Exception:
            pass

    return str(pred)
