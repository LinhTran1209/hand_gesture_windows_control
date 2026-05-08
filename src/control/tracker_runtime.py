import inspect
from typing import Any

import numpy as np

from src.perception.hand_tracker import HandTracker
from .config import HAND_MODEL_PATH


def create_tracker() -> Any:
    init_sig = inspect.signature(HandTracker.__init__)
    supported = init_sig.parameters

    candidate_kwargs = {
        "model_path": HAND_MODEL_PATH,
        "max_num_hands": 2,
        "min_detection_confidence": 0.65,
        "min_presence_confidence": 0.65,
        "min_tracking_confidence": 0.65,
    }

    kwargs = {k: v for k, v in candidate_kwargs.items() if k in supported}
    return HandTracker(**kwargs)


def unpack_process_result(
    result: Any,
    original_frame: np.ndarray,
) -> tuple[np.ndarray, list[Any]]:
    if result is None:
        return original_frame.copy(), []

    if isinstance(result, tuple):
        if len(result) >= 2:
            annotated_frame = result[0] if result[0] is not None else original_frame.copy()
            maybe = result[-1]
            if maybe is None:
                return annotated_frame, []
            if isinstance(maybe, list):
                return annotated_frame, maybe
            return annotated_frame, [maybe]
        if len(result) == 1:
            return original_frame.copy(), unpack_process_result(result[0], original_frame)[1]

    if isinstance(result, list):
        return original_frame.copy(), result

    return original_frame.copy(), [result]


def run_tracker(
    tracker: Any,
    frame: np.ndarray,
    timestamp_ms: int,
) -> tuple[np.ndarray, list[Any]]:
    process_sig = inspect.signature(tracker.process)
    try:
        params = process_sig.parameters
        if "timestamp_ms" in params:
            result = tracker.process(frame, timestamp_ms=timestamp_ms)
        elif len(params) >= 2:
            result = tracker.process(frame, timestamp_ms)
        else:
            result = tracker.process(frame)
    except TypeError:
        try:
            result = tracker.process(frame, timestamp_ms)
        except TypeError:
            result = tracker.process(frame)

    return unpack_process_result(result, frame)
