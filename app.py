from __future__ import annotations

import sys
import time
import traceback
from collections import deque
from pathlib import Path

import cv2 as cv
import pyautogui

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.0

from src.capture.webcam import WebcamCapture, WebcamConfig
from src.features.hand_landmark_features import landmarks_to_feature_vector

from src.control.common import build_label_text, majority_vote, update_action_history
from src.control.config import (
    ACTION_HISTORY_SIZE,
    CAMERA_FPS,
    CAMERA_HEIGHT,
    CAMERA_WIDTH,
    DRAW_LANDMARKS_ON_PREVIEW,
    EPS,
    NO_HAND_RESET_FRAMES,
    POSE_BUFFER_SIZE,
    PREVIEW_WINDOW_NAME,
    RENDER_EVERY_N_FRAMES,
    SECONDARY_SMOOTHING_MIN_VOTES,
    SECONDARY_SMOOTHING_WINDOW,
    SMOOTHING_MIN_VOTES,
    SMOOTHING_WINDOW,
    STATIC_MODEL_META_PATH,
    STATIC_MODEL_PATH,
)
from src.control.dynamic_gestures import (
    detect_two_finger_gesture,
    mark_dynamic_action_executed,
    reset_dynamic_runtime,
)
from src.control.hand_utils import (
    choose_primary_and_secondary,
    get_handedness,
    get_landmarks,
)
from src.control.mode_manager import build_default_mode_manager
from src.control.model_loader import load_model_predictor, predict_one_label
from src.control.mouse_actions import (
    maybe_handle_secondary_pinch_drag,
    maybe_move_mouse,
    maybe_toggle_active,
    maybe_trigger_pinch_action,
    release_mouse_left_if_down,
)
from src.control.overlay import build_preview_canvas, draw_landmarks, draw_mouse_preview
from src.control.preview_window import (
    change_preview_scale,
    configure_fixed_preview_window,
    get_preview_size,
    is_preview_renderable,
    make_preview_frame,
    zoom_step,
)
from src.control.state import ControlState
from src.control.tracker_runtime import create_tracker, run_tracker


def reset_runtime_state(
    state: ControlState,
    static_pred_history: deque[str],
    secondary_pred_history: deque[str],
    dynamic_history: deque[str],
    pose_buffer: deque,
    action_history: deque[str],
) -> None:
    static_pred_history.clear()
    secondary_pred_history.clear()
    reset_dynamic_runtime(dynamic_history, pose_buffer)
    action_history.clear()

    state.active = False
    state.last_click_ts = 0.0
    state.primary_pinch_touching = False
    state.secondary_pinch_touching = False
    state.last_dynamic_ts = 0.0
    state.last_navigation_ts = 0.0
    state.last_mode_ts = 0.0
    state.last_action = "Reset state"
    release_mouse_left_if_down(state, pyautogui, "Release left mouse")
    state.mouse_x = None
    state.mouse_y = None
    state.last_mouse_ts = 0.0
    state.prev_stable_static_pred = "None"
    state.prev_stable_dynamic_pred = "None"
    state.dynamic_collecting = False
    state.primary_center = None


def main() -> None:
    static_model, static_meta = load_model_predictor(
        STATIC_MODEL_PATH, STATIC_MODEL_META_PATH, name="static"
    )
    mode_manager = build_default_mode_manager()

    webcam = WebcamCapture(
        WebcamConfig(
            camera_index=0,
            width=CAMERA_WIDTH,
            height=CAMERA_HEIGHT,
            fps=CAMERA_FPS,
            mirror=True,
        )
    )

    tracker = create_tracker()
    state = ControlState()

    static_pred_history: deque[str] = deque(maxlen=SMOOTHING_WINDOW)
    secondary_pred_history: deque[str] = deque(maxlen=SECONDARY_SMOOTHING_WINDOW)
    dynamic_history: deque[str] = deque(maxlen=3)
    pose_buffer = deque(maxlen=POSE_BUFFER_SIZE)
    action_history: deque[str] = deque(maxlen=ACTION_HISTORY_SIZE)

    last_logged_action = "None"
    stable_static_pred = "None"
    stable_dynamic_pred = "None"
    secondary_static_pred = "None"

    no_hand_frames = 0
    show_panel = True
    frame_index = 0
    prev_time = time.perf_counter()
    start_time = time.perf_counter()

    try:
        webcam.open()
        cv.namedWindow(PREVIEW_WINDOW_NAME, cv.WINDOW_NORMAL)
        preview_w, preview_h = get_preview_size(state)
        cv.resizeWindow(PREVIEW_WINDOW_NAME, preview_w, preview_h)
        configure_fixed_preview_window(PREVIEW_WINDOW_NAME, state, force=True)

        print("[INFO] 1 = Mouse | 2 = Presentation | 3 = Media")
        print(
            "[INFO] q=quit | r=reset | e=toggle active | p=show/hide panel | +/-=zoom"
        )

        while True:
            ok, frame = webcam.read()
            if not ok or frame is None:
                continue

            frame_index += 1
            configure_fixed_preview_window(PREVIEW_WINDOW_NAME, state)
            preview_renderable = is_preview_renderable(PREVIEW_WINDOW_NAME)

            timestamp_ms = int((time.perf_counter() - start_time) * 1000)

            try:
                annotated_frame, detections = run_tracker(tracker, frame, timestamp_ms)
            except Exception as exc:
                annotated_frame = frame.copy()
                detections = []
                if preview_renderable:
                    cv.putText(
                        annotated_frame,
                        f"Tracker error: {type(exc).__name__}",
                        (20, 35),
                        cv.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255),
                        2,
                        cv.LINE_AA,
                    )

            handedness = "Unknown"
            landmarks = None
            secondary_landmarks = None
            secondary_static_pred = "None"
            stable_dynamic_pred = "None"

            if detections and len(detections) > 0:
                no_hand_frames = 0
                primary_det, secondary_det = choose_primary_and_secondary(
                    detections, state
                )
                if primary_det is None:
                    continue

                landmarks = get_landmarks(primary_det)
                handedness = get_handedness(primary_det)

                if secondary_det is not None:
                    secondary_landmarks = get_landmarks(secondary_det)
                else:
                    secondary_pred_history.clear()
                    state.secondary_pinch_touching = False

                if preview_renderable and DRAW_LANDMARKS_ON_PREVIEW:
                    for det_to_draw in detections[:2]:
                        lm_to_draw = get_landmarks(det_to_draw)
                        if lm_to_draw is not None and len(lm_to_draw) == 21:
                            annotated_frame = draw_landmarks(
                                annotated_frame, lm_to_draw
                            )

                current_mode = mode_manager.current_mode

                try:
                    static_feature = landmarks_to_feature_vector(landmarks).reshape(
                        1, -1
                    )
                    raw_static_pred = predict_one_label(
                        static_model, static_feature, static_meta
                    )
                    static_pred_history.append(raw_static_pred)

                    voted_label, _vote_count = majority_vote(
                        static_pred_history, min_votes=SMOOTHING_MIN_VOTES
                    )
                    if voted_label is not None:
                        stable_static_pred = voted_label

                    maybe_toggle_active(state, stable_static_pred)
                    if not state.active:
                        release_mouse_left_if_down(
                            state, pyautogui, "Release left mouse"
                        )

                    if current_mode.use_pointer_move:
                        maybe_move_mouse(
                            state, stable_static_pred, landmarks, pyautogui
                        )

                    maybe_trigger_pinch_action(
                        state,
                        stable_static_pred,
                        landmarks,
                        source="primary",
                        action_callback=lambda: current_mode.handle_pinch(pyautogui),
                    )

                except Exception as exc:
                    stable_static_pred = f"ERR: {type(exc).__name__}"

                if secondary_landmarks is not None and len(secondary_landmarks) == 21:
                    try:
                        secondary_feature = landmarks_to_feature_vector(
                            secondary_landmarks
                        ).reshape(1, -1)
                        secondary_raw_pred = predict_one_label(
                            static_model, secondary_feature, static_meta
                        )
                        secondary_pred_history.append(secondary_raw_pred)

                        secondary_voted, _secondary_vote_count = majority_vote(
                            secondary_pred_history,
                            min_votes=SECONDARY_SMOOTHING_MIN_VOTES,
                        )
                        if secondary_voted is not None:
                            secondary_static_pred = secondary_voted

                        drag_handled = maybe_handle_secondary_pinch_drag(
                            state,
                            stable_static_pred,
                            secondary_static_pred,
                            secondary_landmarks,
                            current_mode.use_pointer_move,
                            pyautogui,
                        )

                        if not drag_handled:
                            maybe_trigger_pinch_action(
                                state,
                                secondary_static_pred,
                                secondary_landmarks,
                                source="secondary",
                                action_callback=lambda: current_mode.handle_pinch(
                                    pyautogui
                                ),
                            )
                    except Exception as exc:
                        secondary_static_pred = f"ERR: {type(exc).__name__}"
                else:
                    state.secondary_pinch_touching = False
                    release_mouse_left_if_down(state, pyautogui, "Release left mouse")

                try:
                    status_text, detected_gesture = detect_two_finger_gesture(
                        state,
                        stable_static_pred,
                        handedness,
                        landmarks,
                        pose_buffer,
                    )

                    if detected_gesture is not None:
                        ok_action, action_msg = current_mode.handle_dynamic_gesture(
                            detected_gesture, pyautogui
                        )
                        state.last_action = action_msg
                        if ok_action:
                            stable_dynamic_pred = detected_gesture
                            dynamic_history.append(detected_gesture)
                            mark_dynamic_action_executed(
                                state, detected_gesture, pose_buffer
                            )
                    else:
                        stable_dynamic_pred = "None"

                except Exception:
                    stable_dynamic_pred = "None"
                    traceback.print_exc()

                if preview_renderable:
                    draw_mouse_preview(annotated_frame, landmarks)

            else:
                stable_static_pred = "None"
                stable_dynamic_pred = "None"
                secondary_static_pred = "None"

                no_hand_frames += 1
                static_pred_history.clear()
                secondary_pred_history.clear()
                reset_dynamic_runtime(dynamic_history, pose_buffer)

                state.dynamic_collecting = False
                state.primary_pinch_touching = False
                state.secondary_pinch_touching = False
                state.primary_center = None
                release_mouse_left_if_down(state, pyautogui, "Release left mouse")

                if no_hand_frames >= NO_HAND_RESET_FRAMES and state.active:
                    state.active = False
                    state.last_action = "Auto LOCK"

            last_logged_action = update_action_history(
                action_history, last_logged_action, state.last_action
            )

            current_time = time.perf_counter()
            fps = 1.0 / max(current_time - prev_time, EPS)
            prev_time = current_time

            should_render = frame_index % max(1, RENDER_EVERY_N_FRAMES) == 0
            if should_render and preview_renderable:
                current_mode = mode_manager.current_mode
                label_text = build_label_text(
                    stable_static_pred, stable_dynamic_pred, state.active
                )
                if secondary_static_pred != "None":
                    label_text += f" | second={secondary_static_pred}"

                composed = build_preview_canvas(
                    annotated_frame,
                    fps,
                    current_mode.display_name,
                    label_text,
                    state,
                    action_history,
                    show_panel,
                    ["1 Mouse", "2 Presentation", "3 Media"],
                    current_mode.get_guide_items(),
                )
                cv.imshow(PREVIEW_WINDOW_NAME, make_preview_frame(composed, state))

            state.prev_stable_static_pred = stable_static_pred
            state.prev_stable_dynamic_pred = stable_dynamic_pred

            key = cv.waitKey(1) & 0xFF
            if key == ord("q"):
                release_mouse_left_if_down(state, pyautogui, "Release left mouse")
                break
            if key == ord("r"):
                reset_runtime_state(
                    state,
                    static_pred_history,
                    secondary_pred_history,
                    dynamic_history,
                    pose_buffer,
                    action_history,
                )
                last_logged_action = "None"
                stable_static_pred = "None"
                stable_dynamic_pred = "None"
                secondary_static_pred = "None"
                configure_fixed_preview_window(PREVIEW_WINDOW_NAME, state, force=True)

            elif key == ord("e"):
                state.active = not state.active
                if not state.active:
                    state.dynamic_collecting = False
                    reset_dynamic_runtime(dynamic_history, pose_buffer)
                    state.primary_pinch_touching = False
                    state.secondary_pinch_touching = False
                    state.primary_center = None
                    release_mouse_left_if_down(state, pyautogui, "Release left mouse")
                state.last_action = f"Manual toggle -> {state.active}"

            elif key == ord("p"):
                show_panel = not show_panel

            elif key in (ord("1"), ord("2"), ord("3")):
                mode = mode_manager.switch_by_key(chr(key))
                if mode is not None:
                    state.last_action = f"Switch mode -> {mode.display_name}"
                    state.primary_pinch_touching = False
                    state.secondary_pinch_touching = False
                    release_mouse_left_if_down(state, pyautogui, "Release left mouse")
                    reset_dynamic_runtime(dynamic_history, pose_buffer)

            elif key in (ord("+"), ord("=")):
                change_preview_scale(state, zoom_step())
                configure_fixed_preview_window(PREVIEW_WINDOW_NAME, state, force=True)

            elif key in (ord("-"), ord("_")):
                change_preview_scale(state, -zoom_step())
                configure_fixed_preview_window(PREVIEW_WINDOW_NAME, state, force=True)

    finally:
        release_mouse_left_if_down(state, pyautogui, "Release left mouse")
        if hasattr(tracker, "close"):
            try:
                tracker.close()
            except Exception:
                pass
        webcam.release()
        cv.destroyAllWindows()


if __name__ == "__main__":
    main()
