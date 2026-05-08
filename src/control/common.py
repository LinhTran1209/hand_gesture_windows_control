import time
from collections import Counter, deque
from typing import Callable

import pyautogui

from .config import LOG_COOLDOWN_SEC, VERBOSE_DYNAMIC_LOG
from .state import ControlState


def majority_vote(history: deque[str], min_votes: int) -> tuple[str | None, int]:
    if not history:
        return None, 0
    label, count = Counter(history).most_common(1)[0]
    if count >= min_votes:
        return label, count
    return None, count


def safe_pyautogui(action_name: str, fn: Callable[[], None]) -> tuple[bool, str]:
    try:
        fn()
        return True, action_name
    except pyautogui.FailSafeException:
        return False, f"{action_name} blocked by FAILSAFE"
    except Exception as exc:
        return False, f"{action_name} error: {type(exc).__name__}"


def log_dynamic(state: ControlState, message: str, force: bool = False) -> None:
    if not VERBOSE_DYNAMIC_LOG:
        return
    now = time.perf_counter()
    if force or (now - state.last_log_ts >= LOG_COOLDOWN_SEC):
        print(f"[DYNAMIC] {message}")
        state.last_log_ts = now


def update_action_history(
    action_history: deque[str],
    last_logged_action: str,
    current_action: str,
) -> str:
    if current_action and current_action != "None" and current_action != last_logged_action:
        action_history.appendleft(current_action)
        return current_action
    return last_logged_action


def build_label_text(stable_static_pred: str, stable_dynamic_pred: str, active: bool) -> str:
    if stable_dynamic_pred != "None":
        return f"{stable_static_pred} -> {stable_dynamic_pred} | active={active}"
    return f"{stable_static_pred} | active={active}"
