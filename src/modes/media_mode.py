from __future__ import annotations

from typing import Any

from src.control.common import safe_pyautogui
from .base_mode import BaseMode, GuideItem

MEDIA_REWIND_10S_KEY = "j"
MEDIA_FORWARD_10S_KEY = "l"


class MediaMode(BaseMode):
    name = "media"
    display_name = "Media"
    hotkey = "3"
    use_pointer_move = False

    def get_guide_items(self) -> list[GuideItem]:
        return [
            ("pinch", "Pinch -> Play/Pause"),
            ("tf_down", "Two fingers open/down -> Volume up"),
            ("tf_up", "Two fingers close/up -> Volume down"),
            ("tf_back", "Two fingers left -> Rewind 10s"),
            ("tf_next", "Two fingers right -> Forward 10s"),
        ]

    def handle_pinch(self, pyautogui_module: Any) -> tuple[bool, str]:
        return safe_pyautogui("Play/Pause", lambda: pyautogui_module.press("space"))

    def handle_dynamic_gesture(
        self,
        gesture: str,
        pyautogui_module: Any,
    ) -> tuple[bool, str]:
        actions = {
            "down": ("Volume up", lambda: pyautogui_module.press("volumeup")),
            "up": ("Volume down", lambda: pyautogui_module.press("volumedown")),
            "back": (
                "Rewind 10s",
                lambda: pyautogui_module.press(MEDIA_REWIND_10S_KEY),
            ),
            "next": (
                "Forward 10s",
                lambda: pyautogui_module.press(MEDIA_FORWARD_10S_KEY),
            ),
        }
        item = actions.get(gesture)
        if item is None:
            return False, f"Unknown media gesture: {gesture}"
        action_name, action = item
        return safe_pyautogui(action_name, action)
