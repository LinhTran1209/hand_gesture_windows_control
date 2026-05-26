from __future__ import annotations

from typing import Any

from src.control.common import safe_pyautogui
from .base_mode import BaseMode, GuideItem
from src.control.config import SCROLL_AMOUNT


class MouseMode(BaseMode):
    name = "mouse"
    display_name = "Mouse"
    hotkey = "1"
    use_pointer_move = True

    def get_guide_items(self) -> list[GuideItem]:
        return [
            ("point", "Point -> Move cursor"),
            ("pinch", "Pinch -> Left click"),
            ("tf_up", "Two fingers close/up -> Scroll down"),
            ("tf_down", "Two fingers open/down -> Scroll up"),
            ("tf_back", "Two fingers left -> Browser back"),
            ("tf_next", "Two fingers right -> Browser forward"),
        ]

    def handle_pinch(self, pyautogui_module: Any) -> tuple[bool, str]:
        return safe_pyautogui("Left click", lambda: pyautogui_module.click())

    def handle_dynamic_gesture(
        self,
        gesture: str,
        pyautogui_module: Any,
    ) -> tuple[bool, str]:
        actions = {
            "up": ("Scroll down", lambda: pyautogui_module.scroll(-SCROLL_AMOUNT)),
            "down": ("Scroll up", lambda: pyautogui_module.scroll(SCROLL_AMOUNT)),
            "back": ("Browser back", lambda: pyautogui_module.hotkey("alt", "left")),
            "next": (
                "Browser forward",
                lambda: pyautogui_module.hotkey("alt", "right"),
            ),
        }
        item = actions.get(gesture)
        if item is None:
            return False, f"Unknown mouse gesture: {gesture}"
        action_name, action = item
        return safe_pyautogui(action_name, action)
