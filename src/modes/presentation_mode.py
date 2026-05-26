from __future__ import annotations

from typing import Any

from src.control.common import safe_pyautogui
from .base_mode import BaseMode, GuideItem


class PresentationMode(BaseMode):
    name = "presentation"
    display_name = "Presentation"
    hotkey = "2"
    use_pointer_move = False

    def get_guide_items(self) -> list[GuideItem]:
        return [
            ("pinch", "Pinch -> Laser/click"),
            ("tf_down", "Two fingers open/down -> Start slideshow"),
            ("tf_up", "Two fingers close/up -> Exit slideshow"),
            ("tf_back", "Two fingers left -> Previous slide"),
            ("tf_next", "Two fingers right -> Next slide"),
        ]

    def handle_pinch(self, pyautogui_module: Any) -> tuple[bool, str]:
        return safe_pyautogui("Presentation click", lambda: pyautogui_module.click())

    def handle_dynamic_gesture(
        self,
        gesture: str,
        pyautogui_module: Any,
    ) -> tuple[bool, str]:
        actions = {
            "down": ("Start slideshow", lambda: pyautogui_module.press("f5")),
            "up": ("Exit slideshow", lambda: pyautogui_module.press("esc")),
            "back": ("Previous slide", lambda: pyautogui_module.press("left")),
            "next": ("Next slide", lambda: pyautogui_module.press("right")),
        }
        item = actions.get(gesture)
        if item is None:
            return False, f"Unknown presentation gesture: {gesture}"
        action_name, action = item
        return safe_pyautogui(action_name, action)
