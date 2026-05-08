from src.control.common import safe_pyautogui
from src.control.config import SCROLL_AMOUNT
from .base_mode import BaseMode, GuideItem


class MouseMode(BaseMode):
    def __init__(self) -> None:
        super().__init__(name="mouse", display_name="Mouse", hotkey="1", use_pointer_move=True)

    def get_guide_items(self) -> list[GuideItem]:
        return [
            ("point", "Point -> Move mouse"),
            ("pinch", "Pinch -> Left click"),
            ("tf_up", "Two fingers close vertical -> Scroll up"),
            ("tf_down", "Two fingers open vertical -> Scroll down"),
            ("tf_back", "Two fingers horizontal -> Back"),
            ("tf_next", "Two fingers horizontal -> Next"),
        ]

    def handle_pinch(self, pyautogui_module):
        return safe_pyautogui("Left click", lambda: pyautogui_module.click(button="left"))

    def handle_dynamic_gesture(self, gesture: str, pyautogui_module):
        if gesture == "up":
            return safe_pyautogui(f"Up scroll (+{SCROLL_AMOUNT})", lambda: pyautogui_module.scroll(SCROLL_AMOUNT))
        if gesture == "down":
            return safe_pyautogui(f"Down scroll (-{SCROLL_AMOUNT})", lambda: pyautogui_module.scroll(-SCROLL_AMOUNT))
        if gesture == "back":
            return safe_pyautogui("Back", lambda: pyautogui_module.hotkey("alt", "left"))
        if gesture == "next":
            return safe_pyautogui("Next", lambda: pyautogui_module.hotkey("alt", "right"))
        return False, f"Unknown mouse gesture: {gesture}"
