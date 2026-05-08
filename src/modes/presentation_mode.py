from src.control.common import safe_pyautogui
from .base_mode import BaseMode, GuideItem


class PresentationMode(BaseMode):
    def __init__(self) -> None:
        super().__init__(name="presentation", display_name="Presentation", hotkey="2", use_pointer_move=True)

    def get_guide_items(self) -> list[GuideItem]:
        return [
            ("point", "Point -> Move cursor / laser"),
            ("pinch", "Pinch -> Left click"),
            ("tf_up", "Two fingers up -> Start slideshow"),
            ("tf_down", "Two fingers down -> Exit slideshow"),
            ("tf_back", "Two fingers left -> Previous slide"),
            ("tf_next", "Two fingers right -> Next slide"),
        ]

    def handle_pinch(self, pyautogui_module):
        return safe_pyautogui("Left click", lambda: pyautogui_module.click(button="left"))

    def handle_dynamic_gesture(self, gesture: str, pyautogui_module):
        if gesture == "up":
            return safe_pyautogui("Start slideshow", lambda: pyautogui_module.press("f5"))
        if gesture == "down":
            return safe_pyautogui("Exit slideshow", lambda: pyautogui_module.press("esc"))
        if gesture == "back":
            return safe_pyautogui("Previous slide", lambda: pyautogui_module.press("left"))
        if gesture == "next":
            return safe_pyautogui("Next slide", lambda: pyautogui_module.press("right"))
        return False, f"Unknown presentation gesture: {gesture}"
