from src.control.common import safe_pyautogui
from .base_mode import BaseMode, GuideItem


class MediaMode(BaseMode):
    def __init__(self) -> None:
        super().__init__(name="media", display_name="Media", hotkey="3", use_pointer_move=False)

    def get_guide_items(self) -> list[GuideItem]:
        return [
            ("pinch", "Pinch -> Play / Pause"),
            ("tf_up", "Two fingers up -> Volume up"),
            ("tf_down", "Two fingers down -> Volume down"),
            ("tf_back", "Two fingers left -> Rewind"),
            ("tf_next", "Two fingers right -> Forward"),
        ]

    def handle_pinch(self, pyautogui_module):
        return safe_pyautogui("Play / Pause", lambda: pyautogui_module.press("space"))

    def handle_dynamic_gesture(self, gesture: str, pyautogui_module):
        if gesture == "up":
            return safe_pyautogui("Volume up", lambda: pyautogui_module.press("volumeup"))
        if gesture == "down":
            return safe_pyautogui("Volume down", lambda: pyautogui_module.press("volumedown"))
        if gesture == "back":
            return safe_pyautogui("Rewind", lambda: pyautogui_module.press("left"))
        if gesture == "next":
            return safe_pyautogui("Forward", lambda: pyautogui_module.press("right"))
        return False, f"Unknown media gesture: {gesture}"
