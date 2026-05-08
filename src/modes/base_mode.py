from dataclasses import dataclass

GuideItem = tuple[str, str]


@dataclass
class BaseMode:
    name: str
    display_name: str
    hotkey: str
    use_pointer_move: bool = False

    def get_guide_items(self) -> list[GuideItem]:
        return []

    def handle_pinch(self, pyautogui_module):
        return False, "Pinch not mapped"

    def handle_dynamic_gesture(self, gesture: str, pyautogui_module):
        return False, f"Dynamic gesture not mapped: {gesture}"
