from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

GuideItem = tuple[str, str]


class BaseMode(ABC):
    """Base interface for one runtime control mode."""

    name: str
    display_name: str
    hotkey: str
    use_pointer_move: bool = False

    @abstractmethod
    def get_guide_items(self) -> list[GuideItem]:
        """Return gesture guide items shown in the right-side preview panel."""

    @abstractmethod
    def handle_pinch(self, pyautogui_module: Any) -> tuple[bool, str]:
        """Handle the static pinch gesture."""

    @abstractmethod
    def handle_dynamic_gesture(
        self,
        gesture: str,
        pyautogui_module: Any,
    ) -> tuple[bool, str]:
        """Handle two_fingers-derived gestures: up, down, back, next."""
