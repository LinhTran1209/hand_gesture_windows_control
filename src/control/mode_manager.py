from dataclasses import dataclass

from src.modes.base_mode import BaseMode
from src.modes.media_mode import MediaMode
from src.modes.mouse_mode import MouseMode
from src.modes.presentation_mode import PresentationMode


@dataclass
class ModeInfo:
    key: str
    display_name: str


class ModeManager:
    def __init__(self, modes: list[BaseMode], default_mode_name: str = "mouse") -> None:
        self._modes = {mode.name: mode for mode in modes}
        if default_mode_name not in self._modes:
            raise ValueError(f"Unknown default mode: {default_mode_name}")
        self._current_mode_name = default_mode_name

    @property
    def current_mode(self) -> BaseMode:
        return self._modes[self._current_mode_name]

    def switch_by_key(self, key_char: str) -> BaseMode | None:
        for mode in self._modes.values():
            if mode.hotkey == key_char:
                self._current_mode_name = mode.name
                return mode
        return None

    def get_mode_infos(self) -> list[ModeInfo]:
        return [ModeInfo(key=mode.hotkey, display_name=mode.display_name) for mode in self._modes.values()]


def build_default_mode_manager() -> ModeManager:
    return ModeManager(
        modes=[MouseMode(), PresentationMode(), MediaMode()],
        default_mode_name="mouse",
    )
