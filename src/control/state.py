from dataclasses import dataclass
from .config import PREVIEW_SCALE_DEFAULT


@dataclass
class ControlState:
    active: bool = False
    last_click_ts: float = 0.0
    primary_pinch_touching: bool = False
    secondary_pinch_touching: bool = False
    last_dynamic_ts: float = 0.0
    last_navigation_ts: float = 0.0
    last_mode_ts: float = 0.0
    last_action: str = "None"
    mouse_x: float | None = None
    mouse_y: float | None = None
    last_mouse_ts: float = 0.0
    mouse_left_down: bool = False
    drag_source: str | None = None
    prev_stable_static_pred: str = "None"
    prev_stable_dynamic_pred: str = "None"
    dynamic_collecting: bool = False
    primary_center: tuple[float, float] | None = None
    preview_scale: float = PREVIEW_SCALE_DEFAULT
    last_preview_window_fix_ts: float = 0.0
    last_log_ts: float = 0.0
