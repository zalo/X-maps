from typing import Any, Callable, Optional

import cv2

from depth_reprojection_pipe import DepthReprojectionPipe
from stats_printer import StatsPrinter

from dataclasses import dataclass, field


@dataclass
class RuntimeParams:
    camera_width: int
    camera_height: int

    projector_width: int
    projector_height: int

    projector_fps: int

    z_near: float
    z_far: float

    calib: str

    projector_time_map: str

    no_frame_dropping: bool

    camera_perspective: bool

    @property
    def should_drop_frames(self):
        return not self.no_frame_dropping


class OpenCVWindow:
    """OpenCV-based window replacement for Metavision MTWindow."""

    def __init__(self, title, width, height):
        self.title = title
        self._should_close = False
        self._keyboard_callback = None
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(title, width, height)

    def should_close(self):
        if self._should_close:
            return True
        try:
            return cv2.getWindowProperty(self.title, cv2.WND_PROP_VISIBLE) < 1
        except cv2.error:
            return True

    def set_close_flag(self):
        self._should_close = True

    def show_async(self, img):
        cv2.imshow(self.title, img)
        key = cv2.waitKey(1)
        if key >= 0 and self._keyboard_callback:
            self._keyboard_callback(key)

    def set_keyboard_callback(self, cb):
        self._keyboard_callback = cb


class FakeWindow:
    def should_close(self):
        return False

    def show_async(self, img):
        pass

    def set_keyboard_callback(self, cb):
        pass


USE_FAKE_WINDOW = False


@dataclass
class DepthReprojectionProcessor:
    params: RuntimeParams

    stats_printer: StatsPrinter = StatsPrinter()

    _pipe: DepthReprojectionPipe = field(init=False)
    _window: Any = field(init=False)

    def should_close(self):
        return self._window.should_close()

    def show_async(self, depth_map):
        self._window.show_async(depth_map)
        self.stats_printer.count("frames shown")

    def __enter__(self):
        self._pipe = DepthReprojectionPipe(
            params=self.params, stats_printer=self.stats_printer, frame_callback=self.show_async
        )

        if USE_FAKE_WINDOW:
            self._window = FakeWindow()
        else:
            self._window = OpenCVWindow(
                title="X Maps Depth",
                width=self.params.camera_width if self.params.camera_perspective else self.params.projector_width,
                height=self.params.camera_height if self.params.camera_perspective else self.params.projector_height,
            )
            print(
                """
Available keyboard shortcuts:
- E:     Switch between frame event filters
- S:     Toggle printing statistics
- Q/Esc: Quit the application"""
            )

        self._window.set_keyboard_callback(self.keyboard_cb)

        return self

    def __exit__(self, *exc_info):
        self.stats_printer.print_stats()
        cv2.destroyAllWindows()
        return False

    def keyboard_cb(self, key):
        """Handle keyboard input from OpenCV waitKey().
        key is the ASCII code of the pressed key.
        """
        if key == 27 or key == ord("q") or key == ord("Q"):  # Escape or Q
            self._window.set_close_flag()
        if key == ord("e") or key == ord("E"):
            self._pipe.select_next_frame_event_filter()
        if key == ord("s") or key == ord("S"):
            self.stats_printer.toggle_silence()

    def process_events(self, evs):
        self.stats_printer.print_stats_if_needed()
        self.stats_printer.count("processed evs", len(evs))
        self._pipe.process_events(evs)
        self.stats_printer.print_stats_if_needed()

    def reset(self):
        self._pipe.reset()
