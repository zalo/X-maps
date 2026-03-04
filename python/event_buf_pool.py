from typing import List
from dataclasses import dataclass, field
import numpy as np

from bias_events_iterator import EVENT_DTYPE


@dataclass
class EventBufPool:
    """Simple pool for numpy event arrays. No longer requires Metavision SDK."""

    def get_buf(self):
        return np.empty(0, dtype=EVENT_DTYPE)

    def return_buf(self, buf):
        pass
