from dataclasses import dataclass
import time


@dataclass
class TimingWatchdog:
    """Compare total CPU processing time to the elapsed event time since first processed event.

    Accepts dv.EventStore objects (uses getLowestTime()) or numpy arrays (uses ["t"][0]).
    """

    # TODO may behave incorrectly when event time wraps around

    stats_printer: "StatsPrinter"

    projector_fps: int

    _first_event_time_us: int = -1

    def _get_first_time(self, evs):
        """Get the first event timestamp from either a dv.EventStore or numpy array."""
        if hasattr(evs, "getLowestTime"):
            return evs.getLowestTime()
        return evs["t"][0]

    def is_processing_behind(self, evs) -> bool:
        first_time = self._get_first_time(evs)

        if self._first_event_time_us == -1:
            self._first_event_time_us = first_time
            # first events are arriving now, so let's start the global timers
            self.stats_printer.reset()
            return False

        total_ev_time_ns = (first_time - self._first_event_time_us) * 1000
        total_processing_t_ns = time.perf_counter_ns() - self.stats_printer.start_time_ns()
        processing_lags_behind_ns = total_processing_t_ns - total_ev_time_ns

        self.stats_printer.add_time_measure_ns("(cpu t - ev[0] t)", processing_lags_behind_ns)

        frames_behind_i = int(processing_lags_behind_ns / (1000 * 1000 * 1000 / self.projector_fps))
        self.stats_printer.add_metric("frames behind", frames_behind_i)

        return frames_behind_i > 0

    def reset(self):
        self._first_event_time_us = -1
