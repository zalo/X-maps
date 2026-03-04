# Event iterator using dv-processing for live iniVation cameras or recorded .aedat4 files

from os import path
import sys
import time
import numpy as np
import dv_processing as dv

# Numpy dtype compatible with the rest of the pipeline (matches original Metavision format)
EVENT_DTYPE = np.dtype([("x", "<u2"), ("y", "<u2"), ("p", "<i2"), ("t", "<i8")])


def eventstore_to_numpy(store):
    """Convert dv.EventStore to numpy structured array with pipeline-compatible field names.

    dv-processing uses field names 'timestamp', 'x', 'y', 'polarity'.
    The rest of the pipeline expects 'x', 'y', 't', 'p'.
    """
    arr = store.numpy()
    result = np.empty(len(arr), dtype=EVENT_DTYPE)
    result["x"] = arr["x"]
    result["y"] = arr["y"]
    result["t"] = arr["timestamp"]
    result["p"] = arr["polarity"].astype(np.int16)
    return result


class DvEventsIterator:
    """Unified event iterator for live iniVation cameras or recorded .aedat4 files.

    Yields dv.EventStore objects. Use eventstore_to_numpy() to convert to numpy.
    """

    def __init__(self, input_filename=None, threshold_on=None, threshold_off=None):
        if not input_filename:
            self.__is_live = True
        elif not (path.exists(input_filename) and path.isfile(input_filename)):
            print("Error: provided input path '{}' does not exist or is not a file.".format(input_filename))
            sys.exit(1)
        else:
            self.__is_live = False

        if self.__is_live:
            self.capture = dv.io.camera.open()
            if self.capture is None:
                print("No live camera found! Exiting...")
                sys.exit(1)
            # Set contrast thresholds if provided
            if threshold_on is not None:
                self.capture.setContrastThresholdOn(threshold_on)
            if threshold_off is not None:
                self.capture.setContrastThresholdOff(threshold_off)
        else:
            self.capture = dv.io.MonoCameraRecording(input_filename)

    def __iter__(self):
        while self.capture.isRunning():
            events = self.capture.getNextEventBatch()
            if events is not None and len(events) > 0:
                yield events
            elif self.__is_live:
                time.sleep(0.001)

    def is_done(self):
        return not self.capture.isRunning()

    def get_size(self):
        """Return (height, width) of the event sensor."""
        resolution = self.capture.getEventResolution()
        # getEventResolution() returns (width, height), we return (height, width)
        return resolution[1], resolution[0]
