# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
from datetime import datetime


def _unix_ms_to_datetime(unix_ms: int) -> datetime:
    return datetime.fromtimestamp(unix_ms / 1000.0, tz=None)  # noqa: DTZ006


class DataStreamWidget:
    def __init__(
        self,
        start_time_ms: int | None = None,
        stop_time_ms: int | None = None,
        run_title: str | None = None,
    ):
        import ipywidgets as widgets
        from IPython.display import display

        self._stop_button = widgets.ToggleButton(description="Stop stream")
        self._stop_button.observe(self._on_button_clicked, "value")
        self._title = widgets.Label("-")
        self._start_time = widgets.Label("-")
        self._stop_time = widgets.Label("-")
        self._time_format = "%m/%d/%Y %H:%M:%S"

        if run_title is not None:
            self.set_title(run_title)
        if start_time_ms is not None:
            self.set_start_time(start_time_ms)
        if stop_time_ms is not None and stop_time_ms != 0:
            # 0 is the default in the run start message flatbuffer
            # if stop time field is not populated
            self.set_stop_time(stop_time_ms)

        display(
            widgets.HBox(
                [
                    self._stop_button,
                    widgets.HBox([widgets.Label('Run title:'), self._title]),
                    widgets.HBox([widgets.Label('Start time:'), self._start_time]),
                    widgets.HBox([widgets.Label('Stop time:'), self._stop_time]),
                ]
            )
        )

    @staticmethod
    def _on_button_clicked(b):
        b["owner"].description = "Stopping..."
        b["owner"].disabled = True

    def set_start_time(self, start_time_ms: int):
        self._start_time.value = _unix_ms_to_datetime(start_time_ms).strftime(
            self._time_format
        )

    def set_stop_time(self, stop_time_ms: int):
        self._stop_time.value = _unix_ms_to_datetime(stop_time_ms).strftime(
            self._time_format
        )

    def set_title(self, new_run_title: str):
        self._title.value = new_run_title

    @property
    def stop_requested(self) -> bool:
        return self._stop_button.value

    def set_stopped(self):
        self._stop_button.description = "Stopped"
        self._stop_button.disabled = True
