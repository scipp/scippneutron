from dataclasses import dataclass


@dataclass(frozen=True)
class StopTimeUpdate:
    stop_time_ms: int  # milliseconds from unix epoch
