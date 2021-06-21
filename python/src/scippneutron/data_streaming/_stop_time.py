from dataclasses import dataclass


@dataclass(frozen=True)
class StopTime:
    stop_time_ms: int  # milliseconds from unix epoch
