# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)
from dataclasses import dataclass
from ..file_loading.load_nexus import StreamInfo
from typing import Optional, List
from enum import Enum
import multiprocessing as mp
from ._consumer_type import ConsumerType
from ._consumer import (start_consumers, stop_consumers, create_consumers,
                        all_consumers_stopped)
from ._data_buffer import StreamedDataBuffer
from queue import Empty as QueueEmpty


class InstructionType(Enum):
    STOP_NOW = 1
    UPDATE_STOP_TIME = 2


@dataclass(frozen=True)
class ManagerInstruction:
    type: InstructionType
    stop_time_ms: Optional[int] = None  # milliseconds from unix epoch


def data_consumption_manager(start_time_ms: int, stop_time_ms: Optional[int],
                             run_id: str, topics: List[str], kafka_broker: str,
                             consumer_type: ConsumerType,
                             stream_info: Optional[List[StreamInfo]], interval_s: float,
                             event_buffer_size: int, slow_metadata_buffer_size: int,
                             fast_metadata_buffer_size: int, chopper_buffer_size: int,
                             worker_instruction_queue: mp.Queue, data_queue: mp.Queue,
                             test_message_queue: Optional[mp.Queue]):
    """
    Starts and stops buffers and data consumers which collect data and
    send them back to the main process via a queue.

    All input args must be mp.Queue or pickleable as this function is launched
    as a multiprocessing.Process.
    """
    buffer = StreamedDataBuffer(data_queue, event_buffer_size,
                                slow_metadata_buffer_size, fast_metadata_buffer_size,
                                chopper_buffer_size, interval_s, run_id)

    if stream_info is not None:
        buffer.init_metadata_buffers(stream_info)

    consumers = create_consumers(start_time_ms, stop_time_ms, set(topics), kafka_broker,
                                 consumer_type, buffer.new_data, test_message_queue)

    start_consumers(consumers)
    buffer.start()

    while not all_consumers_stopped(consumers):
        try:
            instruction = worker_instruction_queue.get(timeout=.5)
            if instruction.type == InstructionType.STOP_NOW:
                stop_consumers(consumers)
            elif instruction.type == InstructionType.UPDATE_STOP_TIME:
                for consumer in consumers:
                    consumer.update_stop_time(instruction.stop_time_ms)
        except QueueEmpty:
            pass
        except (ValueError, OSError):
            # Queue has been closed, stop worker
            stop_consumers(consumers)

    buffer.stop()
