# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)
import numpy as np
import time
from typing import List, Generator, Optional
import asyncio
import scipp as sc
from ..file_loading.load_nexus import _load_nexus_json
import multiprocessing as mp
from queue import Empty as QueueEmpty
from enum import Enum
from ._consumer_type import ConsumerType
from ._serialisation import convert_from_pickleable_dict
from ._stop_time import StopTimeUpdate
from warnings import warn
from ._data_stream_widget import DataStreamWidget
"""
Some type names are included as strings as imports are done in
function scope to avoid optional dependencies being imported
by the top level __init__.py
"""


class StartTime(Enum):
    NOW = "now"
    START_OF_RUN = "start_of_run"


class StopTime(Enum):
    NEVER = "never"
    END_OF_RUN = "end_of_run"


_missing_dependency_message = (
    "Confluent Kafka Python library and/or serialisation library"
    "not found, please install confluent-kafka and "
    "ess-streaming-data-types as detailed in the "
    "installation instructions (https://scipp.github.io/"
    "scippneutron/getting-started/installation.html)")


async def data_stream(
    kafka_broker: str,
    topics: Optional[List[str]] = None,
    event_buffer_size: int = 1_048_576,
    slow_metadata_buffer_size: int = 1000,
    fast_metadata_buffer_size: int = 100_000,
    chopper_buffer_size: int = 10_000,
    interval: sc.Variable = 2. * sc.units.s,
    run_info_topic: Optional[str] = None,
    start_time: StartTime = StartTime.NOW,
    stop_time: StopTime = StopTime.NEVER,
) -> Generator[sc.DataArray, None, None]:
    """
    Periodically yields accumulated data from stream.
    If the buffer fills up more frequently than the set interval
    then data is yielded more frequently.
    1048576 event buffer is around 24 MB (with pulse_time, id, weights, etc)
    :param kafka_broker: Address of the Kafka broker to stream data from
    :param topics: Kafka topics to consume data from (not required if
      run_info_topic is used)
    :param event_buffer_size: Size of buffer to accumulate event data in
    :param slow_metadata_buffer_size: Size of buffer to accumulate slow
      sample env metadata in
    :param fast_metadata_buffer_size: Size of buffer to accumulate fast
      sample env metadata in
    :param chopper_buffer_size: Size of buffer to accumulate chopper
      timestamps in
    :param interval: interval between yielding any new data
      collected from stream
    :param run_info_topic: If provided, the first data batch returned by
      data_stream will be from the last available run start message in
      the topic, data will be consumed from all data sources documented
      in the run start message
    :param start_time: Get data from now or from start of the last run
    :param stop_time: Stop data_stream at end of run, or not
    """
    """
    Additional info:
    - The `topics` argument is generally unused.
    - Instead, prefer the use of `run_info_topic` where it is possible to go
      and find the last run_start message and use that to get info on what
      topics to listen to.
    - Buffer sizes: it is currently not easy to resize scipp data structures,
      so we choose sensible default buffer sizes. This may need to be tweaked
      in the future.
    - `start_time`: it is possible to go back to the start of the run, even if
      `data_stream()` is started after or during the run. It simply finds the
      start time in the last run_start message. The data can persist on Kafka
      for a significant duration (hours? days?), making this lookup possible.
    """

    validate_buffer_size_args(chopper_buffer_size, event_buffer_size,
                              fast_metadata_buffer_size, slow_metadata_buffer_size)

    ctx = mp.get_context("spawn")
    data_queue = ctx.Queue()
    instruction_queue = ctx.Queue()

    # Use "async for" as "yield from" cannot be used in an async function, see
    # https://www.python.org/dev/peps/pep-0525/#asynchronous-yield-from
    async for data_chunk in _data_stream(data_queue, instruction_queue, kafka_broker,
                                         topics, interval, event_buffer_size,
                                         slow_metadata_buffer_size,
                                         fast_metadata_buffer_size, chopper_buffer_size,
                                         run_info_topic, start_time,
                                         stop_time):  # noqa: E125
        yield data_chunk


def validate_buffer_size_args(chopper_buffer_size, event_buffer_size,
                              fast_metadata_buffer_size, slow_metadata_buffer_size):
    for buffer_name, buffer_size in (("event_buffer_size",
                                      event_buffer_size), ("slow_metadata_buffer_size",
                                                           slow_metadata_buffer_size),
                                     ("fast_metadata_buffer_size",
                                      fast_metadata_buffer_size),
                                     ("chopper_buffer_size", chopper_buffer_size)):
        if buffer_size < 1:
            raise ValueError(f"{buffer_name} must be greater than zero")


def _cleanup_queue(queue: Optional[mp.Queue]):
    if queue is not None:
        queue.cancel_join_thread()
        queue.close()
        queue.join_thread()


async def _data_stream(
        data_queue: mp.Queue,
        worker_instruction_queue: mp.Queue,
        kafka_broker: str,
        topics: Optional[List[str]],
        interval: sc.Variable,
        event_buffer_size: int,
        slow_metadata_buffer_size: int,
        fast_metadata_buffer_size: int,
        chopper_buffer_size: int,
        run_info_topic: Optional[str] = None,
        start_at: StartTime = StartTime.NOW,
        end_at: StopTime = StopTime.NEVER,
        query_consumer: Optional["KafkaQueryConsumer"] = None,  # noqa: F821
        consumer_type: ConsumerType = ConsumerType.REAL,
        halt_after_n_data_chunks: int = np.iinfo(np.int32).max,  # noqa: B008
        halt_after_n_warnings: int = np.iinfo(np.int32).max,  # noqa: B008
        test_message_queue: Optional[mp.Queue] = None,  # for tests
        timeout: Optional[sc.Variable] = None,  # for tests
) -> Generator[sc.DataArray, None, None]:
    """
    Main implementation of data stream is extracted to this function so that
    fake consumers can be injected for unit tests
    """

    # Search backwards to find the last run_start message
    try:
        from ._consumer import (get_run_start_message, KafkaQueryConsumer)
        from ._data_consumption_manager import (data_consumption_manager,
                                                ManagerInstruction, InstructionType)
    except ImportError:
        raise ImportError(_missing_dependency_message)

    if topics is None and run_info_topic is None:
        raise ValueError("At least one of 'topics' and 'run_info_topic'"
                         " must be specified")

    # This is defaulted to None in the function signature
    # to avoid it having to be imported earlier
    if query_consumer is None:
        query_consumer = KafkaQueryConsumer(kafka_broker)

    # stream_info contains information on where to look for data and metadata.
    # The data from the start message is yielded as the first chunk of data.
    #
    # TODO: This should, in principle, not look any different from any other
    # chunk of data, right now it seems it may be different?
    # (see https://github.com/scipp/scippneutron/issues/114)
    #
    # Generic data chunk structure: geometry, metadata, and event data are all
    # optional.
    # - first data chunk will most probably contain no event data
    # - subsequent chunks can contain geometry info, if e.g. some pixels have
    #   moved
    # - metadata (e.g. sample environment) might be empty, if values have not
    #   changed
    stream_info = None
    run_id = ""
    run_title = "-"  # for display in widget
    stop_time_ms = None
    n_data_chunks = 0
    if run_info_topic is not None:
        run_start_info = get_run_start_message(run_info_topic, query_consumer)
        run_id = run_start_info.job_id
        run_title = run_start_info.run_name
        # default value for stop_time in message flatbuffer is 0,
        # it means that field has not been populated
        if end_at == StopTime.END_OF_RUN and run_start_info.stop_time != 0:
            stop_time_ms = run_start_info.stop_time
        if topics is None:
            loaded_data, stream_info = _load_nexus_json(run_start_info.nexus_structure,
                                                        get_start_info=True)
            topics = [stream.topic for stream in stream_info]
        else:
            loaded_data, _ = _load_nexus_json(run_start_info.nexus_structure,
                                              get_start_info=False)
        topics.append(run_info_topic)  # listen for stop run message
        yield loaded_data
        n_data_chunks += 1

    if start_at == StartTime.START_OF_RUN:
        start_time = run_start_info.start_time * sc.Unit("milliseconds")
    else:
        start_time = time.time() * sc.units.s

    # Convert to int and float as easier to pass to mp.Process
    # (sc.Variable would have to be serialised/deserialised)
    start_time_ms = int(sc.to_unit(start_time, "milliseconds").value)
    interval_s = float(sc.to_unit(interval, 's').value)

    # Specify to start the process using the "spawn" method, otherwise
    # on Linux the default is to fork the Python interpreter which
    # is "problematic" in a multithreaded process, this can apparently
    # even cause multiprocessing's own Queue to cause problems when forking.
    # See documentation:
    # https://docs.python.org/3/library/multiprocessing.html#contexts-and-start-methods
    # pytorch docs mention Queue problem:
    # https://pytorch.org/docs/stable/notes/multiprocessing.html
    #
    # Note also that daemonising this Process is important so that resources are
    # properly freed when restarting the notebook kernel (or shutting down the
    # notebook entirely).
    data_collect_process = mp.get_context("spawn").Process(
        target=data_consumption_manager,
        args=(start_time_ms, stop_time_ms, run_id, topics, kafka_broker, consumer_type,
              stream_info, interval_s, event_buffer_size, slow_metadata_buffer_size,
              fast_metadata_buffer_size, chopper_buffer_size, worker_instruction_queue,
              data_queue, test_message_queue),
        daemon=True)
    try:
        data_stream_widget = DataStreamWidget(start_time_ms=start_time_ms,
                                              stop_time_ms=stop_time_ms,
                                              run_title=run_title)
        data_collect_process.start()

        # When testing, if something goes wrong, the while loop below can
        # become infinite. So we introduce a timeout.
        if timeout is not None:
            start_timeout = time.time()
            timeout_s = float(sc.to_unit(timeout, 's').value)
        n_warnings = 0
        while data_collect_process.is_alive(
        ) and n_data_chunks < halt_after_n_data_chunks and \
                n_warnings < halt_after_n_warnings and \
                not data_stream_widget.stop_requested:
            if timeout is not None and (time.time() - start_timeout) > timeout_s:
                raise TimeoutError("data_stream timed out in test")
            try:
                new_data = data_queue.get_nowait()

                if isinstance(new_data, Warning):
                    # Raise warnings in this process so that they
                    # can be captured in tests
                    warn(new_data)
                    n_warnings += 1
                    continue
                elif isinstance(new_data, StopTimeUpdate):
                    data_stream_widget.set_stop_time(new_data.stop_time_ms)
                    if end_at == StopTime.END_OF_RUN:
                        worker_instruction_queue.put(
                            ManagerInstruction(InstructionType.UPDATE_STOP_TIME,
                                               new_data.stop_time_ms))
                    continue
                n_data_chunks += 1
                yield convert_from_pickleable_dict(new_data)
            except QueueEmpty:
                await asyncio.sleep(0.5 * interval_s)
    finally:
        # Ensure cleanup happens however the loop exits
        worker_instruction_queue.put(ManagerInstruction(InstructionType.STOP_NOW))
        if data_collect_process.is_alive():
            process_halt_timeout_s = 4.
            data_collect_process.join(process_halt_timeout_s)
        if data_collect_process.is_alive():
            data_collect_process.terminate()
        for queue in (data_queue, worker_instruction_queue, test_message_queue):
            _cleanup_queue(queue)
        data_stream_widget.set_stopped()
