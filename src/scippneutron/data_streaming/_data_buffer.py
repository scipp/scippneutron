# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
import multiprocessing as mp
import multiprocessing.queues
import threading
from collections.abc import Callable
from datetime import datetime
from typing import Any

import numpy as np
import scipp as sc
from streaming_data_types.eventdata_ev42 import deserialise_ev42
from streaming_data_types.exceptions import WrongSchemaException
from streaming_data_types.logdata_f142 import LogDataInfo, deserialise_f142
from streaming_data_types.run_stop_6s4t import deserialise_6s4t
from streaming_data_types.sample_environment_senv import Location as TimestampLocation
from streaming_data_types.sample_environment_senv import Response as FastSampleEnvData
from streaming_data_types.sample_environment_senv import deserialise_senv
from streaming_data_types.timestamps_tdct import Timestamps, deserialise_tdct

from ..io.nexus._json_nexus import StreamInfo
from ._serialisation import convert_to_pickleable_dict
from ._stop_time import StopTimeUpdate
from ._warnings import BufferSizeWarning, UnknownFlatbufferIdWarning

# The ESS data streaming system uses Google FlatBuffers to serialise
# data to transmit in the Kafka message payload. FlatBuffers uses schemas
# to define the data structure to serialise:
# https://github.com/ess-dmsc/streaming-data-types/
# The ess-streaming-data-types library provides convenience "serialise"
# and "deserialise" functions for each schema:
# https://github.com/ess-dmsc/python-streaming-data-types/
# Each schema is identified by a 4 character string which is included in:
# - the filename of the schema,
# - the serialised buffer as the first 4 bytes,
# - the module name in ess-streaming-data-types.
#
# FlatBuffer schema ids.
# These will change in the future if a breaking change needs to be
# made to the schema.
SLOW_FB_ID = "f142"
FAST_FB_ID = "senv"
CHOPPER_FB_ID = "tdct"
EVENT_FB_ID = "ev42"


def _create_metadata_buffer_array(name: str, unit: str, dtype: Any, buffer_size: int):
    return sc.DataArray(
        sc.zeros(dims=[name], shape=(buffer_size,), unit=unit, dtype=dtype),
        coords={
            "time": sc.zeros(
                dims=[name],
                shape=(buffer_size,),
                unit=sc.Unit("nanoseconds"),
                dtype=np.dtype('datetime64[ns]'),
            )
        },
    )


class _SlowMetadataBuffer:
    """
    Buffer for "slowly" changing metadata from Kafka messages serialised
    according to the flatbuffer schema with id SLOW_FB_ID.
    Typically, the data sources are EPICS PVs with updates published to
    Kafka via the Forwarder (https://github.com/ess-dmsc/forwarder/).
    """

    def __init__(self, stream_info: StreamInfo, buffer_size: int):
        self._buffer_mutex = threading.Lock()
        self._buffer_size = buffer_size
        self._name = stream_info.source_name
        self._buffer_filled_size = 0
        self._data_array = _create_metadata_buffer_array(
            self._name, stream_info.unit, stream_info.dtype, buffer_size
        )

    def append_data(self, log_event: LogDataInfo, emit_data: Callable):
        if self._buffer_filled_size == self._buffer_size:
            emit_data()

        # Each LogDataInfo contains a single value-timestamp pair
        with self._buffer_mutex:
            self._data_array[self._name, self._buffer_filled_size] = log_event.value
            self._data_array.coords["time"][
                self._name, self._buffer_filled_size
            ].value = np.datetime64(log_event.timestamp_unix_ns, 'ns')
            self._buffer_filled_size += 1

    def get_metadata_array(self) -> tuple[bool, sc.Variable]:
        """
        Copy collected data from the buffer
        """
        with self._buffer_mutex:
            return_array = self._data_array[
                self._name, : self._buffer_filled_size
            ].copy()
            new_data_exists = self._buffer_filled_size != 0
            self._buffer_filled_size = 0
        return new_data_exists, sc.scalar(return_array)


class _FastMetadataBuffer:
    """
    Buffer for "fast" changing metadata from Kafka messages serialised
    according to the flatbuffer schema with id FAST_FB_ID.
    Typical data sources are sample environment apparatus with relatively
    rapidly values which, for efficiency, publish updates directly to Kafka
    rather than via EPICS and the Forwarder.
    """

    def __init__(
        self, stream_info: StreamInfo, buffer_size: int, data_queue: mp.queues.Queue
    ):
        self._buffer_mutex = threading.Lock()
        self._buffer_size = buffer_size
        self._name = stream_info.source_name
        self._data_queue = data_queue
        self._buffer_filled_size = 0
        self._data_array = _create_metadata_buffer_array(
            self._name, stream_info.unit, stream_info.dtype, buffer_size
        )

    def append_data(self, log_events: FastSampleEnvData, emit_data: Callable):
        # Each FastSampleEnvData contains an array of values and either:
        #  - an array of corresponding timestamps, or
        #  - the timedelta for linearly spaced timestamps
        message_size = log_events.values.size
        if message_size > self._buffer_size:
            self._data_queue.put(
                BufferSizeWarning(
                    "Single message would overflow the fast metadata buffer, "
                    "please restart with a larger buffer size:\n"
                    f"message_size: {message_size}, fast_metadata_buffer_size:"
                    f" {self._buffer_size}. These data have been "
                    f"skipped!"
                )
            )
            return

        if self._buffer_filled_size + message_size > self._buffer_size:
            emit_data()

        def _datetime_to_epoch_ns(input_timestamp: datetime) -> int:
            return int(input_timestamp.timestamp() * 1_000_000_000)

        with self._buffer_mutex:
            self._data_array[
                self._name,
                self._buffer_filled_size : self._buffer_filled_size + message_size,
            ].values = log_events.values
            if log_events.value_ts is not None:
                timestamps = log_events.value_ts
            else:
                # Generate linearly spaced sample timestamps assuming that
                # the message timestamp falls at the start of the sample range
                timestamps = np.arange(0, message_size, dtype=np.int64) * int(
                    log_events.sample_ts_delta
                ) + _datetime_to_epoch_ns(log_events.timestamp)
                if log_events.ts_location == TimestampLocation.Middle:
                    # Shift timestamps so that the message timestamp falls
                    # in the middle of the sample range
                    timestamps = timestamps - int(
                        0.5 * (timestamps[-1] - timestamps[0])
                    )
                elif log_events.ts_location == TimestampLocation.End:
                    # Shift timestamps so that the message timestamp falls at
                    # the end of the sample range
                    timestamps = timestamps - (timestamps[-1] - timestamps[0])
                elif log_events.ts_location == TimestampLocation.Start:
                    pass  # timestamps are already correct
                else:
                    # We have accounted for all enum values so this can only
                    # happen if someone really messed up when serialising the
                    # message or a breaking change was made to the schema
                    # without a new schema id being assigned (which is
                    # against ECDC policy)
                    raise RuntimeError(
                        "Unrecognised timestamp location in fast sample "
                        "environment data message (flatbuffer id: "
                        f"'{FAST_FB_ID}')"
                    )
            self._data_array[
                self._name,
                self._buffer_filled_size : self._buffer_filled_size + message_size,
            ].coords["time"].values = timestamps
            self._buffer_filled_size += message_size

    def get_metadata_array(self) -> tuple[bool, sc.Variable]:
        """
        Copy collected data from the buffer
        """
        with self._buffer_mutex:
            return_array = self._data_array[
                self._name, : self._buffer_filled_size
            ].copy()
            new_data_exists = self._buffer_filled_size != 0
            self._buffer_filled_size = 0
        return new_data_exists, sc.scalar(return_array)


class _ChopperMetadataBuffer:
    """
    Buffer for chopper top-dead-centre timestamps from Kafka messages
    serialised according to the flatbuffer schema with id CHOPPER_FB_ID.
    """

    def __init__(
        self, stream_info: StreamInfo, buffer_size: int, data_queue: mp.queues.Queue
    ):
        self._buffer_mutex = threading.Lock()
        self._buffer_size = buffer_size
        self._name = stream_info.source_name
        self._data_queue = data_queue
        self._buffer_filled_size = 0
        self._data_array = sc.zeros(
            dims=[self._name],
            shape=(buffer_size,),
            unit=sc.Unit("nanoseconds"),
            dtype=np.int64,
        )

    def append_data(self, chopper_timestamps: Timestamps, emit_data: Callable):
        # Each Timestamps contains an array of top-dead-centre timestamps
        message_size = chopper_timestamps.timestamps.size
        if message_size > self._buffer_size:
            self._data_queue.put(
                BufferSizeWarning(
                    "Single message would overflow the chopper data buffer, "
                    "please restart with a larger buffer size:\n"
                    f"message_size: {message_size}, chopper_buffer_size:"
                    f" {self._buffer_size}. These data have been "
                    f"skipped!"
                )
            )
            return

        if self._buffer_filled_size + message_size > self._buffer_size:
            emit_data()

        with self._buffer_mutex:
            self._data_array[
                self._name,
                self._buffer_filled_size : self._buffer_filled_size + message_size,
            ].values = chopper_timestamps.timestamps
            self._buffer_filled_size += message_size

    def get_metadata_array(self) -> tuple[bool, sc.Variable]:
        """
        Copy collected data from the buffer
        """
        with self._buffer_mutex:
            return_array = self._data_array[
                self._name, : self._buffer_filled_size
            ].copy()
            new_data_exists = self._buffer_filled_size != 0
            self._buffer_filled_size = 0
        return new_data_exists, sc.scalar(return_array)


metadata_ids = (SLOW_FB_ID, FAST_FB_ID, CHOPPER_FB_ID)
_MetadataBuffer = _FastMetadataBuffer | _SlowMetadataBuffer | _ChopperMetadataBuffer


class StreamedDataBuffer:
    """
    This owns the buffers for data consumed from Kafka.
    It periodically emits accumulated data to a queue
    and resets the buffer. If a buffer fills up within the emit time
    interval then data are emitted early.

    TODO: This also owns the metadata buffers. Maybe this should be moved to a
    separate place in the future?
    """

    def __init__(
        self,
        queue: mp.queues.Queue,
        event_buffer_size: int,
        slow_metadata_buffer_size: int,
        fast_metadata_buffer_size: int,
        chopper_buffer_size: int,
        interval_s: float,
        run_id: str,
    ):
        self._buffer_mutex = threading.Lock()
        self._interval_s = interval_s
        self._event_buffer_size = event_buffer_size
        self._slow_metadata_buffer_size = slow_metadata_buffer_size
        self._fast_metadata_buffer_size = fast_metadata_buffer_size
        self._chopper_buffer_size = chopper_buffer_size
        self._current_run_id = run_id
        tof_buffer = sc.zeros(
            dims=['event'],
            shape=[event_buffer_size],
            unit=sc.units.ns,
            dtype=sc.DType.int32,
        )
        id_buffer = sc.zeros(
            dims=['event'],
            shape=[event_buffer_size],
            unit=sc.units.one,
            dtype=sc.DType.int32,
        )
        pulse_times = sc.zeros(
            dims=['event'],
            shape=[event_buffer_size],
            unit=sc.units.ns,
            dtype=sc.DType.int64,
        )
        weights = sc.ones(
            dims=['event'], shape=[event_buffer_size], with_variances=True
        )
        self._events_buffer = sc.DataArray(
            weights,
            coords={
                'tof': tof_buffer,
                'detector_id': id_buffer,
                'pulse_time': pulse_times,
            },
        )
        self._current_event = 0
        self._cancelled = False
        self._notify_cancelled = threading.Condition()
        self._unrecognised_fb_id_count = 0
        self._periodic_emit: threading.Thread | None = None
        self._emit_queue = queue
        # Access metadata buffer by
        # self._metadata_buffers[flatbuffer_id][source_name]
        self._metadata_buffers: dict[str, dict[str, _MetadataBuffer]] = {
            flatbuffer_id: {} for flatbuffer_id in metadata_ids
        }

    def init_metadata_buffers(self, stream_info: list[StreamInfo]):
        """
        Create a buffer for each of the metadata sources.
        This is not in the constructor which allows the StreamedDataBuffer
        to be constructed before metadata sources are extracted from the
        run start message, this means the StreamedDataBuffer can be passed
        into data_stream to facilitate unit testing.
        """
        for stream in stream_info:
            if stream.flatbuffer_id == SLOW_FB_ID:
                self._metadata_buffers[stream.flatbuffer_id][stream.source_name] = (
                    _SlowMetadataBuffer(stream, self._slow_metadata_buffer_size)
                )
            elif stream.flatbuffer_id == FAST_FB_ID:
                self._metadata_buffers[stream.flatbuffer_id][stream.source_name] = (
                    _FastMetadataBuffer(
                        stream, self._fast_metadata_buffer_size, self._emit_queue
                    )
                )
            elif stream.flatbuffer_id == CHOPPER_FB_ID:
                self._metadata_buffers[stream.flatbuffer_id][stream.source_name] = (
                    _ChopperMetadataBuffer(
                        stream, self._chopper_buffer_size, self._emit_queue
                    )
                )
            elif stream.flatbuffer_id == EVENT_FB_ID:
                pass  # detection events, not metadata
            else:
                self._emit_queue.put(
                    UnknownFlatbufferIdWarning(
                        f"Stream in run start message specified flatbuffer id "
                        f"'{stream.flatbuffer_id}' which scippneutron does "
                        f"not know how to deserialise."
                    )
                )

    def start(self):
        self._cancelled = False
        self._unrecognised_fb_id_count = 0
        self._periodic_emit = threading.Thread(target=self._emit_loop)
        self._periodic_emit.start()

    def stop(self):
        self._cancelled = True
        with self._notify_cancelled:
            self._notify_cancelled.notifyAll()
        if self._periodic_emit is not None and self._periodic_emit.is_alive():
            self._periodic_emit.join(5.0)
        self._emit_data()  # flush the buffer

    def _emit_data(self):
        with self._buffer_mutex:
            if self._unrecognised_fb_id_count:
                self._emit_queue.put(
                    UnknownFlatbufferIdWarning(
                        f"Received {self._unrecognised_fb_id_count}"
                        " messages with unrecognised FlatBuffer ids"
                    )
                )
                self._unrecognised_fb_id_count = 0
            new_data = self._events_buffer['event', : self._current_event].copy()
            new_data_exists = self._current_event != 0
            for buffers in self._metadata_buffers.values():
                for name, buffer in buffers.items():
                    (new_metadata_exists, metadata_array) = buffer.get_metadata_array()
                    new_data.attrs[name] = metadata_array
                    if new_metadata_exists:
                        new_data_exists = True
            self._current_event = 0
        if new_data_exists:
            self._emit_queue.put(convert_to_pickleable_dict(new_data))

    def _emit_loop(self):
        while not self._cancelled:
            with self._notify_cancelled:
                self._notify_cancelled.wait(timeout=self._interval_s)
            self._emit_data()

    def _handled_event_data(self, new_data: bytes) -> bool:
        try:
            deserialised_data = deserialise_ev42(new_data)
            message_size = deserialised_data.detector_id.size
            if message_size > self._event_buffer_size:
                self._emit_queue.put(
                    BufferSizeWarning(
                        "Single message would overflow the event data buffer, "
                        "please restart with a larger buffer size:\n"
                        f"message_size: {message_size}, event_buffer_size:"
                        f" {self._event_buffer_size}. These data have been "
                        f"skipped!"
                    )
                )
                return True
            # If new data would overfill buffer then emit data
            # currently in buffer first
            if self._current_event + message_size > self._event_buffer_size:
                self._emit_data()
            with self._buffer_mutex:
                frame = self._events_buffer[
                    'event', self._current_event : self._current_event + message_size
                ]
                frame.coords['detector_id'].values = deserialised_data.detector_id
                frame.coords['tof'].values = deserialised_data.time_of_flight
                frame.coords['pulse_time'].values = (
                    deserialised_data.pulse_time
                    * np.ones_like(deserialised_data.time_of_flight)
                )
                self._current_event += message_size
        except WrongSchemaException:
            return False
        return True

    def _handled_metadata(
        self, new_data: bytes, source_field_name: str, deserialise: Callable, fb_id: str
    ) -> bool:
        try:
            deserialised_data = deserialise(new_data)
            try:
                self._metadata_buffers[fb_id][
                    getattr(deserialised_data, source_field_name)
                ].append_data(deserialised_data, self._emit_data)
                return True
            except KeyError:
                # Ignore data from unknown source name
                pass
        except WrongSchemaException:
            return False
        return False

    def _handled_stop_run(self, new_data: bytes):
        try:
            stop_run_data = deserialise_6s4t(new_data)
            if stop_run_data.job_id == self._current_run_id:
                self._emit_queue.put(StopTimeUpdate(stop_run_data.stop_time))
            return True
        except WrongSchemaException:
            return False

    def new_data(self, new_data: bytes):
        """
        This is the callback which is given to the consumers.
        """
        if self._handled_event_data(new_data):
            return
        if self._handled_metadata(
            new_data, "source_name", deserialise_f142, SLOW_FB_ID
        ):
            return
        if self._handled_metadata(new_data, "name", deserialise_senv, FAST_FB_ID):
            return
        if self._handled_metadata(new_data, "name", deserialise_tdct, CHOPPER_FB_ID):
            return
        if self._handled_stop_run(new_data):
            return
        # new data were not handled
        self._unrecognised_fb_id_count += 1
