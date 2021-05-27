import scipp as sc
import numpy as np
import asyncio
from streaming_data_types.eventdata_ev42 import deserialise_ev42
from streaming_data_types.logdata_f142 import deserialise_f142, LogDataInfo
from streaming_data_types.sample_environment_senv import deserialise_senv
from streaming_data_types.sample_environment_senv import (Response as
                                                          FastSampleEnvData)
from streaming_data_types.sample_environment_senv import (Location as
                                                          TimestampLocation)
from streaming_data_types.timestamps_tdct import deserialise_tdct, Timestamps
from streaming_data_types.exceptions import WrongSchemaException
from typing import Optional, Dict, List, Any, Union, Callable
from warnings import warn
from ._loading_json_nexus import StreamInfo
"""
The ESS data streaming system uses Google FlatBuffers to serialise
data to transmit in the Kafka message payload. FlatBuffers uses schemas
to define the data structure to serialise:
https://github.com/ess-dmsc/streaming-data-types/
The ess-streaming-data-types library provides convenience "serialise"
and "deserialise" functions for each schema:
https://github.com/ess-dmsc/python-streaming-data-types/
Each schema is identified by a 4 character string which is included in:
- the filename of the schema,
- the serialised buffer as the first 4 bytes,
- the module name in ess-streaming-data-types.
"""


def _create_metadata_buffer_array(name: str, unit: sc.Unit, dtype: Any,
                                  buffer_size: int):
    return sc.DataArray(
        sc.zeros(dims=[name], shape=(buffer_size, ), unit=unit, dtype=dtype), {
            "time":
            sc.zeros(dims=[name],
                     shape=(buffer_size, ),
                     unit=sc.Unit("nanoseconds"),
                     dtype=np.int64)
        })


# FlatBuffer schema ids.
# These will change in the future if a breaking change needs to be
# made to the schema.
SLOW_FB_ID = "f142"
FAST_FB_ID = "senv"
CHOPPER_FB_ID = "tdct"


class _SlowMetadataBuffer:
    """
    Buffer for "slowly" changing metadata from Kafka messages serialised
    according to the flatbuffer schema with id SLOW_FB_ID.
    Typically the data sources are EPICS PVs with updates published to
    Kafka via the Forwarder (https://github.com/ess-dmsc/forwarder/).
    """
    def __init__(self, stream_info: StreamInfo, buffer_size: int = 1000):
        self._buffer_mutex = asyncio.Lock()
        self._buffer_size = buffer_size
        self._name = stream_info.source_name
        self._buffer_filled_size = 0
        self._data_array = _create_metadata_buffer_array(
            self._name, stream_info.unit, stream_info.dtype, buffer_size)

    async def append_data(self, log_event: LogDataInfo):
        # Each LogDataInfo contains a single value-timestamp pair
        async with self._buffer_mutex:
            self._data_array[
                self._name, self._buffer_filled_size:self._buffer_filled_size +
                1].values[0] = log_event.value
            self._data_array[
                self._name, self._buffer_filled_size:self._buffer_filled_size +
                1].coords["time"].values[0] = log_event.timestamp_unix_ns
            self._buffer_filled_size += 1

    async def get_metadata_array(self) -> sc.Variable:
        """
        Copy collected data from the buffer
        """
        async with self._buffer_mutex:
            return_array = self._data_array[
                self._name, :self._buffer_filled_size].copy()
            self._buffer_filled_size = 0
        return sc.Variable(value=return_array)


class _FastMetadataBuffer:
    """
    Buffer for "fast" changing metadata from Kafka messages serialised
    according to the flatbuffer schema with id FAST_FB_ID.
    Typical data sources are sample environment apparatus with relatively
    rapidly values which, for efficiency, publish updates directly to Kafka
    rather than via EPICS and the Forwarder.
    """
    def __init__(self, stream_info: StreamInfo, buffer_size: int = 100_000):
        self._buffer_mutex = asyncio.Lock()
        self._buffer_size = buffer_size
        self._name = stream_info.source_name
        self._buffer_filled_size = 0
        self._data_array = _create_metadata_buffer_array(
            self._name, stream_info.unit, stream_info.dtype, buffer_size)

    async def append_data(self, log_events: FastSampleEnvData):
        # Each FastSampleEnvData contains an array of values and either:
        #  - an array of corresponding timestamps, or
        #  - the timedelta for linearly spaced timestamps
        number_of_values = log_events.values.size
        async with self._buffer_mutex:
            self._data_array[
                self._name, self._buffer_filled_size:self._buffer_filled_size +
                number_of_values].values = log_events.values
            if log_events.value_ts is not None:
                timestamps = log_events.value_ts
            else:
                timestamps = np.arange(
                    0, number_of_values) * log_events.sample_ts_delta + int(
                        log_events.timestamp.timestamp() * 1_000_000_000)
                if log_events.ts_location == TimestampLocation.Middle:
                    timestamps = timestamps - 0.5 * (timestamps[-1] -
                                                     timestamps[0])
                elif log_events.ts_location == TimestampLocation.End:
                    timestamps = timestamps - (timestamps[-1] - timestamps[0])
            self._data_array[
                self._name, self._buffer_filled_size:self._buffer_filled_size +
                number_of_values].coords["time"].values = timestamps
            self._buffer_filled_size += number_of_values

    async def get_metadata_array(self) -> sc.Variable:
        """
        Copy collected data from the buffer
        """
        async with self._buffer_mutex:
            return_array = self._data_array[
                self._name, :self._buffer_filled_size].copy()
            self._buffer_filled_size = 0
        return sc.Variable(value=return_array)


class _ChopperMetadataBuffer:
    """
    Buffer for chopper top-dead-centre timestamps from Kafka messages
    serialised according to the flatbuffer schema with id CHOPPER_FB_ID.
    """
    def __init__(self, stream_info: StreamInfo, buffer_size: int = 10_000):
        self._buffer_mutex = asyncio.Lock()
        self._buffer_size = buffer_size
        self._name = stream_info.source_name
        self._buffer_filled_size = 0
        self._data_array = sc.zeros(dims=[self._name],
                                    shape=(buffer_size, ),
                                    unit=sc.Unit("nanoseconds"),
                                    dtype=np.int64)

    async def append_data(self, chopper_timestamps: Timestamps):
        # Each Timestamps contains an array of top-dead-centre timestamps
        async with self._buffer_mutex:
            self._data_array[
                self._name, self._buffer_filled_size:self._buffer_filled_size +
                chopper_timestamps.timestamps.
                size].values = chopper_timestamps.timestamps
            self._buffer_filled_size += chopper_timestamps.timestamps.size

    async def get_metadata_array(self) -> sc.Variable:
        """
        Copy collected data from the buffer
        """
        async with self._buffer_mutex:
            return_array = self._data_array[
                self._name, :self._buffer_filled_size].copy()
            self._buffer_filled_size = 0
        return return_array


metadata_ids = (SLOW_FB_ID, FAST_FB_ID, CHOPPER_FB_ID)
_MetadataBuffer = Union[_FastMetadataBuffer, _SlowMetadataBuffer,
                        _ChopperMetadataBuffer]


class StreamedDataBuffer:
    """
    This owns the buffer for data consumed from Kafka.
    It periodically emits accumulated data to a processing pipeline
    and resets the buffer. If the buffer fills up within the emit time
    interval then data is emitted more frequently.
    """
    def __init__(self, queue: asyncio.Queue, event_buffer_size: int,
                 interval: sc.Variable):
        self._buffer_mutex = asyncio.Lock()
        self._interval_s = sc.to_unit(interval, 's').value
        self._buffer_size = event_buffer_size
        tof_buffer = sc.zeros(dims=['event'],
                              shape=[event_buffer_size],
                              unit=sc.units.ns,
                              dtype=sc.dtype.int32)
        id_buffer = sc.zeros(dims=['event'],
                             shape=[event_buffer_size],
                             unit=sc.units.one,
                             dtype=sc.dtype.int32)
        pulse_times = sc.zeros(dims=['event'],
                               shape=[event_buffer_size],
                               unit=sc.units.ns,
                               dtype=sc.dtype.int64)
        weights = sc.ones(dims=['event'],
                          shape=[event_buffer_size],
                          variances=True)
        self._events_buffer = sc.DataArray(weights, {
            'tof': tof_buffer,
            'detector_id': id_buffer,
            'pulse_time': pulse_times
        })
        self._current_event = 0
        self._cancelled = False
        self._unrecognised_fb_id_count = 0
        self._periodic_emit: Optional[asyncio.Task] = None
        self._emit_queue = queue
        # Access metadata buffer by
        # self._metadata_buffers[flatbuffer_id][source_name]
        self._metadata_buffers: Dict[str, Dict[str, _MetadataBuffer]] = {
            flatbuffer_id: {}
            for flatbuffer_id in metadata_ids
        }

    def init_metadata_buffers(self, stream_info: List[StreamInfo]):
        """
        Create a buffer for each of the metadata sources.
        This is not in the constructor which allows the StreamedDataBuffer
        to be constructed before metadata sources are extracted from the
        run start message, this means the StreamedDataBuffer can be passed
        into data_stream to facilitate unit testing.
        """
        for stream in stream_info:
            if stream.flatbuffer_id == SLOW_FB_ID:
                self._metadata_buffers[stream.flatbuffer_id][
                    stream.source_name] = _SlowMetadataBuffer(stream)
            elif stream.flatbuffer_id == FAST_FB_ID:
                self._metadata_buffers[stream.flatbuffer_id][
                    stream.source_name] = _FastMetadataBuffer(stream)
            elif stream.flatbuffer_id == CHOPPER_FB_ID:
                self._metadata_buffers[stream.flatbuffer_id][
                    stream.source_name] = _ChopperMetadataBuffer(stream)

    def start(self):
        self._cancelled = False
        self._unrecognised_fb_id_count = 0
        self._periodic_emit = asyncio.create_task(self._emit_loop())

    def stop(self):
        self._cancelled = True
        if self._periodic_emit is not None:
            self._periodic_emit.cancel()

    async def _emit_data(self):
        async with self._buffer_mutex:
            if self._unrecognised_fb_id_count:
                warn(f"Received {self._unrecognised_fb_id_count}"
                     " messages with unrecognised FlatBuffer ids")
                self._unrecognised_fb_id_count = 0
            new_data = self._events_buffer[
                'event', :self._current_event].copy()
            for _, buffers in self._metadata_buffers.items():
                for name, buffer in buffers.items():
                    metadata_array = await buffer.get_metadata_array()
                    new_data.attrs[name] = metadata_array
            self._current_event = 0
        self._emit_queue.put_nowait(new_data)

    async def _emit_loop(self):
        while not self._cancelled:
            await asyncio.sleep(self._interval_s)
            await self._emit_data()

    async def _handled_event_data(self, new_data: bytes) -> bool:
        try:
            deserialised_data = deserialise_ev42(new_data)
            message_size = deserialised_data.detector_id.size
            if message_size > self._buffer_size:
                warn("Single message would overflow NewDataBuffer, "
                     "please restart with a larger buffer_size:\n"
                     f"message_size: {message_size}, buffer_size:"
                     f" {self._buffer_size}. These data have been "
                     f"skipped!")
                return True
            # If new data would overfill buffer then emit data
            # currently in buffer first
            if self._current_event + message_size > self._buffer_size:
                await self._emit_data()
            async with self._buffer_mutex:
                frame = self._events_buffer[
                    'event',
                    self._current_event:self._current_event + message_size]
                frame.coords[
                    'detector_id'].values = deserialised_data.detector_id
                frame.coords['tof'].values = deserialised_data.time_of_flight
                frame.coords['pulse_time'].values = \
                    deserialised_data.pulse_time * \
                    np.ones_like(deserialised_data.time_of_flight)
                self._current_event += message_size
        except WrongSchemaException:
            return False
        return True

    async def _handled_metadata(self, new_data: bytes, source_field_name: str,
                                deserialise: Callable, fb_id: str) -> bool:
        try:
            deserialised_data = deserialise(new_data)
            try:
                await self._metadata_buffers[fb_id][getattr(
                    deserialised_data,
                    source_field_name)].append_data(deserialised_data)
            except KeyError:
                # Ignore data from unknown source name
                pass
        except WrongSchemaException:
            return False
        return False

    async def new_data(self, new_data: bytes):
        data_handled = await self._handled_event_data(new_data)
        if data_handled:
            return
        data_handled = await self._handled_metadata(new_data, "source_name",
                                                    deserialise_f142,
                                                    SLOW_FB_ID)
        if data_handled:
            return
        data_handled = await self._handled_metadata(new_data, "name",
                                                    deserialise_senv,
                                                    FAST_FB_ID)
        if data_handled:
            return
        data_handled = await self._handled_metadata(new_data, "name",
                                                    deserialise_tdct,
                                                    CHOPPER_FB_ID)
        if data_handled:
            return
        # new data were not handled
        self._unrecognised_fb_id_count += 1
