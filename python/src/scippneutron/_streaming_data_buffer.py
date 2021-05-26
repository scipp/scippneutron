import scipp as sc
import numpy as np
import asyncio
from streaming_data_types.eventdata_ev42 import deserialise_ev42
from streaming_data_types.logdata_f142 import deserialise_f142, LogDataInfo
from streaming_data_types.exceptions import WrongSchemaException
from typing import Optional, Dict, List
from warnings import warn
from ._loading_json_nexus import StreamInfo
"""
The ESS data streaming system uses Google FlatBuffers to serialise
data to transmit in the Kafka message payload. FlatBuffers uses schemas
to define the data structure to serialise:
https://github.com/ess-dmsc/streaming-data-types/
The ess-streaming-data-types library provides convenience "serialise"
and "deserialise" functions for each schema.
Each schema is identified by a 4 character string which is included in:
- the filename of the schema,
- the serialised buffer as the first 4 bytes,
- the module name in ess-streaming-data-types.
"""


class _Bufferf142:
    """
    Buffer for metadata from Kafka messages serialised according
    to the flatbuffer schema with id f142
    """
    def __init__(self, stream_info: StreamInfo, buffer_size: int = 100_000):
        self._buffer_mutex = asyncio.Lock()
        self._buffer_size = buffer_size
        self._unit = stream_info.unit
        self._name = stream_info.source_name
        self._dtype = stream_info.dtype
        self._buffer_filled_size = 0
        self._create_buffer_array(buffer_size)

    def _create_buffer_array(self, buffer_size: int):
        self._data_array = sc.DataArray(
            sc.zeros(dims=[self._name],
                     shape=(buffer_size, ),
                     unit=self._unit,
                     dtype=self._dtype), {
                         "time":
                         sc.zeros(dims=[self._name],
                                  shape=(buffer_size, ),
                                  unit=sc.Unit("nanoseconds"),
                                  dtype=np.int64)
                     })

    async def append_event(self, log_event: LogDataInfo):
        async with self._buffer_mutex:
            self._data_array["time"][
                self._buffer_filled_size] = log_event.value
            self._buffer_filled_size += 1

    async def get_metadata_array(self) -> sc.DataArray:
        """
        Copy collected data from the buffer
        """
        async with self._buffer_mutex:
            return_array = self._data_array[
                self._name, :self._buffer_filled_size].copy()
            self._buffer_filled_size = 0
        return return_array


metadata_ids = ("f142", )  # "tdct", "senv"


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
        self._metadata_buffers: Dict[str, Dict[str, _Bufferf142]] = {
            flatbuffer_id: {}
            for flatbuffer_id in metadata_ids
        }

    def init_metadata_buffers(self, stream_info: List[StreamInfo]):
        """
        Create a buffer dataset for each of the metadata sources.
        This is not in the constructor which allows the StreamedDataBuffer
        to be constructed before metadata sources are extracted from the
        run start message, this means the StreamedDataBuffer can be passed
        into data_stream to facilitate unit testing.
        """
        for stream in stream_info:
            if stream.flatbuffer_id in metadata_ids:
                self._metadata_buffers[stream.flatbuffer_id][
                    stream.source_name] = _Bufferf142(stream)

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
            # TODO remove this? does it cause a problem?
            # if self._current_event == 0:
            #     return
            new_data = self._events_buffer[
                'event', :self._current_event].copy()
            for _, buffers in self._metadata_buffers.items():
                for name, buffer in buffers.items():
                    metadata_array = await buffer.get_metadata_array()
                    new_data.attrs[name] = sc.Variable(value=metadata_array)
            self._current_event = 0
        self._emit_queue.put_nowait(new_data)

    async def _emit_loop(self):
        while not self._cancelled:
            await asyncio.sleep(self._interval_s)
            await self._emit_data()

    async def _handled_ev42_data(self, new_data: bytes) -> bool:
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

    async def _handled_f142_data(self, new_data: bytes) -> bool:
        try:
            deserialised_data = deserialise_f142(new_data)
            try:
                print("not working?")
                await self._metadata_buffers["f142"][
                    deserialised_data.source_name
                ].append_event(deserialised_data)
                print("appended f142 data to buffer!")
            except KeyError:
                # Ignore data from unknown source name
                pass
        except WrongSchemaException:
            return False
        return True

    async def _handled_senv_data(self, new_data: bytes) -> bool:
        return False

    async def _handled_tdct_data(self, new_data: bytes) -> bool:
        return False

    async def new_data(self, new_data: bytes):
        data_handled = await self._handled_ev42_data(new_data)
        if data_handled:
            return
        data_handled = await self._handled_f142_data(new_data)
        if data_handled:
            return
        data_handled = await self._handled_senv_data(new_data)
        if data_handled:
            return
        data_handled = await self._handled_tdct_data(new_data)
        if data_handled:
            return
        # new data were not handled
        self._unrecognised_fb_id_count += 1
