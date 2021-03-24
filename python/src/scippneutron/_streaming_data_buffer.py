import scipp as sc
import numpy as np
import asyncio
from scipp.detail import move_to_data_array
from streaming_data_types.eventdata_ev42 import deserialise_ev42
from streaming_data_types.logdata_f142 import deserialise_f142
from streaming_data_types.exceptions import WrongSchemaException
from typing import Optional, Callable
from warnings import warn


class StreamedDataBuffer:
    """
    This owns the buffer for data consumed from Kafka.
    It periodically emits accumulated data to a processing pipeline
    and resets the buffer. If the buffer fills up within the emit time
    interval then data is emitted more frequently.
    """
    def __init__(self,
                 data_callback: Optional[Callable] = None,
                 interval_s=2.,
                 buffer_size=1048576):
        # 1048576 events is around 24 MB (with pulse_time, id, weights, etc)
        self._buffer_mutex = asyncio.Lock()
        self._interval = interval_s
        self._buffer_size = buffer_size
        tof_buffer = sc.Variable(dims=['event'],
                                 shape=[buffer_size],
                                 unit=sc.units.ns,
                                 dtype=sc.dtype.int32)
        id_buffer = sc.Variable(dims=['event'],
                                shape=[buffer_size],
                                unit=sc.units.one,
                                dtype=sc.dtype.int32)
        pulse_times = sc.Variable(dims=['event'],
                                  shape=[buffer_size],
                                  unit=sc.units.ns,
                                  dtype=sc.dtype.int64)
        weights = sc.Variable(dims=['event'],
                              unit=sc.units.one,
                              values=np.ones(buffer_size, dtype=np.float32),
                              variances=np.ones(buffer_size, dtype=np.float32))
        # TODO tof + pulse-time
        proto_events = {
            'data': weights,
            'coords': {
                'Tof': tof_buffer,
                'id': id_buffer,
                'pulse_time': pulse_times
            }
        }
        self._events_buffer = move_to_data_array(**proto_events)
        self._current_event = 0
        self._cancelled = False
        self._unrecognised_fb_id_count = 0
        self._data_callback = data_callback
        self._periodic_emit: Optional[asyncio.Task] = None

    def start(self):
        self._current_event = 0
        self._cancelled = False
        self._unrecognised_fb_id_count = 0
        self._periodic_emit = asyncio.create_task(self._emit_loop())

    def stop(self):
        self._cancelled = True
        if self._periodic_emit is not None:
            self._periodic_emit.cancel()

    async def _emit_data(self):
        try:
            async with self._buffer_mutex:
                if self._current_event == 0:
                    return
                new_data = self._events_buffer.coords['id'][
                    'event', :self._current_event].copy()
                self._current_event = 0
                if self._unrecognised_fb_id_count:
                    warn(f"Accumulator received "
                         f"{self._unrecognised_fb_id_count}"
                         " messages with unrecognised FlatBuffer ids")
                    self._unrecognised_fb_id_count = 0
            if self._data_callback is not None:
                await self._data_callback(new_data)
        except Exception as e:
            print(e)

    async def _emit_loop(self):
        while not self._cancelled:
            await asyncio.sleep(self._interval)
            await self._emit_data()

    async def new_data(self, new_data: bytes):
        try:
            # Are they event data?
            try:
                deserialised_data = deserialise_ev42(new_data)
                message_size = deserialised_data.detector_id.size
                if message_size > self._buffer_size:
                    print("Single message would overflow NewDataBuffer, "
                          "please restart with a larger buffer_size:\n"
                          f"message_size: {message_size}, buffer_size:"
                          f" {self._buffer_size}")
                # If new data would overfill buffer then emit data
                # currently in buffer first
                if self._current_event + message_size > self._buffer_size:
                    await self._emit_data()
                async with self._buffer_mutex:
                    self._events_buffer.coords['id'][
                        'event', self._current_event:self._current_event +
                        message_size] = deserialised_data.detector_id
                    self._events_buffer.coords['Tof'][
                        'event', self._current_event:self._current_event +
                        message_size] = deserialised_data.time_of_flight
                    self._events_buffer.coords['pulse_time'][
                        'event', self._current_event:self._current_event +
                        message_size] = deserialised_data.pulse_time * \
                        np.ones_like(deserialised_data.time_of_flight)
                    self._current_event += message_size
                return
            except WrongSchemaException:
                pass
            # Are they log data?
            try:
                _ = deserialise_f142(new_data)
                async with self._buffer_mutex:
                    pass  # TODO append to DataArray with matching source name
                return
            except WrongSchemaException:
                self._unrecognised_fb_id_count += 1
        except Exception as e:
            print(e)
