import scipp as sc
import numpy as np
import asyncio
from scipp.detail import move_to_data_array
from streaming_data_types.eventdata_ev42 import deserialise_ev42
from streaming_data_types.exceptions import WrongSchemaException
from typing import Optional
from warnings import warn


class StreamedDataBuffer:
    """
    This owns the buffer for data consumed from Kafka.
    It periodically emits accumulated data to a processing pipeline
    and resets the buffer. If the buffer fills up within the emit time
    interval then data is emitted more frequently.
    """
    def __init__(self,
                 queue: asyncio.Queue,
                 interval_s: float = 2.,
                 buffer_size: int = 1048576):
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
        proto_events = {
            'data': weights,
            'coords': {
                'tof': tof_buffer,
                'detector_id': id_buffer,
                'pulse_time': pulse_times
            }
        }
        self._events_buffer = move_to_data_array(**proto_events)
        self._current_event = 0
        self._cancelled = False
        self._unrecognised_fb_id_count = 0
        self._periodic_emit: Optional[asyncio.Task] = None
        self._emit_queue = queue

    def start(self):
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
                new_data = self._events_buffer[
                    'event', :self._current_event].copy()
                self._current_event = 0
                if self._unrecognised_fb_id_count:
                    warn(f"Accumulator received "
                         f"{self._unrecognised_fb_id_count}"
                         " messages with unrecognised FlatBuffer ids")
                    self._unrecognised_fb_id_count = 0
            self._emit_queue.put_nowait(new_data)
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
                    warn("Single message would overflow NewDataBuffer, "
                         "please restart with a larger buffer_size:\n"
                         f"message_size: {message_size}, buffer_size:"
                         f" {self._buffer_size}")
                # If new data would overfill buffer then emit data
                # currently in buffer first
                if self._current_event + message_size > self._buffer_size:
                    await self._emit_data()
                async with self._buffer_mutex:
                    self._events_buffer.coords['detector_id'][
                        'event', self._current_event:self._current_event +
                        message_size] = deserialised_data.detector_id
                    self._events_buffer.coords['tof'][
                        'event', self._current_event:self._current_event +
                        message_size] = deserialised_data.time_of_flight
                    self._events_buffer.coords['pulse_time'][
                        'event', self._current_event:self._current_event +
                        message_size] = deserialised_data.pulse_time * \
                        np.ones_like(deserialised_data.time_of_flight)
                    self._current_event += message_size
                return
            except WrongSchemaException:
                self._unrecognised_fb_id_count += 1
        except Exception as e:
            print(e)
