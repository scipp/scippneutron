import scipp as sc
import numpy as np
import asyncio
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
    def __init__(self, queue: asyncio.Queue, buffer_size: int,
                 interval: sc.Variable):
        self._buffer_mutex = asyncio.Lock()
        self._interval_s = sc.to_unit(interval, 's').value
        self._buffer_size = buffer_size
        tof_buffer = sc.zeros(dims=['event'],
                              shape=[buffer_size],
                              unit=sc.units.ns,
                              dtype=sc.dtype.int32)
        id_buffer = sc.zeros(dims=['event'],
                             shape=[buffer_size],
                             unit=sc.units.one,
                             dtype=sc.dtype.int32)
        pulse_times = sc.zeros(dims=['event'],
                               shape=[buffer_size],
                               unit=sc.units.ns,
                               dtype=sc.dtype.int64)
        weights = sc.ones(dims=['event'], shape=[buffer_size], variances=True)
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
            if self._current_event == 0:
                return
            new_data = self._events_buffer[
                'event', :self._current_event].copy()
            self._current_event = 0
        self._emit_queue.put_nowait(new_data)

    async def _emit_loop(self):
        while not self._cancelled:
            await asyncio.sleep(self._interval_s)
            await self._emit_data()

    async def new_data(self, new_data: bytes):
        # Are they event data?
        try:
            deserialised_data = deserialise_ev42(new_data)
            message_size = deserialised_data.detector_id.size
            if message_size > self._buffer_size:
                warn("Single message would overflow NewDataBuffer, "
                     "please restart with a larger buffer_size:\n"
                     f"message_size: {message_size}, buffer_size:"
                     f" {self._buffer_size}. These data have been "
                     f"skipped!")
                return
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
            return
        except WrongSchemaException:
            self._unrecognised_fb_id_count += 1
