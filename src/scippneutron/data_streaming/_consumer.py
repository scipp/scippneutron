# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
"""
This module uses the confluent-kafka-python implementation of the
Kafka Client API to communicate with Kafka servers (brokers)

The Kafka documentation gives a useful brief introduction you may
want to read:
https://kafka.apache.org/documentation/#gettingStarted

The following Kafka terminology is used extensively in the code:
- A "topic" is a named data stream, messages are published to, or
  consumed from, a topic.
- A "partition" is a logfile of messages on the broker, each topic
  comprises one or more partitions.
- The "offset" is the index of the message in a partition on the
  Kafka broker. Old messages are deleted from the partition, so the
  first available message may not be at offset 0.
"""

import multiprocessing as mp
import multiprocessing.queues
import threading
from collections.abc import Callable
from queue import Empty as QueueEmpty
from time import time_ns
from warnings import warn

import numpy as np
from confluent_kafka import Consumer, KafkaError, TopicPartition
from streaming_data_types.exceptions import WrongSchemaException
from streaming_data_types.run_start_pl72 import RunStartInfo, deserialise_pl72

from ._consumer_type import ConsumerType


class RunStartError(Exception):
    pass


class FakeConsumer:
    """
    Use in place of confluent_kafka.Consumer
    to avoid network io in unit tests
    """

    def __init__(self, input_queue: mp.queues.Queue | None):
        if input_queue is None:
            raise RuntimeError(
                "A multiprocessing queue for test messages "
                "must be provided when using FakeConsumer"
            )
        # This queue is used to provide the consumer with
        # messages in unit tests, instead of it getting messages
        # from the Kafka broker
        self._input_queue = input_queue

    def assign(self, topic_partitions: list[TopicPartition]):
        pass

    def poll(self, timeout: float):
        try:
            msg = self._input_queue.get(timeout=0.1)
            return msg
        except QueueEmpty:
            pass

    def close(self):
        pass


class KafkaConsumer:
    def __init__(
        self,
        topic_partition: TopicPartition,
        consumer: Consumer | FakeConsumer,
        callback: Callable,
        stop_time_ms: int | None = None,
    ):
        self._consumer = consumer
        # To consume messages the consumer must "subscribe" to one
        # or more topics or "assign" specific topic partitions, the
        # latter allows us to start consuming at an offset specified
        # in the TopicPartition. Each KafkaConsumer consumes from only
        # a single partition, this simplifies the logic around run stop
        # behaviour.
        # Note that offsets are integer indices pointing to a position in the
        # topic partition, they are not a bytes offset.
        self._consumer.assign([topic_partition])
        self._callback = callback
        self._reached_eop = False
        self.cancelled = False
        self._consume_data: threading.Thread | None = None
        self.stopped = True
        self._stop_time_mutex = threading.Lock()
        # default stop time to distant future (run until manually stopped)
        self._stop_time = np.iinfo(np.int64).max
        if stop_time_ms is not None:
            self._stop_time = stop_time_ms

    def start(self):
        self.stopped = False
        self.cancelled = False
        self._consume_data = threading.Thread(target=self._consume_loop)
        self._consume_data.start()

    def _consume_loop(self):
        def time_now_ms() -> int:
            return time_ns() // 1_000_000

        reached_message_after_stop_time = False
        reached_stop_time = False
        at_end_of_partition = False

        while not self.cancelled:
            with self._stop_time_mutex:
                if time_now_ms() > self._stop_time:
                    reached_stop_time = True
                    # Wall clock time is after the run stop time.
                    # Now we just continue until we either consume a
                    # message with a timestamp that is after the stop time
                    # or we reach the end of messages available on Kafka
                    # (end of partition error).
                    if reached_message_after_stop_time:
                        # We've already seen a message timestamped
                        # after the stop time.
                        self.cancelled = True
                        break
            msg = self._consumer.poll(timeout=2.0)
            if msg is None:
                if reached_stop_time and at_end_of_partition:
                    self.cancelled = True
                    break
                continue
            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    at_end_of_partition = True
                    if reached_stop_time:
                        # Wall clock time is after run stop time and there
                        # are no more messages available on Kafka for us to
                        # consume, so cancel running the consumer.
                        self.cancelled = True
                        break
                    continue
                warn(f"Message error in consumer: {msg.error()}", stacklevel=3)
                self.cancelled = True
                break

            at_end_of_partition = False
            with self._stop_time_mutex:
                if msg.timestamp()[1] > self._stop_time:
                    reached_message_after_stop_time = True
                    if reached_stop_time:
                        # Wall clock time is after run stop time and remaining
                        # messages on Kafka have timestamps after the stop
                        # time, so cancel running the consumer.
                        self.cancelled = True
                        break
                    continue
            self._callback(msg.value())

    def stop(self):
        self.cancelled = True
        if self._consume_data is not None and self._consume_data.is_alive():
            self._consume_data.join(5.0)
        self._consumer.close()
        self.stopped = True

    def update_stop_time(self, new_stop_time_ms: int):
        with self._stop_time_mutex:
            self._stop_time = new_stop_time_ms


class KafkaQueryConsumer:
    """
    Wraps Kafka library consumer methods which query the
    broker for metadata and poll for single messages.
    It is a thin wrapper but allows a fake to be used
    in unit tests.
    """

    def __init__(self, broker: str):
        # Set "enable.auto.commit" to False, as we do not need to report to the
        # kafka broker where we got to (it usually does this in case of a
        # crash, but we simply restart the process and go and find the last
        # run_start message.
        #
        # Set "queued.min.messages" to 1 as we will consume backwards through
        # the partition one message at a time; we do not want to retrieve
        # multiple messages in the forward direction each time we step
        # backwards by 1 offset
        conf = {
            "bootstrap.servers": broker,
            "group.id": "consumer_group_name",
            "auto.offset.reset": "latest",
            "enable.auto.commit": False,
            "queued.min.messages": 1,
        }
        self._consumer = Consumer(**conf)

    def get_topic_partitions(self, topic: str, offset: int = -1):
        metadata = self._consumer.list_topics(topic)
        return [
            TopicPartition(topic, partition[1].id, offset=offset)
            for partition in metadata.topics[topic].partitions.items()
        ]

    def seek(self, partition: TopicPartition):
        """
        Set offset in partition, the consumer will seek to that offset
        """
        self._consumer.seek(partition)

    def poll(self, timeout=2.0):
        """
        Poll for a message from Kafka
        """
        return self._consumer.poll(timeout=timeout)

    def get_watermark_offsets(self, partition: TopicPartition) -> tuple[int, int]:
        """
        Get the offset of the first and last available
        message in the given partition
        """
        return self._consumer.get_watermark_offsets(partition, cached=False)

    def assign(self, partitions: list[TopicPartition]):
        self._consumer.assign(partitions)

    def offsets_for_times(self, partitions: list[TopicPartition]):
        return self._consumer.offsets_for_times(partitions)


def create_consumers(
    start_time_ms: int,
    stop_time_ms: int | None,
    topics: list[str],
    kafka_broker: str,
    consumer_type_enum: ConsumerType,  # so we can inject fake consumer
    callback: Callable,
    test_message_queue: mp.queues.Queue | None,
) -> list[KafkaConsumer]:
    """
    Creates one consumer per TopicPartition that start consuming
    at specified timestamp in the data stream

    Having each consumer only be responsible for one partition
    greatly simplifies the logic around stopping at the end of
    the stream (making use of "end of partition" event)
    """
    topic_partitions = []
    if consumer_type_enum == ConsumerType.REAL:
        query_consumer = KafkaQueryConsumer(kafka_broker)
        for topic in topics:
            topic_partitions.extend(
                query_consumer.get_topic_partitions(topic, offset=start_time_ms)
            )
        topic_partitions = query_consumer.offsets_for_times(topic_partitions)

    # Run start messages are typically much larger than the
    # default maximum message size of 1MB. There are
    # corresponding settings on the broker.
    # Note: the "message.max.bytes" does not necessarily have to agree with the
    # size set in the broker. The lower of the two will set the limit.
    # If a message exceeds this maximum size, an error should be reported by
    # the software publishing the run start message (for example NICOS).
    config = {
        "bootstrap.servers": kafka_broker,
        "group.id": "consumer_group_name",
        "auto.offset.reset": "latest",
        "enable.auto.commit": False,
        "message.max.bytes": 100_000_000,
        "fetch.message.max.bytes": 100_000_000,
        "enable.partition.eof": True,  # used by consumer stop logic
    }

    if consumer_type_enum == ConsumerType.REAL:
        consumers = [
            KafkaConsumer(topic_partition, Consumer(config), callback, stop_time_ms)
            for topic_partition in topic_partitions
        ]
    else:
        consumers = [
            KafkaConsumer(
                TopicPartition(""),
                FakeConsumer(test_message_queue),
                callback,
                stop_time_ms,
            )
        ]

    return consumers


def start_consumers(consumers: list[KafkaConsumer]):
    for consumer in consumers:
        consumer.start()


def stop_consumers(consumers: list[KafkaConsumer]):
    for consumer in consumers:
        consumer.stop()


def all_consumers_stopped(consumers: list[KafkaConsumer]) -> bool:
    for consumer in consumers:
        if not consumer.stopped:
            # if the consumer is cancelled then cleanly stop it
            # (join the thread)
            if consumer.cancelled:
                consumer.stop()
                continue
            return False
    return True


def get_run_start_message(
    topic: str, query_consumer: KafkaQueryConsumer
) -> RunStartInfo:
    """
    Get the last run start message on the given topic.

    TODO: we may need to carry out some filtering on instrument name,
    since it is possible that start messages from other instruments end up on
    the same Kafka topic (unless there is a separate topic for each instrument)
    """
    topic_partitions = query_consumer.get_topic_partitions(topic)
    n_partitions = len(topic_partitions)
    if n_partitions != 1:
        raise RuntimeError(
            f"Expected run start topic to contain exactly one partition, "
            f"the specified topic '{topic}' has {n_partitions} partitions."
        )
    partition = topic_partitions[0]
    low_watermark_offset, current_offset = query_consumer.get_watermark_offsets(
        partition
    )
    partition.offset = current_offset
    query_consumer.assign([partition])

    # Consume backwards from the end of the partition
    # until we find a run start message or reach the
    # start of the partition
    while current_offset > low_watermark_offset:
        current_offset -= 1
        partition.offset = current_offset
        query_consumer.seek(partition)
        message = query_consumer.poll(timeout=2.0)
        if message is None:
            raise RunStartError("Timed out when trying to retrieve run start message")
        elif message.error():
            raise RunStartError(
                f"Error encountered consuming run start " f"message: {message.error()}"
            )
        try:
            return deserialise_pl72(message.value())
        except WrongSchemaException:
            # Not a run start message, keep trying
            pass

    raise RunStartError(f"Run start message not found in topic '{topic}'")
