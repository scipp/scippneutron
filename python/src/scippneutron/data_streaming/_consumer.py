from confluent_kafka import Consumer, TopicPartition, KafkaError
from typing import Callable, List, Dict, Optional, Tuple
from warnings import warn
import threading
from streaming_data_types.run_start_pl72 import deserialise_pl72
from streaming_data_types.exceptions import WrongSchemaException
from ._consumer_type import ConsumerType
import multiprocessing as mp
from queue import Empty as QueueEmpty
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


class RunStartError(Exception):
    pass


class KafkaConsumer:
    def __init__(self, topic_partitions: List[TopicPartition], conf: Dict,
                 callback: Callable):
        conf['enable.partition.eof'] = True
        self._consumer = Consumer(conf)
        # To consume messages the consumer must "subscribe" to one
        # or more topics or "assign" specific topic partitions, the
        # latter allows us to start consuming at an offset specified
        # in the TopicPartition
        self._consumer.assign(topic_partitions)
        self._callback = callback
        self._reached_eop = False
        self._cancelled = False
        self._consume_data: Optional[threading.Thread] = None
        self.stopped = True

    def start(self):
        self.stopped = False
        self._cancelled = False
        self._consume_data = threading.Thread(target=self._consume_loop)
        self._consume_data.start()

    def _consume_loop(self):
        while not self._cancelled:
            msg = self._consumer.poll(timeout=1.0)
            if msg is None:
                continue
            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    continue
                warn(f"Message error in consumer: {msg.error()}")
                break
            self._callback(msg.value())

    def stop(self):
        if not self._cancelled:
            self._cancelled = True
            if self._consume_data is not None and self._consume_data.is_alive(
            ):
                self._consume_data.join(5.)
        self._consumer.close()
        self.stopped = True


class FakeConsumer:
    """
    Use in place of KafkaConsumer to avoid having to do
    network IO in unit tests.
    """
    def __init__(self, callback: Callable, input_queue: Optional[mp.Queue]):
        if input_queue is None:
            raise RuntimeError("A multiprocessing queue for test messages "
                               "must be provided when using FakeConsumer")
        self.stopped = True
        self._cancelled = False
        self._callback = callback
        self._consume_data: Optional[threading.Thread] = None
        # This queue is used to provide the consumer with
        # messages in unit tests
        self._input_queue = input_queue

    def start(self):
        self.stopped = False
        self._cancelled = False
        self._consume_data = threading.Thread(target=self._consume_loop)
        self._consume_data.start()

    def _consume_loop(self):
        while not self._cancelled:
            try:
                msg = self._input_queue.get(timeout=10.)
                self._callback(msg)
            except QueueEmpty:
                pass
            except (ValueError, OSError):
                # Queue has been closed, stop worker
                self.stop()
                break

    def stop(self):
        if not self._cancelled:
            self._cancelled = True
            if self._consume_data is not None and self._consume_data.is_alive(
            ):
                self._consume_data.join(5.)
        self.stopped = True


class KafkaQueryConsumer:
    """
    Wraps Kafka library consumer methods which query the
    broker for metadata and poll for single messages.
    It is a thin wrapper but allows a fake to be used
    in unit tests.
    """
    def __init__(self, broker: str):
        # Set "queued.min.messages" to 1 as we will consume backwards through
        # the partition one message at a time; we do not want to retrieve
        # multiple messages in the forward direction each time we step
        # backwards by 1 offset
        conf = {
            "bootstrap.servers": broker,
            "group.id": "consumer_group_name",
            "auto.offset.reset": "latest",
            "enable.auto.commit": False,
            "queued.min.messages": 1
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

    def poll(self, timeout=2.):
        """
        Poll for a message from Kafka
        """
        return self._consumer.poll(timeout=timeout)

    def get_watermark_offsets(self,
                              partition: TopicPartition) -> Tuple[int, int]:
        """
        Get the offset of the first and last available
        message in the given partition
        """
        return self._consumer.get_watermark_offsets(partition, cached=False)

    def assign(self, partitions: List[TopicPartition]):
        self._consumer.assign(partitions)

    def offsets_for_times(self, partitions: List[TopicPartition]):
        return self._consumer.offsets_for_times(partitions)


def create_consumers(
        start_time_ms: int,
        topics: List[str],
        kafka_broker: str,
        consumer_type_enum: ConsumerType,  # so we can inject fake consumer
        callback: Callable,
        test_message_queue: Optional[mp.Queue]) -> List[KafkaConsumer]:
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
                query_consumer.get_topic_partitions(topic,
                                                    offset=start_time_ms))
        topic_partitions = query_consumer.offsets_for_times(topic_partitions)

    config = {
        "bootstrap.servers": kafka_broker,
        "group.id": "consumer_group_name",
        "auto.offset.reset": "latest",
        "enable.auto.commit": False,
        "message.max.bytes": 100_000_000,
        "fetch.message.max.bytes": 100_000_000,
    }

    if consumer_type_enum == ConsumerType.REAL:
        consumers = [
            KafkaConsumer([topic_partition], config, callback)
            for topic_partition in topic_partitions
        ]
    else:
        consumers = [FakeConsumer(callback, test_message_queue)]

    return consumers


def start_consumers(consumers: List[KafkaConsumer]):
    for consumer in consumers:
        consumer.start()


def stop_consumers(consumers: List[KafkaConsumer]):
    for consumer in consumers:
        consumer.stop()


def get_run_start_message(topic: str, query_consumer: KafkaQueryConsumer):
    """
    Get the last run start message on the given topic
    """
    topic_partitions = query_consumer.get_topic_partitions(topic)
    n_partitions = len(topic_partitions)
    if n_partitions != 1:
        raise RuntimeError(
            f"Expected run start topic to contain exactly one partition, "
            f"the specified topic '{topic}' has {n_partitions} partitions.")
    partition = topic_partitions[0]
    low_watermark_offset, current_offset = \
        query_consumer.get_watermark_offsets(partition)
    partition.offset = current_offset
    query_consumer.assign([partition])

    # Consume backwards from the end of the partition
    # until we find a run start message or reach the
    # start of the partition
    while current_offset > low_watermark_offset:
        current_offset -= 1
        partition.offset = current_offset
        query_consumer.seek(partition)
        message = query_consumer.poll(timeout=2.)
        if message is None:
            raise RunStartError(
                "Timed out when trying to retrieve run start message")
        elif message.error():
            raise RunStartError(f"Error encountered consuming run start "
                                f"message: {message.error()}")
        try:
            return deserialise_pl72(message.value())
        except WrongSchemaException:
            # Not a run start message, keep trying
            pass

    raise RunStartError(f"Run start message not found in topic '{topic}'")
