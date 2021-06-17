The Data Streaming Interface
============================

This interface provides access to data in the ESS data streaming system. The functionality
is included in ``scippneutron`` rather than the facility-specific ``ess`` library as
the same system is also used to varying extents at other neutron sources including ISIS
Pulsed Neutron and Muon Source, SINQ: The Swiss Spallation Neutron Source, and ANSTO:
Austrailian Centre for Neutron Scattering.


Apache Kafka
------------

The ESS data streaming system is based on the Apache Kafka data streaming platform. The
basic terminology used in the scippneutron codebase is described here, but the
introductory documentation at `<https://kafka.apache.org>`_ is well worth 5-10 minutes
of your time.

Apache Kafka is a publish-subscribe system for passing data over the network. A *producer* client
publishes data to a named data stream, a *topic*, on a Kafka server, a *broker*. Kafka brokers are
usually deployed as a cluster to increase the data throughput they can support and also provide
data and service redundancy in case a broker goes down due to failure or to be updated etc. *Consumer*
clients subscribe to data on one or more topics.

On the brokers messages received from producers are written to disk in one or more log files per
topic. These log files are called *partitions*. The position of each message in the partition is
called the *offset*. Messages older than a configurable time are deleted, so the first available
offset in each partition may not be 0. Consumers have control over what offset they read from, so
they can start consuming the stream from the oldest available message, the latest message, or at
from the next available message after a specified time.

Many libraries are available for the *Client API*: implementations of a producer and consumer. In
``scippneutron`` we use ``confluent-kafka-python`` which is based on a high performance C implementation
``librdkafka``.

Note, messages published to Kafka are often called *events*, but this terminology is avoided in
``scippneutron`` as we usually use this term for "neutron detection events" specifically.

Topic - a named data stream
- Partition - a log file, one or more per topic
- Offset - the position of a message in a partition
- Broker - Kafka server, usually deployed as a cluster
- Consumer - a client application which subscribes to data on Kafka
- Producer - a client application which publishes data to Kafka


FlatBuffers
-----------

Kafka does not dictate how data are serialised to be transmitted in a message over the network.
The ESS streaming system uses `FlatBuffers <https://google.github.io/flatbuffers/>`_ for serialisation.
Data to be serialised with FlatBuffers are described by an IDL schema, the FlatBuffer compiler ``flatc``
can be used to generate code from the schema which provides a builder class with which to construct
the serialised buffer, as well as methods to extract data from the buffer.

Each type of data, for example neutron detection events, sample environment measurement,
experiment run start event, etc has an associated FlatBuffer schema. These are stored in a repository
`<https://github.com/ess-dmsc/streaming-data-types/>`_. The ESS has also developed a Python library
to provide a convenient `serialise` and `deserialise` method for each schema
`<https://github.com/ess-dmsc/python-streaming-data-types/>`_, this is available as a conda package
`<https://anaconda.org/ess-dmsc/ess-streaming-data-types>`_.

Each schema file defines a ``file_identifier``, which comprises 4 characters. These are the first 4
bytes in the serialised buffer. It is ESS convention to also use these 4 characters in the schema
filename and the module name in the ``streaming-data-types`` python library. If breaking changes are
made to a schema, such as changing field names or removing fields, then a new ``file_identifier`` is
defined.

It may be worth noting that we have found extracting data from serialised FlatBuffers in Python
to be efficient, but serialising can be much more efficient in C++, particularly if serialising
very many small buffers.


Architecture
------------

Why threading.Thread, multiprocessing.Process and asyncio?
Why each consumer responsible for a single TopicPartition and not multiple partitions or even multiple topics?

The interface to streamed data is data_stream(), which is an asynchronous generator. It is run like this
```python
async for data in data_stream(*args):
    ...
```


data_stream is given a Kafka topic on which to find "run start" messages. It looks for the last available run start message. The message contains some data known at the start of an experiment run, for example instrument geometry. These data are yielded from the generator as the first chunk of streamed data, as a data array. The run start message also contains details of all the other data sources important to the experiment and where to find their data on Kafka. This information is passed to the data_consumption_manager() which is started in a separate multiprocessing.Process.

data_consumption_manager() creates a StreamedDataBuffer which comprises buffers for data from each data source known about from the run start message. data_consumption_manager() also creates a KafkaConsumer for each partition in each Kafka topic associated with the data sources. It starts a threading.Thread in each KafkaConsumer which polls the consumer's internal queue, if any data have been collected by the consumer they are passed to the buffer via a callback function. It also starts a threading.Thread in the buffer which periodically puts all all data collected in the buffer as a single DataArray into a multiprocessing.Queue for the data_stream generator to yield. The buffer is responsible for checking the flatbuffer id of each message it receives from the consumers, deserialising the message, checking the source name matches a data source named in the run start message, and if so adding the data to the buffer. If a single message exceeds the buffer size a warning is issued to the user and the data is skipped. If multiple messages arrive which collectively exceed the buffer size before the buffer has put its data on the queue and reset, then the buffer puts its data on the queue early.


Unit Testing
------------

[modified diagram to show approach without normal dep inject]

Manual Testing
--------------

Testing ``scippneutron``'s interface to the streaming system requires running a Kafka server and
populating it with neutron data. The most convenient way to do this on a developer
machine is to use docker containers.

Setup
~~~~~

`Install Docker Engine <https://docs.docker.com/get-docker/>`_ on your system.
If on Linux, do not forget to add your user to the "docker" group,
`see documentation <https://docs.docker.com/engine/install/linux-postinstall/>`_.

Install docker-compose

    .. code-block:: sh

        conda install -c docker-compose

Run Containers
~~~~~~~~~~~~~~

To start up Kafka and the `NeXus Streamer <https://github.com/ess-dmsc/nexus-streamer>`_
to populate it with data from the SANS2D instrument at ISIS Neutron Source,
navigate to the ``docs/developer/data_stream`` directory and run

    .. code-block:: sh

        docker-compose up

``Ctrl+C`` cleanly stops the running containers when you are done.

If you are in doubt whether the containers are working you may want
to use the `kafkacow command line tool <https://github.com/ess-dmsc/kafkacow>`_ to query the Kafka server, see
`installation instructions <https://github.com/ess-dmsc/kafkacow#install>`_.

For example, to check data topics on the Kafka server

    .. code-block:: sh

        kafkacow -L -b localhost

you should see output like this

    .. code-block:: sh

        1 brokers:
           broker 1 at 0.0.0.0:9092

        10 topics:
           "SANS2D_sampleEnv" with 1 partitions:
                partition   0  |  Low offset:      0  |  High offset: 295782 |  leader:  1 |  replicas: 1,  |  isrs: 1,
           "SANS2D_events" with 1 partitions:
                partition   0  |  Low offset:      0  |  High offset:   6271 |  leader:  1 |  replicas: 1,  |  isrs: 1,
        ...

and you can view the event data with

    .. code-block:: sh

        kafkacow -C -b localhost -t SANS2D_events

output:

    .. code-block:: sh

        Mon 12-Apr-2021 13:30:56.903  ||  2021-04-12T13:30:56.903

        Timestamp: 1618234256903 || PartitionID:     0 || Offset:    1150 || File Identifier: ev42 ||
        {
          detector_id: [     61985     62379     62126     ... truncated 756 elements ...     120485   ]
          facility_specific_data: {
            proton_charge: 0.001098
            run_state: RUNNING
          }
          facility_specific_data_type: ISISData
          message_id: 1149
          pulse_time: 1618234368838996887
          source_name: NeXus-Streamer
          time_of_flight: [     12379936     14495801     14658190     ... truncated 756 elements ...     36832880   ]
        }
        ...

Try using ``scippneutron.data_stream``, for example

    .. code-block:: python

        import asyncio
        import scippneutron as scn

        async def my_stream_func():
            async for data in scn.data_stream('localhost:9092', ['SANS2D_events']):
                print(data)
                break  # just print the first batch of data we receive

        streaming_task = asyncio.create_task(my_stream_func())

Note that the producer container (NeXus-Streamer) must be currently running for you to receive data.
By default the producer container stops running after publishing the contents of the SANS2D file it contains.
If you want it to keep repeating publishing data until you terminate docker-compose then set
``single-run`` to ``false`` in ``docs/developer/data_stream/nexus_streamer_config.ini``, but note that this
will use more and more disk space until you terminate docker-compose.

Clean Up
~~~~~~~~

After you are done testing you can clean up the containers and free up used disk space by running

    .. code-block:: sh

        docker rm -v data_stream_producer_1
        docker rm -v data_stream_kafka_1
