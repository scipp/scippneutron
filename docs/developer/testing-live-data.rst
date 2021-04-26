Testing "live data"
===================

The ESS data streaming system is based on the Apache Kafka data streaming platform.
Testing scippneutron's interface to this system requires running a Kafka server and
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
navigate to the ``docs/developer/live_data`` directory and run

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

        async def my_stream_func():
            async for data in scn.data_stream('localhost:9092', ['SANS2D_events']):
                print(data)
                break  # just print the first batch of data we receive

        streaming_task = asyncio.create_task(my_stream_func())

Note that the producer container (NeXus-Streamer) must be currently running for you to receive data.
By default the producer container stops running after publishing the contents of the SANS2D file it contains.
If you want it to keep repeating publishing data until you terminate docker-compose then set
``single-run`` to ``false`` in ``docs/developer/live_data/nexus_streamer_config.ini``, but note that this
will use more and more disk space until you terminate docker-compose.

Clean Up
~~~~~~~~

After you are done testing you can clean up the containers and free up used disk space by running

    .. code-block:: sh

        docker rm -v live_data_producer_1
        docker rm -v live_data_kafka_1
