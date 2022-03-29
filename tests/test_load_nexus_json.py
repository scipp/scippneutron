import scipp as sc
import numpy as np
from .nexus_helpers import NexusBuilder, Source, Stream
import pytest
from scippneutron.file_loading.load_nexus import _load_nexus_json
"""
Many tests for load_nexus_json() are in test_load_nexus
as they are parameterised to run the same checks against
load_nexus() and load_nexus_json().
Tests in this module are for features specific to the json
representation, for example the "stream" objects which link
the data in the json template to data available to
be streamed from Kafka during an experiment.
"""


@pytest.mark.skip("TODO Stream handling with log not implemented")
def test_stream_object_as_transformation_results_in_warning():
    builder = NexusBuilder()
    builder.add_component(Source("source"))
    stream_path = "/entry/streamed_nxlog_transform"
    builder.add_stream(Stream(stream_path))
    builder.add_dataset_at_path("/entry/source/depends_on", stream_path, {})

    with pytest.warns(UserWarning):
        loaded_data, _ = _load_nexus_json(builder.json_string)

    # A 0 distance translation is used in place of the streamed transformation
    default = [0, 0, 0]
    assert np.allclose(loaded_data["source_position"].values, default)
    assert loaded_data["source_position"].unit == sc.Unit("m")
