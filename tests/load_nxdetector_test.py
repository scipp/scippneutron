import scipp as sc
import scippnexus as nexus
import scippneutron as scn


def test_can_load_nxdetector_from_bigfake():
    with nexus.File(scn.data.bigfake()) as f:
        da = f['entry/instrument/detector_1'][...]
        assert da.sizes == {'dim_0': 300, 'dim_1': 300}


def test_can_load_nxdetector_from_PG3():
    import scippneutron as scn
    with nexus.File(scn.data.get_path('PG3_4844_event.nxs')) as f:
        det = f['entry/instrument/bank24']
        da = det[...]
        assert da.sizes == {'x_pixel_offset': 154, 'y_pixel_offset': 7}
        assert 'detector_number' not in da.coords
        assert da.coords['pixel_id'].sizes == da.sizes
        assert da.coords['distance'].sizes == da.sizes
        assert da.coords['polar_angle'].sizes == da.sizes
        assert da.coords['azimuthal_angle'].sizes == da.sizes
        assert da.coords['x_pixel_offset'].sizes == {'x_pixel_offset': 154}
        assert da.coords['y_pixel_offset'].sizes == {'y_pixel_offset': 7}
        # local_name is an example of a dataset with shape=[1] that is treated as scalar
        assert da.coords['local_name'].sizes == {}
        # Extra scalar fields not in underlying NXevent_data
        del da.coords['local_name']
        del da.coords['total_counts']
        assert sc.identical(da.sum(), det.events[()].sum())  # no event lost in binning
