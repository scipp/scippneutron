# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
# @author Simon Heybrock
import pytest
import scipp as sc
from scipp.testing import assert_identical

from scippneutron.tof import chopper_cascade


def test_subframe_init_raises_if_time_and_wavelength_have_different_dims() -> None:
    time = sc.array(dims=['vertex'], values=[0.0, 1.0, 1.0, 0.0], unit='s')
    wavelength = sc.array(dims=['vertex'], values=[1.0, 1.0, 2.0, 2.0], unit='angstrom')
    with pytest.raises(sc.DimensionError):
        chopper_cascade.Subframe(time=time, wavelength=wavelength[0:3])
    with pytest.raises(sc.DimensionError):
        chopper_cascade.Subframe(
            time=time, wavelength=wavelength.rename_dims({'vertex': 'point'})
        )


def test_subframe_init_raises_if_time_cannot_be_converted_to_seconds() -> None:
    time = sc.array(dims=['vertex'], values=[0.0, 1.0, 1.0, 0.0], unit='m')
    wavelength = sc.array(dims=['vertex'], values=[1.0, 1.0, 2.0, 2.0], unit='angstrom')
    with pytest.raises(sc.UnitError):
        chopper_cascade.Subframe(time=time, wavelength=wavelength)


def test_subframe_init_raises_if_wavelength_cannot_be_converted_to_angstrom() -> None:
    time = sc.array(dims=['vertex'], values=[0.0, 1.0, 1.0, 0.0], unit='s')
    wavelength = sc.array(dims=['vertex'], values=[1.0, 1.0, 2.0, 2.0], unit='s')
    with pytest.raises(sc.UnitError):
        chopper_cascade.Subframe(time=time, wavelength=wavelength)


def test_subframe_is_regular() -> None:
    # Triangle with last point after base
    time = sc.array(dims=['vertex'], values=[0.0, 2.0, 3.0], unit='s')
    wavelength = sc.array(dims=['vertex'], values=[1.0, 1.0, 2.0], unit='angstrom')
    subframe = chopper_cascade.Subframe(time=time, wavelength=wavelength)
    assert subframe.is_regular()
    # Triangle with last point inside base
    time = sc.array(dims=['vertex'], values=[0.0, 2.0, 1.0], unit='s')
    subframe = chopper_cascade.Subframe(time=time, wavelength=wavelength)
    assert not subframe.is_regular()
    # Triangle standing on its tip, to also test the min-wavelength
    time = sc.array(dims=['vertex'], values=[1.0, 0.0, 3.0], unit='s')
    wavelength = sc.array(dims=['vertex'], values=[2.0, 1.0, 2.0], unit='angstrom')
    subframe = chopper_cascade.Subframe(time=time, wavelength=wavelength)
    assert subframe.is_regular()
    time = sc.array(dims=['vertex'], values=[1.0, 2.0, 3.0], unit='s')
    subframe = chopper_cascade.Subframe(time=time, wavelength=wavelength)
    assert not subframe.is_regular()


def test_subframe_propagate_by() -> None:
    time = sc.array(dims=['vertex'], values=[0.0, 1.0, 1.0, 0.0], unit='s')
    wavelength = sc.array(dims=['vertex'], values=[1.0, 1.0, 2.0, 2.0], unit='angstrom')
    subframe = chopper_cascade.Subframe(time=time, wavelength=wavelength)
    propagated = subframe.propagate_by(sc.scalar(1.0, unit='m'))
    assert_identical(propagated.wavelength, subframe.wavelength)
    assert (propagated.time > subframe.time).all()
    # Started at same time, but different wavelength
    assert propagated.time[2] > propagated.time[1]
    # Difference should be proportional to wavelength
    dt = propagated.time - subframe.time
    rtol = sc.scalar(1e-12, unit='')
    assert sc.isclose(dt[0], dt[1], atol=sc.scalar(0.0, unit='s'), rtol=rtol)
    assert sc.isclose(dt[2], dt[3], atol=sc.scalar(0.0, unit='s'), rtol=rtol)
    assert sc.allclose(
        (dt[2:3] / dt[1]), sc.scalar(2.0), atol=sc.scalar(0.0), rtol=rtol
    )


def test_subframe_time_is_converted_to_seconds() -> None:
    time = sc.array(dims=['vertex'], values=[0.0, 1.0, 3.0, 2.0], unit='ms')
    wavelength = sc.array(dims=['vertex'], values=[1.0, 1.1, 2.1, 2.0], unit='nm')
    subframe = chopper_cascade.Subframe(time=time, wavelength=wavelength)
    assert_identical(
        subframe.time,
        sc.array(dims=['vertex'], values=[0.0, 0.001, 0.003, 0.002], unit='s'),
    )


def test_subframe_wavelength_is_converted_to_angstrom() -> None:
    time = sc.array(dims=['vertex'], values=[0.0, 1.0, 3.0, 2.0], unit='ms')
    wavelength = sc.array(dims=['vertex'], values=[1.0, 1.1, 2.1, 2.0], unit='nm')
    subframe = chopper_cascade.Subframe(time=time, wavelength=wavelength)
    assert_identical(
        subframe.wavelength,
        sc.array(dims=['vertex'], values=[10.0, 11.0, 21.0, 20.0], unit='angstrom'),
    )


def test_subframe_start_end_properties() -> None:
    time = sc.array(dims=['vertex'], values=[0.0, 1.0, 3.0, 2.0], unit='ms')
    wavelength = sc.array(dims=['vertex'], values=[1.0, 1.1, 2.1, 2.0], unit='nm')
    subframe = chopper_cascade.Subframe(time=time, wavelength=wavelength)
    assert_identical(subframe.start_time, sc.scalar(0.0, unit='s'))
    assert_identical(subframe.end_time, sc.scalar(0.003, unit='s'))
    assert_identical(subframe.start_wavelength, sc.scalar(10.0, unit='angstrom'))
    assert_identical(subframe.end_wavelength, sc.scalar(21.0, unit='angstrom'))


@pytest.fixture()
def frame() -> chopper_cascade.Frame:
    time = sc.array(dims=['vertex'], values=[0.0, 1.0, 3.0, 2.0], unit='ms')
    wavelength = sc.array(dims=['vertex'], values=[1.0, 1.1, 2.1, 2.0], unit='nm')
    subframe1 = chopper_cascade.Subframe(time=time, wavelength=wavelength)
    subframe2 = chopper_cascade.Subframe(time=time * 2.0, wavelength=wavelength * 3.0)
    return chopper_cascade.Frame(
        distance=sc.scalar(1.0, unit='m'), subframes=[subframe1, subframe2]
    )


def test_frame_propagate_to(frame: chopper_cascade.Frame) -> None:
    # Propagate to same distance as frame distance, i.e., no propagation
    distance = sc.scalar(1.0, unit='m')
    propagated = frame.propagate_to(distance)
    assert propagated.distance == distance
    assert propagated == frame
    distance = sc.scalar(2.0, unit='m')
    propagated = frame.propagate_to(distance)
    assert propagated.distance == distance
    assert (propagated.subframes[0].time > frame.subframes[0].time).all()
    assert (propagated.subframes[1].time > frame.subframes[1].time).all()
    # Wavelengths are unaffected by propagation
    assert_identical(propagated.subframes[0].wavelength, frame.subframes[0].wavelength)
    assert_identical(propagated.subframes[1].wavelength, frame.subframes[1].wavelength)


def test_frame_propagate_to_works_with_empty_frame() -> None:
    frame = chopper_cascade.Frame(distance=sc.scalar(1.0, unit='m'), subframes=[])
    distance = sc.scalar(2.0, unit='m')
    propagated = frame.propagate_to(distance)
    assert_identical(propagated.distance, distance)
    assert propagated.subframes == []


def test_frame_chop_with_no_effect(frame: chopper_cascade.Frame) -> None:
    # Chopper is open for a long time, it should have no effect
    chopper = chopper_cascade.Chopper(
        distance=frame.distance,
        time_open=sc.array(dims=['slit'], values=[0.0], unit='s'),
        time_close=sc.array(dims=['slit'], values=[1.0], unit='s'),
    )
    assert frame.chop(chopper) == frame


def test_frame_chop_returns_empty_frame_if_chopper_is_closed(
    frame: chopper_cascade.Frame,
) -> None:
    # Chopper is closed during entire arrival time of neutrons
    chopper = chopper_cascade.Chopper(
        distance=frame.distance * 1.0,
        time_open=sc.array(dims=['slit'], values=[1.0], unit='s'),
        time_close=sc.array(dims=['slit'], values=[2.0], unit='s'),
    )
    chopped = frame.chop(chopper)
    assert chopped.distance == chopper.distance
    assert len(chopped.subframes) == 0


def test_frame_chop_works_with_empty_frame() -> None:
    frame = chopper_cascade.Frame(distance=sc.scalar(1.0, unit='m'), subframes=[])
    chopper = chopper_cascade.Chopper(
        distance=frame.distance * 1.5,
        time_open=sc.array(dims=['slit'], values=[0.0], unit='s'),
        time_close=sc.array(dims=['slit'], values=[1.0], unit='s'),
    )
    chopped = frame.chop(chopper)
    assert_identical(chopped.distance, chopper.distance)
    assert chopped.subframes == []


def test_frame_chop_returns_frame_with_distance_set_to_chopper_distance(
    frame: chopper_cascade.Frame,
) -> None:
    chopper = chopper_cascade.Chopper(
        distance=sc.scalar(2.0, unit='m'),
        time_open=sc.array(dims=['slit'], values=[0.0], unit='s'),
        time_close=sc.array(dims=['slit'], values=[1.0], unit='s'),
    )
    chopped = frame.chop(chopper)
    assert_identical(chopped.distance, chopper.distance)


def test_frame_chop_raises_if_chopper_distance_is_less_than_frame_distance(
    frame: chopper_cascade.Frame,
) -> None:
    chopper = chopper_cascade.Chopper(
        distance=sc.scalar(0.9, unit='m'),
        time_open=sc.array(dims=['slit'], values=[0.0], unit='s'),
        time_close=sc.array(dims=['slit'], values=[1.0], unit='s'),
    )
    with pytest.raises(ValueError, match='smaller than frame distance'):
        frame.chop(chopper)


def test_frame_chop_trims_subframes_using_chopper_open() -> None:
    time = sc.array(dims=['vertex'], values=[0.0, 2.0, 4.0, 2.0], unit='s')
    wavelength = sc.array(
        dims=['vertex'], values=[10.0, 10.0, 20.0, 20.0], unit='angstrom'
    )
    subframe = chopper_cascade.Subframe(time=time, wavelength=wavelength)
    frame = chopper_cascade.Frame(
        distance=sc.scalar(1.0, unit='m'), subframes=[subframe, subframe]
    )
    chopper = chopper_cascade.Chopper(
        distance=frame.distance,
        time_open=sc.array(dims=['slit'], values=[1.0], unit='s'),
        time_close=sc.array(dims=['slit'], values=[6.0], unit='s'),
    )
    chopped = frame.chop(chopper)
    expected_time = sc.array(
        dims=['vertex'], values=[1.0, 2.0, 4.0, 2.0, 1.0], unit='s'
    )
    # The 15 is from the cut of the line from (2, 20) to (0, 10) in at t=1
    expected_wavelength = sc.array(
        dims=['vertex'], values=[10.0, 10.0, 20.0, 20.0, 15.0], unit='angstrom'
    )
    expected_subframe = chopper_cascade.Subframe(
        time=expected_time, wavelength=expected_wavelength
    )
    expected = chopper_cascade.Frame(
        distance=frame.distance, subframes=[expected_subframe, expected_subframe]
    )
    assert chopped == expected


def test_frame_chop_trims_subframes_using_chopper_close() -> None:
    time = sc.array(dims=['vertex'], values=[0.0, 2.0, 4.0, 2.0], unit='s')
    wavelength = sc.array(
        dims=['vertex'], values=[10.0, 10.0, 20.0, 20.0], unit='angstrom'
    )
    subframe = chopper_cascade.Subframe(time=time, wavelength=wavelength)
    frame = chopper_cascade.Frame(
        distance=sc.scalar(1.0, unit='m'), subframes=[subframe, subframe]
    )
    chopper = chopper_cascade.Chopper(
        distance=frame.distance,
        time_open=sc.array(dims=['slit'], values=[0.0], unit='s'),
        time_close=sc.array(dims=['slit'], values=[3.0], unit='s'),
    )
    chopped = frame.chop(chopper)
    expected_time = sc.array(
        dims=['vertex'], values=[0.0, 2.0, 3.0, 3.0, 2.0], unit='s'
    )
    # The 15 is from the cut of the line from (2, 10) to (4, 20) in at t=1
    expected_wavelength = sc.array(
        dims=['vertex'], values=[10.0, 10.0, 15.0, 20.0, 20.0], unit='angstrom'
    )
    expected_subframe = chopper_cascade.Subframe(
        time=expected_time, wavelength=expected_wavelength
    )
    expected = chopper_cascade.Frame(
        distance=frame.distance, subframes=[expected_subframe, expected_subframe]
    )
    assert chopped == expected


def test_frame_chop_with_multi_slit_chopper_splits_subframes() -> None:
    time = sc.array(dims=['vertex'], values=[0.0, 2.0, 4.0, 2.0], unit='s')
    wavelength = sc.array(
        dims=['vertex'], values=[10.0, 10.0, 20.0, 20.0], unit='angstrom'
    )
    subframe = chopper_cascade.Subframe(time=time, wavelength=wavelength)
    frame = chopper_cascade.Frame(
        distance=sc.scalar(1.0, unit='m'), subframes=[subframe, subframe]
    )
    chopper = chopper_cascade.Chopper(
        distance=frame.distance,
        time_open=sc.array(dims=['slit'], values=[0.0, 3.0], unit='s'),
        time_close=sc.array(dims=['slit'], values=[1.0, 4.0], unit='s'),
    )
    time_left = sc.array(dims=['vertex'], values=[0.0, 1.0, 1.0], unit='s')
    time_right = sc.array(dims=['vertex'], values=[3.0, 4.0, 3.0], unit='s')
    wavelength_left = sc.array(
        dims=['vertex'], values=[10.0, 10.0, 15.0], unit='angstrom'
    )
    wavelength_right = sc.array(
        dims=['vertex'], values=[15.0, 20.0, 20.0], unit='angstrom'
    )
    expected_subframe_left = chopper_cascade.Subframe(
        time=time_left, wavelength=wavelength_left
    )
    expected_subframe_right = chopper_cascade.Subframe(
        time=time_right, wavelength=wavelength_right
    )
    expected = chopper_cascade.Frame(
        distance=frame.distance,
        subframes=[
            expected_subframe_left,
            expected_subframe_right,
            expected_subframe_left,
            expected_subframe_right,
        ],
    )
    assert frame.chop(chopper) == expected


def test_frame_chop_with_multi_slit_chopper_blind_slit() -> None:
    time = sc.array(dims=['vertex'], values=[0.0, 2.0, 4.0, 2.0], unit='s')
    wavelength = sc.array(
        dims=['vertex'], values=[10.0, 10.0, 20.0, 20.0], unit='angstrom'
    )
    subframe = chopper_cascade.Subframe(time=time, wavelength=wavelength)
    frame = chopper_cascade.Frame(
        distance=sc.scalar(1.0, unit='m'), subframes=[subframe, subframe]
    )
    # The second slit opens after all neutrons have passed, it will have no effect.
    chopper = chopper_cascade.Chopper(
        distance=frame.distance,
        time_open=sc.array(dims=['slit'], values=[0.0, 5.0], unit='s'),
        time_close=sc.array(dims=['slit'], values=[3.0, 6.0], unit='s'),
    )
    chopped = frame.chop(chopper)
    equivalent_single_slit_chopper = chopper_cascade.Chopper(
        distance=frame.distance,
        time_open=sc.array(dims=['slit'], values=[0.0], unit='s'),
        time_close=sc.array(dims=['slit'], values=[3.0], unit='s'),
    )
    expected = frame.chop(equivalent_single_slit_chopper)
    assert chopped == expected


def test_frame_bounds_gives_global_min_and_max() -> None:
    time = sc.array(dims=['vertex'], values=[0.0, 2.0, 4.0, 2.0], unit='s')
    wavelength = sc.array(
        dims=['vertex'], values=[10.0, 10.0, 20.0, 20.0], unit='angstrom'
    )
    subframe1 = chopper_cascade.Subframe(time=time, wavelength=wavelength)
    subframe2 = chopper_cascade.Subframe(time=time * 2.0, wavelength=wavelength * 3.0)
    frame = chopper_cascade.Frame(
        distance=sc.scalar(1.0, unit='m'), subframes=[subframe1, subframe2]
    )
    bounds = frame.bounds()
    assert_identical(
        bounds['time'], sc.array(dims=['bound'], values=[0.0, 8.0], unit='s')
    )
    assert_identical(
        bounds['wavelength'],
        sc.array(dims=['bound'], values=[10.0, 60.0], unit='angstrom'),
    )


def test_frame_sequence_sets_up_rectangular_subframe() -> None:
    frames = chopper_cascade.FrameSequence.from_source_pulse(
        time_min=sc.scalar(0.0, unit='s'),
        time_max=sc.scalar(1.0, unit='s'),
        wavelength_min=sc.scalar(1.0, unit='angstrom'),
        wavelength_max=sc.scalar(2.0, unit='angstrom'),
    )
    expected = chopper_cascade.Frame(
        distance=sc.scalar(0.0, unit='m'),
        subframes=[
            chopper_cascade.Subframe(
                time=sc.array(dims=['vertex'], values=[0.0, 1.0, 1.0, 0.0], unit='s'),
                wavelength=sc.array(
                    dims=['vertex'], values=[1.0, 1.0, 2.0, 2.0], unit='angstrom'
                ),
            )
        ],
    )
    assert len(frames) == 1
    assert frames[0] == expected


@pytest.fixture()
def source_frame_sequence() -> chopper_cascade.FrameSequence:
    return chopper_cascade.FrameSequence.from_source_pulse(
        time_min=sc.scalar(0.0, unit='ms'),
        time_max=sc.scalar(1.0, unit='ms'),
        wavelength_min=sc.scalar(1.0, unit='angstrom'),
        wavelength_max=sc.scalar(10.0, unit='angstrom'),
    )


def test_frame_sequence_propagate_to_returns_new_sequence_with_added_propagated_frame(
    source_frame_sequence: chopper_cascade.FrameSequence,
) -> None:
    frames = source_frame_sequence
    distance = sc.scalar(1.5, unit='m')
    result = frames.propagate_to(distance)
    assert len(frames) == 1
    assert len(result) == 2
    assert result[1] == frames[0].propagate_to(distance)
    result2 = result.propagate_to(distance * 2)
    assert len(result2) == 3
    assert result2[2] == frames[0].propagate_to(distance * 2)
    assert result2[2] == result[1].propagate_to(distance * 2)


def test_frame_sequence_chop_returns_new_sequence_with_added_chopped_frames(
    source_frame_sequence: chopper_cascade.FrameSequence,
) -> None:
    frames = source_frame_sequence
    chopper1 = chopper_cascade.Chopper(
        distance=sc.scalar(1.5, unit='m'),
        time_open=sc.array(dims=['slit'], values=[0.0], unit='s'),
        time_close=sc.array(dims=['slit'], values=[0.001], unit='s'),
    )
    chopper2 = chopper_cascade.Chopper(
        distance=sc.scalar(2.5, unit='m'),
        time_open=sc.array(dims=['slit'], values=[0.001], unit='s'),
        time_close=sc.array(dims=['slit'], values=[0.003], unit='s'),
    )
    result = frames.chop([chopper1, chopper2])
    assert len(frames) == 1
    assert len(result) == 3  # source + 2 choppers
    assert len(result[2].subframes) == 1  # something makes it through
    assert result[0] == frames[0]
    assert result[0].chop(chopper1) == result[1]
    assert result[1].chop(chopper2) == result[2]


def test_frame_sequence_chop_applies_choppers_ordered_by_distance(
    source_frame_sequence: chopper_cascade.FrameSequence,
) -> None:
    frames = source_frame_sequence
    chopper1 = chopper_cascade.Chopper(
        distance=sc.scalar(1.5, unit='m'),
        time_open=sc.array(dims=['slit'], values=[0.0], unit='s'),
        time_close=sc.array(dims=['slit'], values=[0.001], unit='s'),
    )
    chopper2 = chopper_cascade.Chopper(
        distance=sc.scalar(2.5, unit='m'),
        time_open=sc.array(dims=['slit'], values=[0.001], unit='s'),
        time_close=sc.array(dims=['slit'], values=[0.003], unit='s'),
    )
    result12 = frames.chop([chopper1, chopper2])
    result21 = frames.chop([chopper2, chopper1])
    assert result12 == result21
    assert result12[2] == result21[2]


def test_frame_sequence_getitem_by_distance_selects_correct_chopper(
    source_frame_sequence: chopper_cascade.FrameSequence,
) -> None:
    frames = source_frame_sequence
    chopper1 = chopper_cascade.Chopper(
        distance=sc.scalar(1.5, unit='m'),
        time_open=sc.array(dims=['slit'], values=[0.0], unit='s'),
        time_close=sc.array(dims=['slit'], values=[0.001], unit='s'),
    )
    chopper2 = chopper_cascade.Chopper(
        distance=sc.scalar(2.5, unit='m'),
        time_open=sc.array(dims=['slit'], values=[0.001], unit='s'),
        time_close=sc.array(dims=['slit'], values=[0.003], unit='s'),
    )
    frames = frames.chop([chopper1, chopper2])
    distance = sc.scalar(1.0, unit='m')
    result = frames[distance]
    assert_identical(result.bounds(), frames[0].propagate_to(distance).bounds())
    distance = sc.scalar(2.0, unit='m')
    result = frames[distance]
    assert_identical(result.bounds(), frames[1].propagate_to(distance).bounds())
    distance = sc.scalar(2.4, unit='m')
    result = frames[distance]
    assert_identical(result.bounds(), frames[1].propagate_to(distance).bounds())
    distance = sc.scalar(3.0, unit='m')
    result = frames[distance]
    assert_identical(result.bounds(), frames[2].propagate_to(distance).bounds())
