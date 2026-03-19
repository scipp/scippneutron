#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import scipp as sc
import scipy as sp
from scipy.special import lambertw


def cosine_wave(time, A, T, phi, Amean):
    return A * sc.cos(sc.scalar(2 * np.pi, unit='rad') * time / T + phi) + Amean


def sine_wave(time, **par):
    A = par['A']
    T = par['T']
    Toff = par['Toff']
    Aoff = par['Aoff']

    return A * sc.sin(sc.scalar(2 * np.pi, unit='rad') * (time - Toff) / T) + Aoff


def ss_fft(signal):
    diff = signal.coords['time'][1:] - signal.coords['time'][:-1]
    mean_diff = sc.mean(diff)
    dt = mean_diff.values / 1e9

    Fs = 1 / dt
    L = len(signal)

    Y = np.fft.rfft(signal.values, n=L)
    P = np.abs(Y) / L
    P[1:-1] = 2 * P[1:-1]
    ph = np.angle(Y)

    f = Fs * np.arange(0, L // 2 + 1) / L

    return P, ph, f


def rheo_waveform_params(rheo_binned, method):
    if method == 'curve_fit':
        rheo_for_fit = rheo_binned.copy()
        # set time origin to 0
        rheo_for_fit.coords['time'] = (
            rheo_for_fit.coords['time'] - rheo_for_fit.coords['time'][0]
        )
        # rheo_for_fit.coords['time'] = rheo_for_fit.coords['time'].to(unit='s')

        # Get a guess for period using DFT
        P, ph, f = ss_fft(rheo_for_fit)
        maxindex = np.argmax(P[1:]) + 1
        T_0 = sc.scalar(1e9 / f[maxindex], unit='ns')
        A_0 = (
            (sc.max(rheo_for_fit) - sc.min(rheo_for_fit)) / sc.scalar(2, unit='')
        ).data
        Aoff_0 = sc.mean(rheo_for_fit).data
        mask_first_p = rheo_for_fit.coords['time'] < T_0
        Toff_0 = (
            rheo_for_fit.coords['time'][mask_first_p][
                (rheo_for_fit[mask_first_p] == sc.max(rheo_for_fit[mask_first_p])).data
            ][0]
            - T_0 / 4
        )

        print({'A': A_0, 'T': T_0, 'Toff': Toff_0, 'Aoff': Aoff_0})
        par, _ = sc.curve_fit(
            ['time'],
            sine_wave,
            rheo_for_fit,
            p0={'A': A_0, 'T': T_0, 'Toff': Toff_0, 'Aoff': Aoff_0},
        )
        par['T'] = par['T'].to(unit='ns')
        par['Toff'] = par['Toff'].to(unit='ns')
        if par['A'].values < 0:
            par['A'] = -par['A']
            par['Toff'] = par['Toff'] + par['T'] / 2

    elif method == 'fft':
        P, ph, f = ss_fft(rheo_binned)
        maxindex = np.argmax(P[1:]) + 1

        A = P[maxindex]
        T = 1 / f[maxindex]
        Aoff = sc.mean(rheo_binned).values
        phi = ph[maxindex]
        Toff = -T * (phi + np.pi / 2) / (2 * np.pi)

        par = sc.DataGroup(
            A=sc.DataArray(data=sc.Variable(dims=[], values=A, unit='')),
            T=sc.DataArray(data=sc.Variable(dims=[], values=T * 1e9, unit='ns')),
            Toff=sc.DataArray(data=sc.Variable(dims=[], values=Toff * 1e9, unit='ns')),
            Aoff=sc.DataArray(data=sc.Variable(dims=[], values=Aoff, unit='')),
        )

    else:
        print('Method not supported')
        par = sc.DataGroup(
            A=sc.DataArray(data=sc.Variable(dims=[], values=0, unit='')),
            T=sc.DataArray(data=sc.Variable(dims=[], values=0, unit='ns')),
            Toff=sc.DataArray(data=sc.Variable(dims=[], values=0, unit='ns')),
            Aoff=sc.DataArray(data=sc.Variable(dims=[], values=0, unit='')),
        )

    return par


def period_edges(signal, plot_flag, value=None):
    if value == None:
        signal_zeroed = signal - signal[0]
    else:
        signal_zeroed = signal - value

    if signal_zeroed.values[0] > signal_zeroed.values[1]:
        # Falling edges
        period_edges = np.where(np.diff(np.sign(signal_zeroed.values)) < 0)[0]
    else:
        # Rising edges
        period_edges = np.where(np.diff(np.sign(signal_zeroed.values)) > 0)[0]

    if plot_flag:
        plt.figure()
        plt.plot(
            np.diff(period_edges),
            label='',
            linestyle='',
            marker='o',
            markerfacecolor='none',
        )
        plt.xlabel('Samples')
        plt.ylabel('diff(Edges)')
        plt.legend()
        plt.show()

    return period_edges


def ppT_autocorrelation(signal, plot_flag=None):
    dt = np.mean(np.diff(signal.coords['time'].values)) / 1e9
    signal = signal.values
    N = len(signal)
    # zero-padding to the next power of 2
    Npad = 2 ** int(np.ceil(np.log2(2 * N)))
    signal_padded = np.zeros(Npad)
    signal_padded[:N] = signal
    X = np.fft.rfft(signal_padded)
    P = X * np.conj(X)
    R = np.fft.irfft(P)
    R = np.real(R[:N])
    # R_full = np.correlate(signal, signal, mode='full')
    # N = len(signal)
    # R = R_full[N - 1:]
    # R = R / R[0]

    locs, _ = sp.signal.find_peaks(R[1:])
    locs = locs + 1

    if len(locs) == 0:
        ppT = np.nan
        print("Warning: Peak not found")
    else:
        ppT = locs[0]

    # Parabolic interpolation
    if ppT > 0 and ppT < len(R) - 1:
        y1, y2, y3 = R[ppT - 1], R[ppT], R[ppT + 1]
        delta = 0.5 * (y1 - y3) / (y1 - 2 * y2 + y3)
        ppT = ppT + delta

    freq = 1 / ppT / dt

    if plot_flag:
        R = R / R[0]
        plt.figure()
        plt.plot(R, label='Autocorrelation')
        plt.plot(
            locs,
            R[locs],
            marker='o',
            linestyle='None',
            markeredgecolor='r',
            markerfacecolor='none',
            label='Peaks',
        )
        plt.xlabel('Samples')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.show()

    return freq, ppT


def ppT_fft(signal, plot_flag=None):
    P, ph, f = ss_fft(signal)
    max_index = np.argmax(P[1:]) + 1

    freq = f[max_index]
    dt = np.mean(np.diff(signal.coords['time'].values)) / 1e9
    ppT = 1 / freq / dt

    if plot_flag:
        plt.figure()
        plt.plot(f, P)
        plt.xlabel('f (Hz)')
        plt.ylabel('|P1(f)|')
        plt.title('Single-Sided Amplitude Spectrum')
        plt.grid(True)
        plt.yscale('log')
        plt.xscale('log')
        # plt.xlim(f[0], f[-1])
        # plt.ylim(1e-10, 1e-2)
        plt.show()

    return freq, ppT


def ppT_curve_fit(signal):
    par = rheo_waveform_params(signal, "curve_fit")

    dt = np.mean(np.diff(signal.coords['time'].values)) / 1e9
    freq = 1e9 / par['T'].values
    ppT = 1 / freq / dt

    return freq, ppT


def plot_rheo_periods(rheo, rheo_fitted, N_period, plot=None, periods=None):
    """
    rheo: rheometer scipp data array
    t_period: period duration
    periods: tupla (p_start, p_end)
    """
    time_rheo = rheo.coords['time']
    time_fft = rheo_fitted.coords['time']

    if plot in ['plot', 'overplot']:
        if plot == 'plot':
            plt.figure()
            if periods is not None:
                idx_start = (periods[0] - 1) * N_period
                idx_end = periods[1] * N_period
                plt.plot(
                    time_rheo[idx_start:idx_end].values - time_rheo[idx_start].values,
                    rheo[idx_start:idx_end].values,
                    marker='.',
                    linestyle='',
                    color='k',
                    markersize=3,
                )
                plt.plot(
                    time_fft[idx_start:idx_end].values - time_fft[idx_start].values,
                    rheo_fitted[idx_start:idx_end].values,
                    marker='',
                    linestyle='-',
                    color='r',
                    markersize=1,
                )
            else:
                plt.plot(time_rheo.values, rheo.values)
                plt.plot(time_fft.values, rheo_fitted.values)
            plt.xlabel('Time [ns]')
            plt.ylabel('Rheometer Deflection Angle')
            # plt.legend()
            plt.show()
        if plot == 'overplot':
            plt.figure()
            if periods is not None:
                for i in range(periods[0] - 1, periods[1]):
                    idx_start = i * N_period
                    idx_end = (i + 1) * N_period
                    plt.plot(
                        time_rheo[idx_start:idx_end].values
                        - time_rheo[idx_start].values,
                        rheo[idx_start:idx_end].values,
                        marker='.',
                        linestyle='',
                        color='k',
                        markersize=1,
                    )
                    plt.plot(
                        time_fft[idx_start:idx_end].values - time_fft[idx_start].values,
                        rheo_fitted[idx_start:idx_end].values,
                        marker='',
                        linestyle='-',
                        color='r',
                        markersize=1,
                    )
            else:
                for i in range(len(rheo) // N_period):
                    idx_start = i * N_period
                    idx_end = (i + 1) * N_period
                    plt.plot(
                        time_rheo[idx_start:idx_end].values
                        - time_rheo[idx_start].values,
                        rheo[idx_start:idx_end].values,
                        marker='.',
                        linestyle='',
                        color='k',
                        markersize=1,
                    )
                    plt.plot(
                        time_rheo[idx_start:idx_end].values
                        - time_rheo[idx_start].values,
                        rheo_fitted[idx_start:idx_end].values,
                        marker='',
                        linestyle='-',
                        color='r',
                        markersize=1,
                    )
            plt.xlabel('Time [ns]')
            plt.ylabel('Rheometer Deflection Angle')
            # plt.legend()
            plt.show()


def time_at_sample(
    event_time_zero, event_time_offset, distance_to_moderator, distance_to_sample
):
    return (
        # Think on this
        event_time_offset.to(unit=event_time_zero.unit)
        * distance_to_sample
        / distance_to_moderator
        + event_time_zero
        # event_time_offset * distance_to_sample / distance_to_moderator + event_time_zero.to(dtype=event_time_offset.dtype, unit=event_time_offset.unit)
        # event_time_offset * distance_to_sample / distance_to_moderator + event_time_zero.to(unit=event_time_offset.unit)
    )


def period_slice(time_at_sample, T, Toff, N_slices):
    first_edge = (Toff % T - T / 4 - T / N_slices / 2).to(unit=time_at_sample.unit)
    # print(first_edge)
    return (
        (
            (
                N_slices
                * ((time_at_sample - first_edge) / (T.to(unit=time_at_sample.unit)))
            )
            % N_slices
        ).to(dtype='int32')
        # ((N_slices * ((time_at_sample) / (T.to(unit=time_at_sample.unit)))) % N_slices).to(dtype='int32')
    )


def period_slice_trigger(time_at_sample, period_times, N_slices):
    func = sc.DataArray(
        period_times[1:] - period_times[:-1], coords={'time_at_sample': period_times}
    )
    lookUpTable = sc.lookup(func, 'time_at_sample')
    period_duration = lookUpTable[time_at_sample]

    func = sc.DataArray(period_times[:-1], coords={'time_at_sample': period_times})
    lookUpTable = sc.lookup(func, 'time_at_sample')
    period_start = lookUpTable[time_at_sample]

    return (
        (N_slices * ((time_at_sample - period_start) / period_duration)) % N_slices
    ).to(dtype='int32')


def dead_time_correction(
    events, bad_events, proton_charge, tdead, TOFmin, TOFmax, TOFbin
):
    import matplotlib.pyplot as plt

    # Grid on TOF placed at the edges of the bins
    tof_bin_edges = sc.arange(
        'event_time_offset',
        TOFmin - TOFbin / 2,
        TOFmax + TOFbin / 2,
        TOFbin,
        unit='us',
        dtype='float64',
    )
    # tof_bins = sc.arange('event_time_offset', TOFmin, TOFmax, TOFbin, unit='us', dtype='float64')

    # Histogram the event_time_offset
    mask = (proton_charge != sc.scalar(0, unit='pC')).data
    CT = events.hist(event_time_offset=tof_bin_edges, dim=events.dims)
    # CT = events[mask.rename_dims(time = 'event_time_zero')].hist(event_time_offset=tof_bin_edges, dim=events.dims)
    BCT = bad_events.hist(event_time_offset=tof_bin_edges, dim=events.dims)
    # BCT = bad_events[mask.rename_dims(time = 'event_time_zero')].hist(event_time_offset=tof_bin_edges, dim=events.dims)
    CT = CT + BCT
    CT = CT.values
    CT = CT / len(proton_charge[mask])

    # if useNP == 0: # (paralyzable)
    b = -lambertw(-CT * tdead / TOFbin) / (tdead / TOFbin)
    dtc = b / CT
    dtc = dtc.real
    dtc = np.nan_to_num(dtc, nan=1)
    # if useNP == 1: dtc = 1 / (1 - CT * tdead / tofbin)  # (non-paralyzable)

    # plt.plot(dtc)
    # plt.show()

    return sc.DataArray(
        sc.array(dims=['event_time_offset'], values=dtc),
        coords={'event_time_offset': tof_bin_edges},
    )

    # ---------------------------------------------------------------------------


def gravity_correct(LAM, ThetaIn, dSamp, dSlit):
    dSamp = dSamp / 1000  # dSamp is m from sample to incident slit
    dSlit = dSlit / 1000  # dSlit is m between slits

    # calculation from the ILL paper. this works for inclined beams.
    g = 9.8067  # m/s^2
    h = 6.6260715e-34  # Js=kg m^2/s
    mn = 1.67492749804e-27  # kg
    V = h / (mn * LAM * 1e-10)
    K = g / (2 * V**2)

    # define the sample position as x=0, y=0. increasing x is towards moderator
    xs = 0
    ys = 0

    # positions of slits
    x1 = dSamp
    x2 = dSamp + dSlit

    # height of slits determined by incident theta, y=0 is the sample height
    y1 = x1 * np.tan(ThetaIn * np.pi / 180)
    y2 = x2 * np.tan(ThetaIn * np.pi / 180)

    # this is the location of the top of the parabola
    x0 = (y1 - y2 + K * (x1**2 - x2**2)) / (2 * K * (x1 - x2))
    y0 = y2 + K * (x2 - x0) ** 2
    xs = x0 - np.sqrt(y0 / K)
    ThetaSam = (
        np.arctan(2 * K * (x0 - xs)) * 180 / np.pi
    )  # angle is arctan(dy/dx) at sample
    dTheta = ThetaSam - ThetaIn

    return dTheta


# -----------------------------------------------------------------------------
