import numpy as np
import scipp as sc
import scipy as sp
import matplotlib.pyplot as plt
from scipy.stats import chi2


def sine_wave(time, A, T, Toff, Aoff):
    return A * sc.sin(sc.scalar(2 * np.pi, unit='rad') * (time - Toff) / T) + Aoff

def ss_fft(signal):
    dt = sc.mean(signal.coords["time"][1:] - signal.coords["time"][:-1]).to(unit="s").value

    Y = np.fft.rfft(signal.values)

    amplitude = np.abs(Y) / len(signal)
    amplitude[1:-1 if len(signal) % 2 == 0 else None] *= 2
    phase = np.angle(Y)
    frequency = np.fft.rfftfreq(len(signal), d=dt)

    return amplitude, phase, frequency

def rheo_waveform_params(rheo_binned, method):

    if method == "curve_fit":
        rheo_for_fit = rheo_binned.copy()
        rheo_for_fit.coords["time"] -= rheo_for_fit.coords["time"][0]

        amplitude, _, frequency = ss_fft(rheo_for_fit)
        max_index = np.argmax(amplitude[1:]) + 1
        T0 = sc.scalar(1 / frequency[max_index], unit="s").to(unit="ns")
        A0 = (sc.max(rheo_for_fit) - sc.min(rheo_for_fit)).data / 2
        Aoff0 = sc.mean(rheo_for_fit).data

        first_period = rheo_for_fit.coords["time"] < T0
        tmax = rheo_for_fit.coords["time"][first_period][
            (rheo_for_fit[first_period] == sc.max(rheo_for_fit[first_period])).data][0]
        Toff0 = tmax - T0 / 4
        par, _ = sc.curve_fit(["time"], sine_wave, rheo_for_fit, p0={"A": A0, "T": T0, "Toff": Toff0, "Aoff": Aoff0})
        par["T"] = par["T"].to(unit="ns")
        par["Toff"] = par["Toff"].to(unit="ns")

        if par["A"].value < 0:
            par["A"] = -par["A"]
            par["Toff"] += par["T"] / 2

    elif method == "fft":
        amplitude, phase, frequency = ss_fft(rheo_binned)
        max_index = np.argmax(amplitude[1:]) + 1

        freq = frequency[max_index]
        T = 1 / freq
        A = amplitude[max_index]
        Aoff = sc.mean(rheo_binned).value
        Toff = -T * (phase[max_index] + np.pi / 2) / (2 * np.pi)
        par = sc.DataGroup(A=sc.scalar(A), T=sc.scalar(T, unit="s").to(unit="ns"),
                           Toff=sc.scalar(Toff, unit="s").to(unit="ns"), Aoff=sc.scalar(Aoff))

    else:
        raise ValueError("method must be 'curve_fit' or 'fft'.")

    return par
    
def ppT_autocorrelation(signal, plot_flag=None):

    dt = sc.mean(signal.coords["time"][1:] - signal.coords["time"][:-1]).value

    y = signal.values - np.mean(signal.values)
    N = len(y)

    Npad = int(2 ** np.ceil(np.log2(2 * N)))
    Y = np.fft.rfft(y, n=Npad)
    R = np.fft.irfft(Y * np.conj(Y), n=Npad)[:N]

    peaks, _ = sp.signal.find_peaks(R[1:])
    ppT = peaks[0] + 1

    if 0 < ppT < len(R) - 1:
        y1, y2, y3 = R[ppT - 1], R[ppT], R[ppT + 1]
        ppT += 0.5 * (y1 - y3) / (y1 - 2 * y2 + y3)

    freq = 1 / (ppT * dt)

    if plot_flag:
        plt.figure()
        plt.plot(R / R[0])
        plt.xlabel("Lag [samples]")
        plt.ylabel("Autocorrelation")
        plt.show()

    return freq, ppT

def period_edges_trigger(signal, value, direction="rising"):
    
    y = (signal - value).data
    time = signal.coords["time"]
    zero = sc.scalar(0, unit=y.unit)

    if direction == "rising":
        crossing = (y[:-1] < zero) & (y[1:] >= zero)
    elif direction == "falling":
        crossing = (y[:-1] > zero) & (y[1:] <= zero)
    else:
        raise ValueError("direction must be 'rising' or 'falling'.")

    left = y[:-1][crossing]
    right = y[1:][crossing]
    fraction = -left / (right - left)

    t = time - time[0]
    edges = t[:-1][crossing] + fraction * (t[1:] - t[:-1])[crossing]

    return (time[0] + edges).rename_dims({signal.dim: "period"})

def resolve_period_times(start, stop, T=None, Toff=None):

    k0 = sc.floor((start - Toff) / T) - 1
    k1 = sc.ceil((stop - Toff) / T) + 1
    k = sc.arange("period", k0, k1 + 1, unit="", dtype="int64")

    return Toff + k * T

def period_slice(pulse_time, sample_offset, period_times, N_slices, n):

    M = N_slices * n

    reference = period_times["period", 0].to(dtype="int64")
    reference_time = sc.epoch(unit="ns") + reference
    periods = (period_times - reference).rename_dims(period="time_at_sample")
    time_at_sample = pulse_time - reference_time + sample_offset

    duration = sc.DataArray(periods[1:] - periods[:-1], coords={"time_at_sample": periods})
    start = sc.DataArray(periods[:-1], coords={"time_at_sample": periods})
    period_duration = sc.lookup(duration, "time_at_sample", fill_value=sc.scalar(1.0, unit=periods.unit))[time_at_sample]
    period_start = sc.lookup(start, "time_at_sample", fill_value=sc.scalar(0.0, unit=periods.unit))[time_at_sample]
    phase = (time_at_sample - period_start) / period_duration

    return (sc.floor(M * (phase + sc.scalar(1.0 / (2 * N_slices), unit=""))) % M).to(dtype="int32")

def phase_coverage(proton_charge, sample_offset_edges, period_times):
    reference = period_times["period", 0].to(dtype="int64")
    reference_time = sc.epoch(unit="ns") + reference
    pulse_time = proton_charge.coords["time"] - reference_time
    periods = period_times - reference

    return (
        (pulse_time + sample_offset_edges["TOF", 0] >= periods["period", 0])
        & (pulse_time + sample_offset_edges["TOF", -1] < periods["period", -1])
    )

######################################################################################################################

def phase_overlap_events(elementary, exposure, *, first, second, n):
    """Select shared events and scale their variances by 1 / (Q_first Q_second)."""

    M = elementary.sizes["source_period_slice"]

    source = elementary.coords["source_period_slice"]
    shared = ((source - first) % M < n) & ((source - second) % M < n)

    events = elementary[shared].bins.concat("source_period_slice")
    denominator = exposure["period_slice", first] * exposure["period_slice", second]
    denominator.data = sc.where(denominator.masks["no_coverage"], sc.scalar(1.0, unit=denominator.unit), denominator.data)
    events /= sc.sqrt(denominator)

    return events

def elementary_phase_moments(proton_charge, sample_offset_edges, period_times, *, N_slices, n, harmonics=()):
    """Compute elementary exposure and Fourier moments from selected pulse charges."""

    M = N_slices * n
    harmonics = tuple(harmonics)
    q = sc.values(proton_charge.data)

    reference = period_times["period", 0].to(dtype="int64")
    reference_time = sc.epoch(unit="ns") + reference
    periods = (period_times - reference).to(dtype="float64").rename_dims(period="time_at_sample")
    pulse_time = proton_charge.coords["time"] - reference_time
    period_start = sc.lookup(sc.DataArray(periods[:-1], coords={"time_at_sample": periods}), "time_at_sample", fill_value=sc.scalar(0.0, unit=periods.unit))
    period_duration = sc.lookup(sc.DataArray(periods[1:] - periods[:-1], coords={"time_at_sample": periods}), "time_at_sample", fill_value=sc.scalar(1.0, unit=periods.unit))

    cell_edges = sc.arange("source_period_slice", 0.0, M + 1.0, 1.0, unit="")
    left = cell_edges[:-1] / M
    right = cell_edges[1:] / M
    origin = sc.scalar(-1.0 / (2 * N_slices), unit="")
    phase_zero = -origin

    harmonic = sc.array(dims=["harmonic"], values=harmonics, unit="", dtype="float64")
    omega = sc.scalar(2.0 * np.pi, unit="") * harmonic
    rad = sc.scalar(1.0, unit="rad")

    def basis_integral(lo, hi):
        width = hi - lo
        constant = sc.concat((width,), dim="basis")
        if not harmonics:
            return constant

        angle_lo = rad * omega * (lo + origin)
        angle_hi = rad * omega * (hi + origin)
        cosine = (sc.sin(angle_hi) - sc.sin(angle_lo)) / omega
        sine = (sc.cos(angle_lo) - sc.cos(angle_hi)) / omega
        oscillatory = sc.concat((cosine, sine), dim="component")
        oscillatory = oscillatory.transpose(["harmonic", "component", *width.dims]).flatten(dims=["harmonic", "component"], to="basis")

        return sc.concat((constant, oscillatory), dim="basis")

    cell_integral = basis_integral(left, right)
    zero_upper = sc.where(right < phase_zero, right, phase_zero)
    zero_primitive = sc.where(zero_upper > left, basis_integral(left, zero_upper), sc.scalar(0.0, unit=""))

    def primitive(sample_offset):
        time_at_sample = pulse_time + sample_offset
        start = period_start[time_at_sample]
        duration = period_duration[time_at_sample]
        phase = (time_at_sample - start) / duration
        u = (phase - origin) % sc.scalar(1.0, unit="")

        cell_position = M * u
        cell_left = sc.floor(cell_position) / M
        weight = q * duration
        partial = basis_integral(cell_left, u)

        cell_weight = sc.DataArray(weight, coords={"source_period_slice": cell_position}).hist(source_period_slice=cell_edges, dim="time").data
        partial_sum = sc.DataArray(weight * partial, coords={"source_period_slice": cell_position}).hist(source_period_slice=cell_edges, dim="time").data
        total_weight = sc.sum(cell_weight, dim="source_period_slice")
        weight_above = total_weight - sc.cumsum(cell_weight, dim="source_period_slice")
        shifted = cell_integral * weight_above + partial_sum
        complete_cycles = cell_integral * sc.sum(q * start, dim="time")
        wrapped_weight = sc.sum(sc.where(u < phase_zero, weight, sc.scalar(0.0, unit=weight.unit)), dim="time")

        return complete_cycles + shifted - zero_primitive * total_weight + cell_integral * wrapped_weight

    integrals = sc.concat(tuple(map(primitive, sc.slices(sample_offset_edges, "TOF").values())), dim="TOF_edge")
    moments = (integrals["TOF_edge", 1:] - integrals["TOF_edge", :-1]).rename_dims({"TOF_edge": "TOF"}) / (sample_offset_edges["TOF", 1:] - sample_offset_edges["TOF", :-1])
    moments = moments.transpose(["source_period_slice", "basis", "TOF"])

    labels = ["constant"] + [label for k in harmonics for label in (f"cos_{k}", f"sin_{k}")]

    return sc.DataArray(moments, coords={
        "source_period_slice": sc.arange("source_period_slice", M, unit="", dtype="int32"),
        "basis": sc.array(dims=["basis"], values=labels, unit=None),
    })

def proton_charge_phase_exposure(proton_charge, sample_offset_edges, period_times, *, N_slices, n):
    moments = elementary_phase_moments(proton_charge, sample_offset_edges, period_times, N_slices=N_slices, n=n)
    elementary = moments["basis", 0].copy(deep=False)
    del elementary.coords["basis"]
    exposure = sum_elementary_windows(elementary, n)

    return exposure, elementary

def select_phase_exposure(physical_exposure, proton_charge, *, N_slices, mode="physical"):
    exposure = physical_exposure.copy(deep=False)
    if mode == "uniform":
        total = sc.values(proton_charge).sum("time").data
        exposure.data = (total / N_slices).broadcast(sizes=physical_exposure.sizes)

    exposure.masks["no_coverage"] = physical_exposure.data <= sc.scalar(0.0, unit=physical_exposure.unit)
    return exposure

def phase_exposure_response(proton_charge, sample_offset_edges, period_times, *, N_slices, n, harmonics, exposure_mode="physical", pulse_selection=None):
    accepted = phase_coverage(proton_charge, sample_offset_edges, period_times)
    for mask in proton_charge.masks.values():
        accepted &= ~mask
    if pulse_selection is not None:
        accepted &= pulse_selection

    proton_charge = proton_charge[accepted]
    moments = elementary_phase_moments(proton_charge, sample_offset_edges, period_times, N_slices=N_slices, n=n, harmonics=harmonics)
    window_moments = sum_elementary_windows(moments, n)

    physical = window_moments["basis", 0].copy(deep=False)
    del physical.coords["basis"]

    denominator = select_phase_exposure(physical, proton_charge, N_slices=N_slices, mode=exposure_mode)
    safe = sc.where(denominator.masks["no_coverage"], sc.scalar(1.0, unit=denominator.unit), denominator.data)
    response = window_moments / safe
    response.masks["no_coverage"] = denominator.masks["no_coverage"]

    return sc.DataGroup(response=response, physical_exposure=physical, normalization=denominator, elementary_moments=moments)


def sum_elementary_windows(elementary, n):
    dim = "source_period_slice"
    M = elementary.sizes[dim]
    data = elementary.data
    extended = sc.concat((sc.zeros_like(data[dim, :1]), data, data[dim, :n - 1]), dim=dim)
    cumulative = sc.cumsum(extended, dim=dim)
    summed = (cumulative[dim, n:n + M] - cumulative[dim, :M]).rename_dims({dim: "period_slice"})
    coords = {name: coord for name, coord in elementary.coords.items() if dim not in coord.dims}
    coords["period_slice"] = elementary.coords[dim].rename_dims({dim: "period_slice"})

    return sc.DataArray(summed, coords=coords)


def with_event_coord(events, name, coord):
    constituents = events.bins.constituents
    flat_coord = coord.bins.constituents["data"]
    content = constituents["data"].assign_coords({name: flat_coord})
    data = sc.bins(data=content, dim=constituents["dim"], begin=constituents["begin"], end=constituents["end"], validate_indices=False)

    return sc.DataArray(data, coords=dict(events.coords), masks=dict(events.masks))


def dense_mask(da, dims):
    invalid = sc.zeros(sizes={d: da.sizes[d] for d in dims}, dtype="bool", unit=None)
    for mask in da.masks.values():
        invalid |= mask
    return invalid.transpose(dims).values


def fourier_evaluation(M, labels):
    phase = 2 * np.pi * np.arange(M) / M
    columns = []
    for label in labels:
        if label == "constant":
            columns.append(np.ones(M))
        else:
            kind, order = label.split("_")
            columns.append(getattr(np, kind)(int(order) * phase))

    return np.column_stack(columns)


def deconv_exposure(signal, response, *, covariance, rcond=1e-12, max_condition=1e8, check_only=False, required_valid=None):
    signal = signal.transpose(["period_slice", "lambda"])
    covariance = covariance.transpose(["period_slice", "period_slice_2", "lambda"])
    M, L = signal.shape

    if "lambda" in response.dims:
        response = response.transpose(["period_slice", "basis", "lambda"])
    else:
        response = response.broadcast(sizes={"period_slice": M, "basis": response.sizes["basis"], "lambda": L})
        response.coords["lambda"] = signal.coords["lambda"]

    labels = list(response.coords["basis"].values)
    P = len(labels)
    invalid = dense_mask(signal, ["period_slice", "lambda"])
    for mask in response.masks.values():
        veto = sc.any(mask, dim="basis") if "basis" in mask.dims else mask
        invalid |= sc.broadcast(veto, sizes=signal.sizes).transpose(["period_slice", "lambda"]).values

    bad = np.ones(L, dtype=bool)
    rank = np.zeros(L, dtype=np.int64)
    valid_count = np.zeros(L, dtype=np.int64)
    condition = np.full(L, np.inf)
    singular = np.zeros((P, L))
    decompositions = []

    for ell in range(L):
        cmask = dense_mask(covariance["lambda", ell], ["period_slice", "period_slice_2"])
        good = ~invalid[:, ell] & ~np.diag(cmask)
        ii = np.flatnonzero(good)
        if ii.size and np.any(cmask[np.ix_(ii, ii)]):
            raise ValueError("Missing covariance between selected observations.")

        valid_count[ell] = ii.size
        if ii.size < P:
            continue

        X = response.values[:, :, ell][ii]
        U, s, Vh = np.linalg.svd(X, full_matrices=False)
        singular[:s.size, ell] = s
        rank[ell] = np.count_nonzero(s > rcond * s[0])
        if rank[ell] != P:
            continue

        condition[ell] = s[0] / s[-1]
        if condition[ell] > max_condition:
            continue

        bad[ell] = False
        if not check_only:
            decompositions.append((ell, ii, U, s, Vh))

    valid = sc.DataArray(sc.array(dims=["lambda"], values=~bad, unit=None), coords={"lambda": signal.coords["lambda"]})
    if check_only:
        return valid
    if required_valid is not None and not sc.all(valid | ~required_valid).value:
        return None

    B = fourier_evaluation(M, labels)
    estimate = np.full((M, L), np.nan)
    fit = np.full((M, L), np.nan)
    transfer = np.zeros((M, M, L))
    Cout = np.zeros((M, M, L))
    beta = np.full((P, L), np.nan)
    Cbeta = np.zeros((P, P, L))
    chi2 = np.full(L, np.nan)
    residual_rank = np.zeros(L, dtype=np.int64)
    null_residual = np.full(L, np.nan)
    residual_rms = np.full(L, np.nan)

    for ell, ii, U, s, Vh in decompositions:
        X = response.values[:, :, ell][ii]
        C = covariance.values[:, :, ell][np.ix_(ii, ii)]
        cscale = max(np.max(np.abs(C)), np.finfo(float).tiny)
        C = (C + C.T) / 2
        inverse = (Vh.T / s) @ U.T
        A = B @ inverse
        y = signal.values[ii, ell]

        beta[:, ell] = inverse @ y
        Cbeta[:, :, ell] = inverse @ C @ inverse.T
        estimate[:, ell] = B @ beta[:, ell]
        fit[:, ell] = response.values[:, :, ell] @ beta[:, ell]
        transfer[:, ii, ell] = A
        Cout[:, :, ell] = A @ C @ A.T
        Cout[:, :, ell] = (Cout[:, :, ell] + Cout[:, :, ell].T) / 2

        residual = y - X @ beta[:, ell]
        R = np.eye(ii.size) - X @ inverse
        residual_C = R @ C @ R.T
        ev, vectors = np.linalg.eigh((residual_C + residual_C.T) / 2)
        keep = ev > rcond * cscale
        residual_rank[ell] = np.count_nonzero(keep)
        rotated = vectors.T @ residual
        chi2[ell] = np.sum(rotated[keep] ** 2 / ev[keep])
        null_residual[ell] = np.linalg.norm(rotated[~keep])
        residual_rms[ell] = np.sqrt(np.mean(residual ** 2))

    wave = signal.coords["lambda"]
    phase = sc.array(dims=["phase"], values=2 * np.pi * np.arange(M) / M, unit="rad")
    invalid_wave = sc.array(dims=["lambda"], values=bad, unit=None)
    coords = {"phase": phase, "lambda": wave}
    masks = {"not_identifiable": invalid_wave}
    coef_coords = {"basis": response.coords["basis"], "lambda": wave}

    out = sc.DataArray(sc.array(dims=["phase", "lambda"], values=estimate, variances=np.maximum(np.einsum("iil->il", Cout), 0), unit=signal.unit), coords=coords, masks=masks)
    out_cov = sc.DataArray(sc.array(dims=["phase", "phase_2", "lambda"], values=Cout, unit=covariance.unit), coords={**coords, "phase_2": phase.rename_dims(phase="phase_2")}, masks=masks)
    coef = sc.DataArray(sc.array(dims=["basis", "lambda"], values=beta, variances=np.maximum(np.einsum("iil->il", Cbeta), 0), unit=signal.unit), coords=coef_coords, masks=masks)
    coef_cov = sc.DataArray(sc.array(dims=["basis", "basis_2", "lambda"], values=Cbeta, unit=covariance.unit), coords={**coef_coords, "basis_2": response.coords["basis"].rename_dims(basis="basis_2")}, masks=masks)

    diagnostics = sc.Dataset({
        "rank": sc.array(dims=["lambda"], values=rank, unit=None),
        "valid_observations": sc.array(dims=["lambda"], values=valid_count, unit=None),
        "condition_number": sc.array(dims=["lambda"], values=condition, unit=""),
        "residual_chi2": sc.array(dims=["lambda"], values=chi2, unit=""),
        "residual_covariance_rank": sc.array(dims=["lambda"], values=residual_rank, unit=None),
        "residual_null_norm": sc.array(dims=["lambda"], values=null_residual, unit=signal.unit),
        "residual_rms": sc.array(dims=["lambda"], values=residual_rms, unit=signal.unit),
    }, coords={"lambda": wave})

    return sc.DataGroup(
        signal=out,
        covariance=out_cov,
        coefficients=coef,
        coefficient_covariance=coef_cov,
        transfer=sc.DataArray(sc.array(dims=["phase", "period_slice", "lambda"], values=transfer, unit=""), coords={**coords, "period_slice": signal.coords["period_slice"]}, masks=masks),
        fitted=sc.DataArray(sc.array(dims=["period_slice", "lambda"], values=fit, unit=signal.unit), coords=signal.coords, masks={**dict(signal.masks), "not_identifiable": invalid_wave}),
        singular_values=sc.DataArray(sc.array(dims=["basis", "lambda"], values=singular, unit=""), coords=coef_coords),
        diagnostics=diagnostics,
    )


def integrate_phase_band(signal, covariance):
    dim = "phase" if "phase" in signal.dims else "period_slice"
    other = "phase_2" if dim == "phase" else "period_slice_2"
    signal = signal.transpose([dim, "lambda"])
    covariance = covariance.transpose([dim, other, "lambda"])

    lam = signal.coords["lambda"]
    weights = lam["lambda", 1:] - lam["lambda", :-1]
    incomplete = np.any(dense_mask(signal, [dim, "lambda"]), axis=1)
    incomplete_covariance = np.any(dense_mask(covariance, [dim, other, "lambda"]), axis=2)
    incomplete_covariance |= incomplete_covariance.T
    incomplete |= np.diag(incomplete_covariance)

    total = (signal * weights).sum("lambda")
    C = (covariance * weights ** 2).sum("lambda")
    invalid = sc.array(dims=[dim], values=incomplete, unit=None)
    total.masks["incomplete_band"] = invalid
    C.masks["invalid_first"] = invalid
    C.masks["invalid_second"] = invalid.rename_dims({dim: other})
    C.masks["incomplete_covariance"] = sc.array(dims=[dim, other], values=incomplete_covariance, unit=None)
    total.variances = np.maximum(np.diag(C.values), 0)

    return sc.DataGroup(signal=total, covariance=C)


def integrate_fourier_band(coefficients, covariance):
    coefficients = coefficients.transpose(["basis", "lambda"])
    covariance = covariance.transpose(["basis", "basis_2", "lambda"])
    lam = coefficients.coords["lambda"]
    weights = lam["lambda", 1:] - lam["lambda", :-1]

    incomplete = np.any(dense_mask(coefficients, ["basis", "lambda"])) or np.any(dense_mask(covariance, ["basis", "basis_2", "lambda"]))
    beta = (coefficients * weights).sum("lambda")
    Cbeta = (covariance * weights ** 2).sum("lambda")
    beta.variances = np.maximum(np.diag(Cbeta.values), 0)
    mask = sc.scalar(bool(incomplete))
    beta.masks["incomplete_band"] = mask
    Cbeta.masks["incomplete_band"] = mask

    return sc.DataGroup(coefficients=beta, covariance=Cbeta)


def reconstruct_fourier_band(coefficients, covariance, *, hmax, phase):
    labels = list(coefficients.coords["basis"].values)
    selected_labels = ["constant"] + [name for h in range(1, hmax + 1) for name in (f"cos_{h}", f"sin_{h}")]
    indices = [labels.index(label) for label in selected_labels]
    beta = coefficients.values[indices]
    Cbeta = covariance.values[np.ix_(indices, indices)]

    phi = phase.values
    columns = [np.ones(phi.size)]
    for h in range(1, hmax + 1):
        columns.extend((np.cos(h * phi), np.sin(h * phi)))

    B = np.column_stack(columns)
    values = B @ beta
    Cout = B @ Cbeta @ B.T
    Cout = (Cout + Cout.T) / 2
    basis = sc.array(dims=["basis"], values=selected_labels, unit=None)

    selected_coefficients = sc.DataArray(sc.array(dims=["basis"], values=beta, variances=np.maximum(np.diag(Cbeta), 0), unit=coefficients.unit), coords={"basis": basis}, masks=dict(coefficients.masks))
    selected_covariance = sc.DataArray(sc.array(dims=["basis", "basis_2"], values=Cbeta, unit=covariance.unit), coords={"basis": basis, "basis_2": basis.rename_dims(basis="basis_2")}, masks=dict(covariance.masks))
    signal = sc.DataArray(sc.array(dims=["phase"], values=values, variances=np.maximum(np.diag(Cout), 0), unit=coefficients.unit), coords={"phase": phase}, masks=dict(coefficients.masks))
    signal_covariance = sc.DataArray(sc.array(dims=["phase", "phase_2"], values=Cout, unit=covariance.unit), coords={"phase": phase, "phase_2": phase.rename_dims(phase="phase_2")}, masks=dict(covariance.masks))

    return sc.DataGroup(signal=signal, covariance=signal_covariance, coefficients=selected_coefficients, coefficient_covariance=selected_covariance)


def select_response_harmonics(response, harmonics):
    labels = {"constant"} | {name for k in harmonics for name in (f"cos_{k}", f"sin_{k}")}
    keep = sc.array(dims=["basis"], values=[label in labels for label in response.coords["basis"].values], unit=None)
    return response[keep]


def estimate_hmax(signal, response, *, covariance, alpha=0.05, rcond=1e-12, max_condition=1e8):
    labels = list(response.coords["basis"].values)
    cos_h = {int(label.split("_")[1]) for label in labels if label.startswith("cos_")}
    sin_h = {int(label.split("_")[1]) for label in labels if label.startswith("sin_")}
    available = sorted(cos_h & sin_h)
    constant_response = select_response_harmonics(response, ())
    reference_valid = deconv_exposure(signal, constant_response, covariance=covariance, rcond=rcond, max_condition=max_condition, check_only=True)
    if not sc.any(reference_valid).value:
        raise RuntimeError("The constant component is not identifiable in any wavelength channel.")

    H = 0
    full = None
    for h in reversed(available):
        candidate = select_response_harmonics(response, range(1, h + 1))
        full = deconv_exposure(signal, candidate, covariance=covariance, rcond=rcond, max_condition=max_condition, required_valid=reference_valid)
        if full is not None:
            H = h
            break

    if full is None:
        full = deconv_exposure(signal, constant_response, covariance=covariance, rcond=rcond, max_condition=max_condition)

    band_fourier = integrate_fourier_band(full["coefficients"], full["coefficient_covariance"])
    coef = band_fourier["coefficients"]
    Ccoef = band_fourier["covariance"]
    labels = list(coef.coords["basis"].values)
    beta = coef.values
    Cbeta = Ccoef.values
    harmonics = np.arange(1, H + 1, dtype=np.int64)
    wald = np.full(H, np.nan)
    dof = np.zeros(H, dtype=np.int64)
    p_value = np.full(H, np.nan)

    for i, h in enumerate(harmonics):
        indices = [labels.index(f"cos_{h}"), labels.index(f"sin_{h}")]
        b = beta[indices]
        C = Cbeta[np.ix_(indices, indices)]
        Cinv, rank = sp.linalg.pinvh(C, rtol=rcond, return_rank=True)
        wald[i] = b @ Cinv @ b
        dof[i] = rank
        p_value[i] = chi2.sf(wald[i], rank)

    p_adjusted = np.minimum(H * p_value, 1.0)
    significant = p_adjusted < alpha
    selected = int(harmonics[significant][-1]) if np.any(significant) else 0

    diagnostics = sc.Dataset({
        "wald": sc.array(dims=["harmonic"], values=wald, unit=""),
        "dof": sc.array(dims=["harmonic"], values=dof, unit=None),
        "p_value": sc.array(dims=["harmonic"], values=p_value, unit=""),
        "p_adjusted": sc.array(dims=["harmonic"], values=p_adjusted, unit=""),
        "significant": sc.array(dims=["harmonic"], values=significant, unit=None),
    }, coords={"harmonic": sc.array(dims=["harmonic"], values=harmonics, unit=None)})

    return selected, diagnostics, full, band_fourier


def reconstruct_wavelength_band(signal, covariance, response, *, lambda_lo, lambda_hi, alpha=0.05, rcond=1e-12, max_condition=1e8):
    band = signal["lambda", lambda_lo:lambda_hi]
    C_band = covariance["lambda", lambda_lo:lambda_hi]
    response_all = response["lambda", lambda_lo:lambda_hi]
    hmax, hmax_diagnostics, deconv, band_fourier = estimate_hmax(band, response_all, covariance=C_band, alpha=alpha, rcond=rcond, max_condition=max_condition)

    fit_labels = deconv["coefficients"].coords["basis"].values
    fit_harmonics = [int(label.split("_")[1]) for label in fit_labels if label.startswith("cos_")]
    fit_response = select_response_harmonics(response_all, fit_harmonics)
    selected_response = select_response_harmonics(response_all, range(1, hmax + 1))
    observed = integrate_phase_band(band, C_band)
    reconstructed = reconstruct_fourier_band(band_fourier["coefficients"], band_fourier["covariance"], hmax=hmax, phase=deconv["signal"].coords["phase"])

    return sc.DataGroup(
        band=band,
        covariance=C_band,
        response=selected_response,
        hmax=sc.scalar(hmax, unit=None),
        harmonics=sc.arange("harmonic", 1, hmax + 1, unit=None, dtype="int64"),
        hmax_diagnostics=hmax_diagnostics,
        deconvolution=deconv,
        band_coefficients=band_fourier["coefficients"],
        band_coefficient_covariance=band_fourier["covariance"],
        observed=observed,
        reconstructed=reconstructed,
        fit_response=fit_response,
    )


def calabrese_response(N_slices, n, harmonics):
    M = N_slices * n
    phase = 2 * np.pi * np.arange(M) / M
    columns = [np.ones(M)]
    labels = ["constant"]

    for h in harmonics:
        transfer = np.sinc(h / N_slices)
        columns.extend((transfer * np.cos(h * phase), transfer * np.sin(h * phase)))
        labels.extend((f"cos_{h}", f"sin_{h}"))

    return sc.DataArray(sc.array(dims=["period_slice", "basis"], values=np.column_stack(columns), unit=""), coords={
        "period_slice": sc.arange("period_slice", M, unit="", dtype="int32"),
        "basis": sc.array(dims=["basis"], values=labels, unit=None),
    })


def compare_with_calabrese(signal, covariance, physical_response, *, N_slices, n, hmax, rcond=1e-12, max_condition=1e8, exact=None):
    signal = signal.transpose(["period_slice", "lambda"])
    covariance = covariance.transpose(["period_slice", "period_slice_2", "lambda"])
    physical_response = physical_response.transpose(["period_slice", "basis", "lambda"])
    labels = physical_response.coords["basis"].values
    harmonics = [int(label.split("_")[1]) for label in labels if label.startswith("cos_")]

    if exact is None:
        exact = deconv_exposure(signal, physical_response, covariance=covariance, rcond=rcond, max_condition=max_condition)

    box_response = calabrese_response(N_slices, n, harmonics)
    box_response = box_response.broadcast(sizes={**box_response.sizes, "lambda": signal.sizes["lambda"]})
    box_response.coords["lambda"] = signal.coords["lambda"]
    for name, mask in physical_response.masks.items():
        box_response.masks[name] = sc.any(mask, dim="basis") if "basis" in mask.dims else mask

    calabrese = deconv_exposure(signal, box_response, covariance=covariance, rcond=rcond, max_condition=max_condition)
    exact_fourier = integrate_fourier_band(exact["coefficients"], exact["coefficient_covariance"])
    calabrese_fourier = integrate_fourier_band(calabrese["coefficients"], calabrese["coefficient_covariance"])
    phase = exact["signal"].coords["phase"]
    band_exact = reconstruct_fourier_band(exact_fourier["coefficients"], exact_fourier["covariance"], hmax=hmax, phase=phase)
    band_calabrese = reconstruct_fourier_band(calabrese_fourier["coefficients"], calabrese_fourier["covariance"], hmax=hmax, phase=phase)

    selected_labels = band_exact["coefficients"].coords["basis"].values
    B = fourier_evaluation(signal.sizes["period_slice"], selected_labels)
    projector = B @ np.linalg.pinv(B)
    A_exact = np.einsum("ab,bcl->acl", projector, exact["transfer"].values)
    A_calabrese = np.einsum("ab,bcl->acl", projector, calabrese["transfer"].values)
    D = A_exact - A_calabrese

    lam = signal.coords["lambda"]
    widths = (lam["lambda", 1:] - lam["lambda", :-1]).values
    M = signal.sizes["period_slice"]
    Cdiff = np.zeros((M, M))
    Ccross = np.zeros((M, M))

    for ell, width in enumerate(widths):
        ae_full = A_exact[:, :, ell]
        ac_full = A_calabrese[:, :, ell]
        used = np.any(ae_full != 0.0, axis=0) | np.any(ac_full != 0.0, axis=0)
        ii = np.flatnonzero(used)
        if ii.size == 0:
            continue

        C = covariance.values[:, :, ell][np.ix_(ii, ii)]
        cmask = dense_mask(covariance["lambda", ell], ["period_slice", "period_slice_2"])
        if np.any(cmask[np.ix_(ii, ii)]):
            raise ValueError(f"Missing covariance between observations used in wavelength channel {ell}.")

        C = (C + C.T) / 2
        d = width * D[:, :, ell][:, ii]
        ae = width * ae_full[:, ii]
        ac = width * ac_full[:, ii]
        Cdiff += d @ C @ d.T
        Ccross += ae @ C @ ac.T

    Cdiff = (Cdiff + Cdiff.T) / 2
    difference = sc.values(band_exact["signal"]) - sc.values(band_calabrese["signal"])
    difference.variances = np.maximum(np.diag(Cdiff), 0)
    diff_covariance = sc.DataArray(sc.array(dims=["phase", "phase_2"], values=Cdiff, unit=band_exact["covariance"].unit), coords=band_exact["covariance"].coords, masks=dict(difference.masks))
    cross_covariance = sc.DataArray(sc.array(dims=["phase", "phase_2"], values=Ccross, unit=band_exact["covariance"].unit), coords=band_exact["covariance"].coords, masks=dict(difference.masks))

    return sc.DataGroup(
        exposure=exact,
        calabrese=calabrese,
        band_exposure=band_exact,
        band_calabrese=band_calabrese,
        band_difference=sc.DataGroup(signal=difference, covariance=diff_covariance),
        band_cross_covariance=cross_covariance,
        band_observed=integrate_phase_band(signal, covariance),
    )


def relative_difference(exposure, calabrese, difference):
    E = exposure.values
    D = difference.values
    VE = exposure.variances
    VC = calabrese.variances
    VD = difference.variances
    CovDE = 0.5 * (VD + VE - VC)

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        relative = 100.0 * D / E
        variance = 100.0 ** 2 * (VD / E ** 2 + D ** 2 * VE / E ** 4 - 2.0 * D * CovDE / E ** 3)

    invalid = sc.array(dims=difference.dims, values=(E == 0.0) | ~np.isfinite(relative) | ~np.isfinite(variance))
    for source in (exposure, calabrese, difference):
        for mask in source.masks.values():
            invalid |= mask

    return sc.DataArray(sc.array(dims=difference.dims, values=relative, variances=np.maximum(variance, 0.0), unit=""), coords=difference.coords, masks={"invalid": invalid})