"""每站单天线的二维 DPD-HR；与同名 MATLAB 文件共享数值合同。

输入坐标为米、频率为 Hz；IQ 为 (L, N)。返回 pos, mtr, info。
mtr[ix, iy]，info 中的索引统一为 MATLAB 的 1 起始、列优先约定。
详见 DPD_MVDR说明.md；本模块只依赖 NumPy，不执行文件写入或绘图。
"""

import numpy as np


class DPD_MVDR_Error(ValueError):
    """code 与 MATLAB 异常 identifier 一致。"""

    def __init__(self, code, message):
        self.code = "DPD_MVDR:" + code
        super().__init__(self.code + ": " + message)


def _fail(code, message):
    raise DPD_MVDR_Error(code, message)


def _real(value):
    try:
        array = np.asarray(value)
    except (TypeError, ValueError):
        _fail("InvalidInput", "Expected finite real numeric values.")
    if array.dtype.kind not in "iuf" or not array.size or not np.all(np.isfinite(array)):
        _fail("InvalidInput", "Expected finite real numeric values.")
    return array.astype(np.float64)


def _scalar(value, lower=0, strict=False):
    array = _real(value)
    if array.size != 1:
        _fail("InvalidInput", "Expected scalar.")
    result = float(array.item())
    if result < lower or (strict and result == lower):
        _fail("InvalidInput", "Scalar out of range.")
    return result


def DPD_MVDR(rcvPos, sig_rcv, init_pos, edge, lamda, fs, band, fc, opts=None):
    """保留 MATLAB 的8个位置参数，opts 为可选字典；始终返回三个输出。

    J=None/[] 为自动分段。可用配置及默认值见 options 字典。
    显式 J 不保证时延近似有效，调用方须核对 info.margin_met。
    pos 是全局峰；相对峰差在 PeakTieTolerance 内时按列优先取首项。
    """
    options = dict(J=None, SegmentMargin=40, DiagLoadMode="relative",
                   DiagLoad=1e-6, SpectrumLoad=0, FrequencyRange=None,
                   PropagationSpeed=299792458, PeakTieTolerance=1e-10)
    if opts is not None:
        if not isinstance(opts, dict) or set(opts) - set(options):
            _fail("InvalidOptions", "Unknown option or opts is not a dict.")
        options.update(opts)
    receivers = _real(rcvPos)
    center = _real(init_pos).reshape(-1)
    if receivers.ndim != 2 or receivers.shape[1] != 2 or receivers.shape[0] < 2 or center.size != 2:
        _fail("InvalidInput", "Expected L-by-2 receivers (L>=2) and a 2-element center.")
    try:
        signal = np.asarray(sig_rcv)
    except (TypeError, ValueError):
        _fail("InvalidInput", "Expected finite L-by-N IQ.")
    if (signal.dtype.kind not in "iufc" or signal.ndim != 2
            or signal.shape[0] != receivers.shape[0] or signal.shape[1] < 2
            or not np.all(np.isfinite(signal))):
        _fail("InvalidInput", "Expected finite L-by-N IQ, N>=2.")
    signal = signal.astype(np.complex128)
    edge, lamda = _scalar(edge), _scalar(lamda, strict=True)
    fs, band = _scalar(fs, strict=True), _scalar(band, strict=True)
    fc_array = _real(fc)
    if fc_array.size != 1 or band > fs:
        _fail("InvalidInput", "Require scalar fc and band<=fs.")
    fc = float(fc_array.item())
    margin_required = _scalar(options["SegmentMargin"], strict=True)
    speed = _scalar(options["PropagationSpeed"], strict=True)
    load_value = _scalar(options["DiagLoad"])
    spectrum_load = _scalar(options["SpectrumLoad"])
    tie_tolerance = _scalar(options["PeakTieTolerance"])
    if tie_tolerance >= 1:
        _fail("InvalidOptions", "PeakTieTolerance must be <1.")
    mode = options["DiagLoadMode"]
    if not isinstance(mode, str) or mode.lower() not in ("relative", "absolute"):
        _fail("InvalidOptions", "DiagLoadMode must be relative or absolute.")
    mode = mode.lower()
    stations, samples = signal.shape
    baseline = max(np.linalg.norm(receivers[i] - receivers[j])
                   for i in range(stations - 1) for j in range(i + 1, stations))
    if not np.isfinite(baseline) or baseline <= 0:
        _fail("InvalidGeometry", "Receiver baseline must be positive and finite.")
    delay_bound = baseline / speed
    requested_j = options["J"]
    if requested_j is None or np.asarray(requested_j).size == 0:
        minimum_samples = max(2, np.ceil(fs * margin_required * delay_bound))
        segments = int(np.floor(samples / minimum_samples))
        if segments < 1:
            _fail("InsufficientDuration", "No segment meets SegmentMargin; provide validated J or longer observations.")
    else:
        value = _scalar(requested_j, 1)
        if value != np.floor(value):
            _fail("InvalidInput", "J must be an integer.")
        segments = int(value)
    nfft = samples // segments
    if nfft < 2:
        _fail("InvalidSegments", "Each segment must contain at least two samples.")
    f = np.arange(-(nfft // 2), (nfft + 1) // 2, dtype=np.float64) * (fs / nfft)
    selected_range = options["FrequencyRange"]
    if selected_range is None or np.asarray(selected_range).size == 0:
        frequency_range = np.array([-band / 2, band / 2])
    else:
        frequency_range = _real(selected_range).reshape(-1)
        if (frequency_range.size != 2 or frequency_range[0] > frequency_range[1]
                or frequency_range[0] < -fs / 2 or frequency_range[1] > fs / 2):
            _fail("InvalidInput", "Invalid FrequencyRange.")
    ids = np.flatnonzero((f >= frequency_range[0]) & (f <= frequency_range[1]))
    if not ids.size:
        _fail("EmptyFrequencySelection", "No FFT bins in the requested range.")
    frequencies = f[ids]
    count = ids.size
    xf = np.empty((stations, count, segments), dtype=np.complex128)
    for j in range(segments):
        transformed = np.fft.fftshift(np.fft.fft(signal[:, j*nfft:(j+1)*nfft], axis=1), axes=1)
        xf[:, :, j] = transformed[:, ids]
    if not np.all(np.isfinite(xf)):
        _fail("NumericalFailure", "Non-finite FFT.")
    if not np.any(xf != 0):
        _fail("ZeroEnergy", "Selected observations have zero energy.")
    inverse_cov = np.empty((stations, stations, count), dtype=np.complex128)
    loads, ratios = np.empty(count), np.empty(count)
    identity = np.eye(stations)
    epsilon = np.finfo(np.float64).eps
    for k in range(count):
        covariance = np.zeros((stations, stations), dtype=np.complex128)
        for j in range(segments):
            v = xf[:, k, j]
            covariance += np.outer(v, v.conj())
        covariance /= segments
        covariance = (covariance + covariance.conj().T) / 2
        loads[k] = load_value * np.trace(covariance).real / stations if mode == "relative" else load_value
        loaded = covariance + loads[k] * identity
        if not np.all(np.isfinite(loaded)):
            _fail("NumericalFailure", "Non-finite covariance.")
        eigenvalues = np.linalg.eigvalsh(loaded)
        ratios[k] = eigenvalues[0] / eigenvalues[-1] if eigenvalues[-1] != 0 else np.nan
        if eigenvalues[-1] <= 0 or not np.isfinite(ratios[k]) or ratios[k] <= 64 * epsilon:
            _fail("SingularCovariance", "Covariance not numerically positive definite; review snapshots and loading.")
        inverse_cov[:, :, k] = np.linalg.solve(loaded, identity)
    grid_ratio = 2 * edge / lamda
    intervals = np.floor(grid_ratio + 8 * epsilon * max(1, abs(grid_ratio)))
    if not np.isfinite(intervals):
        _fail("InvalidInput", "Non-finite grid size.")
    offsets = np.arange(int(intervals) + 1, dtype=np.float64) * lamda
    x_vec, y_vec = center[0] - edge + offsets, center[1] - edge + offsets
    spectrum = np.empty((x_vec.size, y_vec.size))
    for ix, x in enumerate(x_vec):
        for iy, y in enumerate(y_vec):
            tau = np.sqrt(np.sum((np.array([x, y]) - receivers)**2, axis=1)) / speed
            matrix = np.zeros((stations, stations), dtype=np.complex128)
            for k, frequency in enumerate(frequencies):
                a = np.exp(-1j * 2 * np.pi * (fc + frequency) * tau)
                matrix += inverse_cov[:, :, k] * np.outer(a.conj(), a)
            matrix = (matrix + matrix.conj().T) / 2
            if not np.all(np.isfinite(matrix)):
                _fail("NumericalFailure", "Non-finite position matrix.")
            eigenvalues = np.linalg.eigvalsh(matrix)
            denominator = eigenvalues[0] + spectrum_load
            if denominator <= 64 * epsilon * (eigenvalues[-1] + spectrum_load) or not np.isfinite(denominator):
                _fail("InvalidSpectrum", "Position matrix not numerically positive definite.")
            spectrum[ix, iy] = 1 / denominator
    if not np.all(np.isfinite(spectrum)) or np.any(spectrum <= 0):
        _fail("InvalidSpectrum", "Spectrum must be finite and positive.")
    maximum = np.max(spectrum)
    ties = np.flatnonzero(spectrum.ravel(order="F") >= maximum - tie_tolerance * abs(maximum))
    ix, iy = np.unravel_index(ties[0], spectrum.shape, order="F")
    position = np.array([x_vec[ix], y_vec[iy]])
    margin = (nfft / fs) / delay_bound
    info = dict(J=segments, N_total=samples, N_used=segments*nfft, N_discarded=samples-segments*nfft,
                N_fft=nfft, K=count, frequencies_hz=frequencies, frequency_indices=ids+1,
                frequency_range_hz=frequency_range, x_vec=x_vec, y_vec=y_vec, baseline_m=baseline,
                delay_bound_s=delay_bound, segment_duration_s=nfft/fs, segment_margin=margin,
                required_margin=margin_required, margin_met=bool(margin >= margin_required),
                rank_bound=min(stations, segments), rank_limited=segments < stations,
                diag_load_per_frequency=loads, cov_eigenvalue_ratio=ratios, diag_load_mode=mode,
                diag_load_value=load_value, spectrum_load=spectrum_load, propagation_speed=speed,
                fc_hz=fc, fft_normalization="none", peak_index=np.array([ix+1, iy+1]),
                peak_linear_index=int(ties[0]+1), peak_tie_count=ties.size,
                peak_tie_tolerance=tie_tolerance, peak_value=spectrum[ix, iy], maximum_value=maximum,
                status="OK" if margin >= margin_required else "LOW_SEGMENT_MARGIN")
    return position, spectrum, info
