"""DPD_MVDR 双语言轻量回归；只生成合成输入，不读取科研数据。

运行：项目 Python verify_DPD_MVDR.py；需要 MATLAB 在 PATH 中或 --matlab。
每次写入新的 outputs_e2e/verification/dpd_mvdr/<时间戳>，保留失败尝试。
"""

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import traceback
import uuid

import numpy as np
from scipy.io import loadmat, savemat

from DPD_MVDR import DPD_MVDR, DPD_MVDR_Error


ROOT = Path(__file__).resolve().parent


def cases():
    rng = np.random.default_rng(20260927)
    receivers = np.array([[50., 0.], [0., 50.], [-50., 0.], [0., -50.]])

    def noise(n):
        return (rng.normal(size=(4, n)) + 1j*rng.normal(size=(4, n))) / np.sqrt(2)

    def case(name, n=128, opts=None, **changes):
        value = dict(name=name, rcvPos=receivers.copy(), sig_rcv=noise(n),
                     init_pos=np.array([0., 0.]), edge=20., lamda=20., fs=1e6,
                     band=1e6, fc=0., opts={"J": 8} if opts is None else opts,
                     error="")
        value.update(changes)
        return value

    result = [
        case("automatic", n=160, opts={}, omit_opts=True),
        case("even_full"),
        case("odd_tail", n=123),
        case("offset_band", opts={"J": 8, "FrequencyRange": [-250000., -100000.]}),
        case("absolute", opts={"J": 8, "DiagLoadMode": "absolute", "DiagLoad": .001, "SpectrumLoad": .001}),
        case("no_loading", opts={"J": 8, "DiagLoad": 0.}),
        case("rank_limited", opts={"J": 2}),
        case("single_snapshot", opts={"J": 1, "DiagLoad": 1e-3}),
        case("one_grid_point", edge=0.),
        case("decimal_grid", edge=.3, lamda=.1, init_pos=np.array([1.2, -.7])),
        case("nondivisible_grid", edge=1., lamda=.6),
        case("dc_ties", opts={"J": 8, "FrequencyRange": [0., 0.]}),
        case("real_input", sig_rcv=rng.normal(size=(4, 128))),
        case("single_precision", sig_rcv=noise(128).astype(np.complex64)),
    ]
    base = result[1]
    result.extend([
        case("high_carrier", sig_rcv=base["sig_rcv"], fc=5.8e9),
        case("weak_scale", sig_rcv=base["sig_rcv"]*1e-9),
        case("receiver_permutation", sig_rcv=base["sig_rcv"][[2, 0, 3, 1]],
             rcvPos=receivers[[2, 0, 3, 1]]),
    ])
    # 作者M=1的频域数据生成方式；IFFT仅封装相同系数，不验证有限窗近似。
    stations, segments, bins = 4, 6, 64
    fs = 150e3
    f_author = np.arange(bins) * fs / bins
    rcv = np.array([[100., -150.], [100., -50.], [100., 50.], [100., 150.]])*1000
    sources = np.array([[-1.5, -50.], [1.5, -50.]])*1000
    source = (rng.normal(size=(2, bins, segments)) + 1j*rng.normal(size=(2, bins, segments)))/np.sqrt(2)
    source /= np.linalg.norm(source, axis=1, keepdims=True)
    fd = np.empty((stations, bins, segments), dtype=np.complex128)
    for station in range(stations):
        alpha = .995 + .1*(rng.normal(size=2)+1j*rng.normal(size=2))/np.sqrt(2)
        tau = np.linalg.norm(sources-rcv[station], axis=1)/3e8
        phase = np.exp(-2j*np.pi*tau[:, None]*f_author)
        fd[station] = 10*np.einsum("q,qk,qkj->kj", alpha, phase, source)
        fd[station] += (rng.normal(size=(bins, segments))+1j*rng.normal(size=(bins, segments)))/np.sqrt(2*bins)
    signal = np.concatenate([np.fft.ifft(np.fft.ifftshift(fd[:, :, j], axes=1), axis=1)
                             for j in range(segments)], axis=1)
    result.append(case("author_m1", n=segments*bins, sig_rcv=signal, rcvPos=rcv,
                       fs=fs, band=fs, init_pos=np.array([0., -50000.]), edge=1500., lamda=1500.,
                       opts={"J": segments, "DiagLoadMode": "absolute", "DiagLoad": .001,
                             "SpectrumLoad": .001, "PropagationSpeed": 3e8},
                       reference_fd=fd, reference_f=f_author))
    # 具有已知网格峰的单源，检验相位符号和坐标顺序。
    segments, bins, fs = 8, 32, 10e6
    frequencies = np.arange(-bins//2, bins//2)*fs/bins
    truth = np.array([20., -20.])
    tau = np.linalg.norm(receivers-truth, axis=1)/299792458
    source = rng.normal(size=(bins, segments))+1j*rng.normal(size=(bins, segments))
    fd = np.exp(-2j*np.pi*tau[:, None, None]*frequencies[None, :, None])*source[None, :, :]
    signal = np.concatenate([np.fft.ifft(np.fft.ifftshift(fd[:, :, j], axes=1), axis=1)
                             for j in range(segments)], axis=1)
    result.append(case("known_peak", n=segments*bins, sig_rcv=signal, fs=fs, band=fs,
                       opts={"J": segments, "DiagLoad": 1e-3}, truth=truth))
    failures = [
        case("short_auto", n=4096, rcvPos=receivers*10, fs=100e6, opts={}, error="InsufficientDuration"),
        case("zero_energy", sig_rcv=np.zeros((4, 128)), error="ZeroEnergy"),
        case("zero_geometry", rcvPos=np.zeros((4, 2)), error="InvalidGeometry"),
        case("negative_edge", edge=-1., error="InvalidInput"),
        case("zero_step", lamda=0., error="InvalidInput"),
        case("zero_J", opts={"J": 0}, error="InvalidInput"),
        case("fractional_J", opts={"J": 1.5}, error="InvalidInput"),
        case("too_many_segments", opts={"J": 129}, error="InvalidSegments"),
        case("empty_bins", opts={"J": 8, "FrequencyRange": [1234., 1234.]}, error="EmptyFrequencySelection"),
        case("bad_range", opts={"J": 8, "FrequencyRange": [-2e6, 0.]}, error="InvalidInput"),
        case("bad_band", band=2e6, error="InvalidInput"),
        case("bad_shape", rcvPos=np.zeros((4, 3)), error="InvalidInput"),
        case("bad_station_count", sig_rcv=noise(128)[:3], error="InvalidInput"),
        case("unknown_option", opts={"Unknown": 1.}, error="InvalidOptions"),
        case("bad_mode", opts={"DiagLoadMode": "wrong"}, error="InvalidOptions"),
        case("negative_load", opts={"DiagLoad": -.1}, error="InvalidInput"),
        case("singular_unloaded", opts={"J": 1, "DiagLoad": 0.}, error="SingularCovariance"),
        case("nan_input", sig_rcv=np.full((4, 128), np.nan), error="InvalidInput"),
    ]
    return result + failures


MATLAB_DRIVER = r"""
root = '{root}'; folder = '{folder}'; addpath(root);
issues = checkcode(fullfile(root,'DPD_MVDR.m'),'-id');
check_ids = {{issues.id}}; check_lines = [issues.line];
save(fullfile(folder,'code_analysis.mat'),'check_ids','check_lines');
inputs = dir(fullfile(folder,'input_*.mat'));
for ii = 1:numel(inputs)
    c = load(fullfile(folder,inputs(ii).name));
    output = strrep(inputs(ii).name,'input_','output_');
    try
        if isfield(c,'omit_opts') && c.omit_opts
            [pos,mtr,info] = DPD_MVDR(c.rcvPos,c.sig_rcv,c.init_pos,c.edge,c.lamda,c.fs,c.band,c.fc);
        else
            [pos,mtr,info] = DPD_MVDR(c.rcvPos,c.sig_rcv,c.init_pos,c.edge,c.lamda,c.fs,c.band,c.fc,c.opts);
        end
        error_code = ''; reference = [];
        if isfield(c,'reference_fd')
            % 独立保留作者的显式对角矩阵和最终求逆表达式，M=1。
            L = size(c.rcvPos,1); J = size(c.reference_fd,3); K = numel(c.reference_f);
            Ri = zeros(L,L,K);
            for k = 1:K
                v = reshape(c.reference_fd(:,k,:),L,J);
                Ri(:,:,k) = inv(v*v'/J + .001*eye(L));
            end
            reference = zeros(numel(info.y_vec),numel(info.x_vec));
            for ix = 1:numel(info.x_vec)
                for iy = 1:numel(info.y_vec)
                    tau = sqrt(sum(([info.x_vec(ix),info.y_vec(iy)]-c.rcvPos).^2,2))/3e8;
                    D = zeros(L);
                    for k = 1:K
                        Lambda = diag(exp(-1i*2*pi*c.reference_f(k)*tau));
                        D = D + Lambda'*Ri(:,:,k)*Lambda;
                    end
                    reference(iy,ix) = max(real(eig(inv(D+.001*eye(L)))));
                end
            end
            reference = reference.';
        end
        save(fullfile(folder,output),'pos','mtr','info','reference','error_code');
    catch ME
        error_code = ME.identifier;
        save(fullfile(folder,output),'error_code');
    end
end
"""


def close(actual, expected, label, rtol=1e-8):
    actual, expected = np.asarray(actual), np.asarray(expected)
    scale = max(float(np.max(np.abs(expected))), np.finfo(float).tiny)
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=scale*1e-12, err_msg=label)
    return float(np.max(np.abs(actual-expected))/scale)


def run(folder, matlab, report):
    inputs = cases()
    results = {}
    for index, item in enumerate(inputs):
        payload = {k: v for k, v in item.items() if k not in ("name", "error", "truth")}
        savemat(folder/f"input_{index:03d}.mat", payload)
    driver = MATLAB_DRIVER.format(root=ROOT.as_posix().replace("'", "''"),
                                  folder=folder.as_posix().replace("'", "''"))
    driver_path = folder/"run_matlab.m"
    driver_path.write_text(driver, encoding="utf-8")
    env = os.environ.copy()
    env.update(PYTHONUTF8="1", PYTHONIOENCODING="utf-8:replace")
    command = "run('" + driver_path.as_posix().replace("'", "''") + "')"
    completed = subprocess.run([matlab, "-batch", command], cwd=ROOT, env=env,
                               encoding="utf-8", errors="replace", capture_output=True, timeout=180)
    (folder/"matlab.log").write_text((completed.stdout or "")+(completed.stderr or ""), encoding="utf-8")
    if completed.returncode:
        raise RuntimeError(f"MATLAB exited {completed.returncode}; see matlab.log")
    analysis = loadmat(folder/"code_analysis.mat", simplify_cells=True)
    report["matlab_code_analysis"] = dict(ids=np.asarray(analysis["check_ids"]).reshape(-1).tolist(),
                                          lines=np.asarray(analysis["check_lines"]).reshape(-1).tolist())
    assert not report["matlab_code_analysis"]["ids"], report["matlab_code_analysis"]
    report["cases"] = []
    for index, item in enumerate(inputs):
        actual = loadmat(folder/f"output_{index:03d}.mat", simplify_cells=True)
        args = [item[k] for k in ("rcvPos", "sig_rcv", "init_pos", "edge", "lamda", "fs", "band", "fc")]
        error = ""
        try:
            pos, spectrum, info = DPD_MVDR(*args, **({} if item.get("omit_opts") else {"opts": item["opts"]}))
        except DPD_MVDR_Error as exception:
            error = exception.code
        expected_error = "DPD_MVDR:"+item["error"] if item["error"] else ""
        matlab_error = str(actual["error_code"]) if np.asarray(actual["error_code"]).size else ""
        assert error == matlab_error == expected_error, (item["name"], error, matlab_error, expected_error)
        row = dict(name=item["name"], error_code=error)
        if not error:
            reference_spectrum = np.asarray(actual["mtr"]).reshape(spectrum.shape)
            row["spectrum_relative_error"] = close(spectrum, reference_spectrum, item["name"])
            np.testing.assert_array_equal(pos, actual["pos"])
            matlab_info = actual["info"]
            assert set(info) == set(matlab_info)
            for key, value in info.items():
                other = matlab_info[key]
                if isinstance(value, str):
                    assert value == other, (item["name"], key)
                elif key in ("frequency_indices", "peak_index", "peak_linear_index", "peak_tie_count",
                             "J", "N_fft", "N_total", "N_used", "N_discarded", "K", "margin_met",
                             "rank_bound", "rank_limited"):
                    np.testing.assert_array_equal(np.asarray(value).reshape(-1), np.asarray(other).reshape(-1))
                else:
                    close(np.asarray(value).reshape(-1), np.asarray(other).reshape(-1), key)
            if "reference_fd" in item:
                row["author_reference_relative_error"] = close(spectrum, actual["reference"], "author M=1")
            if "truth" in item:
                np.testing.assert_array_equal(pos, item["truth"])
            results[item["name"]] = (pos, spectrum, info)
        report["cases"].append(row)
        print("PASS", item["name"], flush=True)
    # 独立物理/工程不变量，避免仅证明两份实现共享同一错误。
    close(results["weak_scale"][1]/1e-18, results["even_full"][1], "amplitude scaling")
    close(results["receiver_permutation"][1], results["even_full"][1], "receiver order")
    close(results["high_carrier"][1], results["even_full"][1], "carrier invariance")
    assert results["odd_tail"][2]["N_discarded"] == 3
    assert results["dc_ties"][2]["peak_tie_count"] == 9
    np.testing.assert_array_equal(results["dc_ties"][2]["peak_index"], [1, 1])
    assert results["rank_limited"][2]["rank_limited"]
    assert not results["author_m1"][2]["margin_met"]
    assert results["decimal_grid"][1].shape == (7, 7)
    assert results["nondivisible_grid"][1].shape == (4, 4)
    report["status"] = "PASS"
    report["case_count"] = len(inputs)
    report["max_spectrum_relative_error"] = max(row.get("spectrum_relative_error", 0.) for row in report["cases"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matlab", default=shutil.which("matlab"))
    args = parser.parse_args()
    if not args.matlab:
        parser.error("MATLAB unavailable; provide --matlab. No dependencies are installed automatically.")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")+"_"+uuid.uuid4().hex[:8]
    folder = ROOT/"outputs_e2e"/"verification"/"dpd_mvdr"/stamp
    expected = (ROOT/"outputs_e2e").resolve()
    if not folder.resolve().is_relative_to(expected) or folder.exists():
        raise RuntimeError("Invalid or nonempty verification destination")
    folder.mkdir(parents=True, exist_ok=False)
    report = dict(status="FAIL", scope="synthetic engineering checks, not localization performance",
                  spectrum_rtol=1e-8, spectrum_atol="1e-12 * maximum reference magnitude",
                  source_sha256={name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                                 for name in ("DPD_MVDR.m", "DPD_MVDR.py", "DPD_MVDR_Offical.m", "verify_DPD_MVDR.py")})
    try:
        run(folder, args.matlab, report)
    except Exception:
        report["failure"] = traceback.format_exc()
        raise
    finally:
        (folder/"report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print("REPORT", folder/"report.json", flush=True)


if __name__ == "__main__":
    main()
