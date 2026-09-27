function [pos, mtr, info] = DPD_MVDR(rcvPos, sig_rcv, init_pos, edge, lamda, fs, band, fc, opts)
%DPD_MVDR 每站单天线的二维 DPD-HR 谱（Tirer & Weiss，式27）。
% rcvPos: L×2，米；sig_rcv: L×N，同步时域复基带 IQ。
% init_pos: 中心坐标；edge: 搜索半宽（米）；lamda: 网格步长（米）。
% fs/band/fc: Hz。默认选择 [-band/2, band/2] 内 FFT 频点。
% opts 可省略：J=[]（自动），SegmentMargin=40，DiagLoadMode='relative'，
% DiagLoad=1e-6，SpectrumLoad=0，FrequencyRange=[]，
% PropagationSpeed=299792458，PeakTieTolerance=1e-10。
% 显式 J 允许低时延裕度，但必须检查 info.margin_met/rank_limited。
% relative 加载为 DiagLoad*trace(R)/L；absolute 为 DiagLoad。
% 作者归一化频域数据的对照设置为 absolute/0.001/SpectrumLoad=0.001。
% FFT 不归一化；尾部不足一段的样本丢弃并报告。无窗函数或补零。
% pos 仅为全局峰坐标，不是完整多源输出；mtr(ix,iy) 保持原布局。
% 相对峰差 <= PeakTieTolerance 时，按 MATLAB 列优先顺序取第一个。
% info 中所有索引均为1起始，Python版本也遵循此约定。
% 细节和双语言检查入口见 DPD_MVDR说明.md。
if nargin < 9, opts = struct(); end
defaults = struct('J', [], 'SegmentMargin', 40, 'DiagLoadMode', 'relative', ...
    'DiagLoad', 1e-6, 'SpectrumLoad', 0, 'FrequencyRange', [], ...
    'PropagationSpeed', 299792458, 'PeakTieTolerance', 1e-10);
if ~isstruct(opts) || ~isscalar(opts)
    fail('InvalidOptions', 'opts must be a scalar struct.');
end
names = fieldnames(opts);
for i = 1:numel(names)
    if ~isfield(defaults, names{i}), fail('InvalidOptions', 'Unknown option.'); end
    defaults.(names{i}) = opts.(names{i});
end
opts = defaults;
real_array(rcvPos, 'rcvPos'); real_array(init_pos, 'init_pos');
if ~ismatrix(rcvPos) || size(rcvPos,2) ~= 2 || size(rcvPos,1) < 2 || numel(init_pos) ~= 2
    fail('InvalidInput', 'Expected L-by-2 receivers (L>=2) and a 2-element center.');
end
if ~isnumeric(sig_rcv) || ~ismatrix(sig_rcv) || any(~isfinite(sig_rcv(:))) || ...
        size(sig_rcv,1) ~= size(rcvPos,1) || size(sig_rcv,2) < 2
    fail('InvalidInput', 'Expected finite L-by-N IQ, N>=2.');
end
scalar_range(edge, 0, false); scalar_range(lamda, 0, true);
scalar_range(fs, 0, true); scalar_range(band, 0, true); real_array(fc, 'fc');
if ~isscalar(fc) || band > fs, fail('InvalidInput', 'Require scalar fc and band<=fs.'); end
scalar_range(opts.SegmentMargin, 0, true);
scalar_range(opts.PropagationSpeed, 0, true);
scalar_range(opts.DiagLoad, 0, false); scalar_range(opts.SpectrumLoad, 0, false);
scalar_range(opts.PeakTieTolerance, 0, false);
if opts.PeakTieTolerance >= 1, fail('InvalidOptions', 'PeakTieTolerance must be <1.'); end
if ~(ischar(opts.DiagLoadMode) && isrow(opts.DiagLoadMode)) && ...
        ~(isstring(opts.DiagLoadMode) && isscalar(opts.DiagLoadMode))
    fail('InvalidOptions', 'DiagLoadMode must be relative or absolute.');
end
mode = lower(char(opts.DiagLoadMode));
if ~ismember(mode, {'relative','absolute'})
    fail('InvalidOptions', 'DiagLoadMode must be relative or absolute.');
end
rcvPos = double(rcvPos); sig_rcv = double(sig_rcv); init_pos = double(init_pos(:).');
edge = double(edge); lamda = double(lamda); fs = double(fs); band = double(band); fc = double(fc);
numeric_options = {'SegmentMargin','PropagationSpeed','DiagLoad','SpectrumLoad','PeakTieTolerance'};
for i = 1:numel(numeric_options), opts.(numeric_options{i}) = double(opts.(numeric_options{i})); end
[L,N] = size(sig_rcv);
baseline = 0;
for i = 1:L-1
    for j = i+1:L, baseline = max(baseline, norm(rcvPos(i,:)-rcvPos(j,:))); end
end
if ~isfinite(baseline) || baseline <= 0, fail('InvalidGeometry', 'Receiver baseline must be positive and finite.'); end
delay_bound = baseline / opts.PropagationSpeed;
if isempty(opts.J)
    minimum_samples = max(2, ceil(fs * opts.SegmentMargin * delay_bound));
    J = floor(N / minimum_samples);
    if J < 1, fail('InsufficientDuration', 'No segment meets SegmentMargin; provide validated J or longer observations.'); end
else
    scalar_range(opts.J, 1, false);
    J = double(opts.J);
    if J ~= floor(J), fail('InvalidInput', 'J must be an integer.'); end
end
K = floor(N / J);
if K < 2, fail('InvalidSegments', 'Each segment must contain at least two samples.'); end
f = (-floor(K/2):ceil(K/2)-1) * (fs/K);
if isempty(opts.FrequencyRange)
    frequency_range = [-band/2,band/2];
else
    real_array(opts.FrequencyRange, 'FrequencyRange');
    if numel(opts.FrequencyRange) ~= 2, fail('InvalidInput', 'FrequencyRange must have two elements.'); end
    frequency_range = double(opts.FrequencyRange(:).');
    if frequency_range(1) > frequency_range(2) || frequency_range(1) < -fs/2 || frequency_range(2) > fs/2
        fail('InvalidInput', 'Invalid FrequencyRange.');
    end
end
ids = find(f >= frequency_range(1) & f <= frequency_range(2));
if isempty(ids), fail('EmptyFrequencySelection', 'No FFT bins in the requested range.'); end
frequencies = f(ids); F = numel(ids);
xf = complex(zeros(L,F,J));
for j = 1:J
    segment = fftshift(fft(sig_rcv(:,(j-1)*K+1:j*K),K,2),2);
    xf(:,:,j) = segment(:,ids);
end
if any(~isfinite(xf(:))), fail('NumericalFailure', 'Non-finite FFT.'); end
if ~any(xf(:) ~= 0), fail('ZeroEnergy', 'Selected observations have zero energy.'); end
inverse_cov = complex(zeros(L,L,F)); loads = zeros(1,F); ratios = zeros(1,F);
for k = 1:F
    R = complex(zeros(L));
    for j = 1:J
        v = xf(:,k,j); R = R + v*v';
    end
    R = R / J; R = (R+R')/2;
    if strcmp(mode,'relative'), loads(k) = opts.DiagLoad * real(trace(R))/L;
    else, loads(k) = opts.DiagLoad; end
    loaded = R + loads(k)*eye(L);
    if any(~isfinite(loaded(:))), fail('NumericalFailure', 'Non-finite covariance.'); end
    ev = real(eig(loaded)); ratios(k) = min(ev)/max(ev);
    if max(ev) <= 0 || ~isfinite(ratios(k)) || ratios(k) <= 64*eps
        fail('SingularCovariance', 'Covariance not numerically positive definite; review snapshots and loading.');
    end
    inverse_cov(:,:,k) = loaded \ eye(L);
end
grid_ratio = 2*edge/lamda;
intervals = floor(grid_ratio + 8*eps*max(1,abs(grid_ratio)));
if ~isfinite(intervals), fail('InvalidInput', 'Non-finite grid size.'); end
x_vec = init_pos(1)-edge + (0:intervals)*lamda;
y_vec = init_pos(2)-edge + (0:intervals)*lamda;
mtr = zeros(numel(x_vec),numel(y_vec));
for ix = 1:numel(x_vec)
    for iy = 1:numel(y_vec)
        tau = sqrt(sum(([x_vec(ix),y_vec(iy)]-rcvPos).^2,2))/opts.PropagationSpeed;
        S = complex(zeros(L));
        for k = 1:F
            a = exp(-1i*2*pi*(fc+frequencies(k))*tau);
            S = S + inverse_cov(:,:,k).*(conj(a)*a.');
        end
        S = (S+S')/2;
        if any(~isfinite(S(:))), fail('NumericalFailure', 'Non-finite position matrix.'); end
        ev = real(eig(S)); denominator = min(ev)+opts.SpectrumLoad;
        if denominator <= 64*eps*(max(ev)+opts.SpectrumLoad) || ~isfinite(denominator)
            fail('InvalidSpectrum', 'Position matrix not numerically positive definite.');
        end
        mtr(ix,iy) = 1/denominator;
    end
end
if any(~isfinite(mtr(:))) || any(mtr(:) <= 0), fail('InvalidSpectrum', 'Spectrum must be finite and positive.'); end
maximum = max(mtr(:));
ties = find(mtr(:) >= maximum-opts.PeakTieTolerance*abs(maximum));
[ix,iy] = ind2sub(size(mtr),ties(1)); pos = [x_vec(ix),y_vec(iy)];
margin = (K/fs)/delay_bound;
info = struct('J',J,'N_total',N,'N_used',J*K,'N_discarded',N-J*K, ...
    'N_fft',K,'K',F,'frequencies_hz',frequencies,'frequency_indices',ids, ...
    'frequency_range_hz',frequency_range,'x_vec',x_vec,'y_vec',y_vec, ...
    'baseline_m',baseline,'delay_bound_s',delay_bound,'segment_duration_s',K/fs, ...
    'segment_margin',margin,'required_margin',opts.SegmentMargin, ...
    'margin_met',margin>=opts.SegmentMargin,'rank_bound',min(L,J),'rank_limited',J<L, ...
    'diag_load_per_frequency',loads,'cov_eigenvalue_ratio',ratios, ...
    'diag_load_mode',mode,'diag_load_value',opts.DiagLoad,'spectrum_load',opts.SpectrumLoad, ...
    'propagation_speed',opts.PropagationSpeed,'fc_hz',fc,'fft_normalization','none', ...
    'peak_index',[ix,iy],'peak_linear_index',ties(1),'peak_tie_count',numel(ties), ...
    'peak_tie_tolerance',opts.PeakTieTolerance,'peak_value',mtr(ix,iy),'maximum_value',maximum);
if info.margin_met, info.status = 'OK'; else, info.status = 'LOW_SEGMENT_MARGIN'; end
end

function real_array(value, name)
if ~isnumeric(value) || ~isreal(value) || isempty(value) || any(~isfinite(value(:)))
    fail('InvalidInput', [name ' must be finite real numeric values.']);
end
end

function scalar_range(value, lower, strict)
real_array(value, 'Scalar');
if ~isscalar(value) || value < lower || (strict && value == lower)
    fail('InvalidInput', 'Scalar out of range.');
end
end

function fail(code, message)
error(['DPD_MVDR:' code], '%s', message);
end
