function mtr_batch = MVDR_calculator_gpu_batch(rcvPos, sig_td_batch, init_pos, edge, lamda, fs, J_seg)
% MVDR_calculator_gpu_batch  一次性计算多个子带的MVDR空间谱（GPU加速）
%
% 与 DPD_calculator_gpu_batch 完全对应的MVDR版本：
%   ML-DPD: R(m1,m2,p) = Σ_f X_m1(f)·X_m2*(f)·exp(j2πfΔτ)  → max_eig(R)
%   MVDR:   S(m1,m2,p) = Σ_f R_inv(m1,m2,f)·exp(j2πfΔτ)    → 1/min_eig(S)
%
%   GPU网格搜索结构完全相同，仅"互谱"来源和特征值提取方式不同：
%     ML-DPD: 互谱 = X_m1(f)·conj(X_m2(f))，取 max eigenvalue
%     MVDR:   互谱 = R_inv(m1,m2,f)，取 1/min eigenvalue
%
% 输入：
%   rcvPos:       [rcvNum × 2]         接收站二维坐标
%   sig_td_batch: [N_batch × rcvNum × len]  时域子带信号（不归一化）
%                 也支持 [rcvNum × len] 单子带输入
%   init_pos:     [1 × 2] 搜索区域中心
%   edge:         搜索半边长 (m)
%   lamda:        网格间距 (m)
%   fs:           采样率 (Hz)
%   J_seg:        分段数（默认4，需 ≥ rcvNum 保证协方差满秩）
%
% 输出：
%   mtr_batch:    [N_batch × num_x × num_y] 各子带MVDR空间谱
%                 单子带输入时返回 [num_x × num_y]

vc = 299792458;
if nargin < 7, J_seg = 4; end

%% ═══════════════════════════════════════
%  输入维度处理
%% ═══════════════════════════════════════
if ndims(sig_td_batch) == 2
    sig_td_batch = reshape(sig_td_batch, 1, size(sig_td_batch,1), size(sig_td_batch,2));
    single_mode = true;
else
    single_mode = false;
end

[N_batch, rcvNum, len] = size(sig_td_batch);

% 网格
x_vec  = init_pos(1)-edge : lamda : init_pos(1)+edge;
y_vec  = init_pos(2)-edge : lamda : init_pos(2)+edge;
num_x  = length(x_vec);
num_y  = length(y_vec);
num_grid = num_x * num_y;

% 分段参数
K_seg = floor(len / J_seg);
f_seg = (-K_seg/2 : K_seg/2-1) * (fs / K_seg);

% 有效频率范围: 信号已预先搬移至中心频率为0，取 ±B_win/2
% 调用前需确保每个子带信号已搬移到基带中心
B_win = 10e6;
valid_idx = find(abs(f_seg) <= B_win / 2);
N_valid = length(valid_idx);
f_valid = f_seg(valid_idx);

%% ═══════════════════════════════════════
%  Step 1: 分段FFT + 逐频率协方差估计 + 求逆
%  R_inv_all: (rcvNum, rcvNum, N_valid, N_batch)
%% ═══════════════════════════════════════
R_inv_all = zeros(rcvNum, rcvNum, N_valid, N_batch);

for b = 1:N_batch
    sig_b = squeeze(sig_td_batch(b, :, :));   % (rcvNum, len)

    % 分段 + FFT
    seg_fft = zeros(rcvNum, K_seg, J_seg);
    for j = 1:J_seg
        idx_s = (j-1)*K_seg + 1;
        idx_e = j*K_seg;
        seg_fft(:, :, j) = fftshift(fft(sig_b(:, idx_s:idx_e), K_seg, 2), 2);
    end

    % 逐有效频率bin: 协方差 → 对角加载 → 求逆
    for fi = 1:N_valid
        fk = valid_idx(fi);
        R_hat = zeros(rcvNum);
        for j = 1:J_seg
            xf = seg_fft(:, fk, j);
            R_hat = R_hat + xf * xf';
        end
        R_hat = R_hat / J_seg;

        dl = (trace(real(R_hat)) / rcvNum) * 1e-6;
        R_hat_load = R_hat + dl * eye(rcvNum);
        R_inv_all(:,:,fi,b) = inv(R_hat_load);
    end
end

%% ═══════════════════════════════════════
%  Step 2: 从 R_inv 提取"对角项"和"互谱"（和ML-DPD结构对应）
%
%  对角项: diag_vals(m, b) = Σ_f R_inv(m,m,f)     ← 与网格无关
%  互谱:   cs_mat(f, pair, b) = R_inv(m1,m2,f)    ← 替代 X_m1·conj(X_m2)
%% ═══════════════════════════════════════
diag_vals = zeros(rcvNum, N_batch);
for b = 1:N_batch
    for m = 1:rcvNum
        diag_vals(m, b) = real(sum(squeeze(R_inv_all(m, m, :, b))));
    end
end

nPairs  = rcvNum * (rcvNum - 1) / 2;
pair_m1 = zeros(1, nPairs);
pair_m2 = zeros(1, nPairs);
cs_mat  = zeros(N_valid, nPairs, N_batch);

idx = 0;
for m1 = 1:rcvNum
    for m2 = m1+1:rcvNum
        idx = idx + 1;
        pair_m1(idx) = m1;
        pair_m2(idx) = m2;
        for b = 1:N_batch
            cs_mat(:, idx, b) = squeeze(R_inv_all(m1, m2, :, b));
        end
    end
end

%% ═══════════════════════════════════════
%  Step 3: 网格时延（CPU，仅依赖几何，所有子带共享）
%  与 DPD_calculator_gpu_batch 完全相同
%% ═══════════════════════════════════════
[xg, yg] = meshgrid(x_vec, y_vec);
grid_x = xg(:).';
grid_y = yg(:).';

taus = zeros(num_grid, rcvNum);
for m = 1:rcvNum
    taus(:,m) = sqrt((grid_x.' - rcvPos(m,1)).^2 + ...
                     (grid_y.' - rcvPos(m,2)).^2) / vc;
end

%% ═══════════════════════════════════════
%  Step 4: GPU批量构造 S_all 矩阵
%  S_all: (rcvNum, rcvNum, num_grid * N_batch)
%  结构与 DPD_calculator_gpu_batch 的 R_all 完全平行
%  唯一区别: "互谱"来源是 R_inv 而非原始互谱
%% ═══════════════════════════════════════
g_taus     = gpuArray(taus);
g_cs_mat   = gpuArray(cs_mat);          % (N_valid, nPairs, N_batch)
g_f        = gpuArray(f_valid);         % (1, N_valid)
g_TWO_PI_F = 2j * pi * g_f;

total_pts = num_grid * N_batch;
S_all = complex(zeros(rcvNum, rcvNum, total_pts, 'double', 'gpuArray'));

% ── 对角线: 与网格无关，直接扩展 ──
for m = 1:rcvNum
    diag_expand = repmat(gpuArray(diag_vals(m, :)), num_grid, 1);
    S_all(m, m, :) = reshape(diag_expand, 1, 1, []);
end

% ── 非对角线: phase_mat × R_inv互谱（核心矩乘，和ML-DPD完全同构）──
for ip = 1:nPairs
    m1 = pair_m1(ip);
    m2 = pair_m2(ip);

    dtau      = g_taus(:, m1) - g_taus(:, m2);    % (num_grid, 1)
    phase_mat = exp(dtau * g_TWO_PI_F);             % (num_grid, N_valid)

    cs_pair = reshape(g_cs_mat(:, ip, :), N_valid, N_batch);  % (N_valid, N_batch)

    val = phase_mat * cs_pair;   % (num_grid, N_batch)

    S_all(m1, m2, :) = reshape(val,       1, 1, []);
    S_all(m2, m1, :) = reshape(conj(val), 1, 1, []);
end

%% ═══════════════════════════════════════
%  Step 5: pageeig 批量求特征值 → 取最小特征值
%  ML-DPD: max(abs(D))   → 直接输出
%  MVDR:   min(real(D))  → 取倒数输出
%% ═══════════════════════════════════════
S_cpu   = gather(S_all);
% 强制Hermitian对称（消除数值误差）
S_cpu   = (S_cpu + conj(permute(S_cpu, [2,1,3]))) / 2;
D       = pageeig(S_cpu);                  % (rcvNum, total_pts)
min_eig = min(real(D), [], 1);             % (1, total_pts)

% S是正定矩阵，min_eig理论上>0，clamp防数值误差
min_eig = max(min_eig, 1e-30);

% reshape
min_eig_mat = reshape(min_eig, num_grid, N_batch);   % (num_grid, N_batch)

%% ═══════════════════════════════════════
%  Step 6: 输出 = 1/min_eig
%% ═══════════════════════════════════════
if single_mode
    mtr_batch = reshape(1 ./ min_eig_mat(:,1), num_y, num_x).';
else
    mtr_batch = zeros(N_batch, num_x, num_y);
    for b = 1:N_batch
        mtr_batch(b, :, :) = reshape(1 ./ min_eig_mat(:,b), num_y, num_x).';
    end
end

end