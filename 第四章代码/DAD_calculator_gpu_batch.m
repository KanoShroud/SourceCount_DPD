function mtr_batch = DAD_calculator_gpu_batch(rcvPos, sig_rcv_batch, fs, init_angle, lamda, range)
% DAD_calculator_gpu_batch  一次性计算多个子带的 DAD 角度谱
%
% ═══════════════════════════════════════════════════════════════════
%  从 DAD_calculator_spec → DAD_calculator_gpu_batch 的核心优化
%  （与 DPD_calculator_spec → DPD_calculator_gpu_batch 完全平行）
%
%  原始版本瓶颈（DAD_calculator_spec）：
%    - 外层 for 逐角度，内层 for 逐接收站 → 双重循环
%    - eigs 每次迭代求解，效率低
%    - 多子带需反复调用，GPU 显存频繁分配/释放
%
%  本版本优化：
%    1. gpuArray 只分配一次，避免 N_batch 次重复分配/释放
%    2. 时延-相位矩阵 phase_mat 只算一次（仅依赖几何关系，与子带无关）
%    3. 矩阵乘法 phase_mat * cs_pair 一次算完所有子带的非对角互谱
%    4. pageeig 一次性处理 [rcvNum×rcvNum×(num_angle×N_batch)] 批量特征分解
%
%  数学说明（与 DPD 的对应关系）：
%    DPD: time_delay(m, grid_k)  = norm(grid_k - rcvPos(m,:)) / vc
%    DAD: time_delay(m, angle_k) = dot(-[cos(θ_k),sin(θ_k)],
%                                      rcvPos(m,:)-barycenter) / vc
%    两者相关矩阵 R(m1,m2) = Σ_f cs(m1,m2,f)·exp(2jπf·(τ_m1-τ_m2))
%    结构完全相同，仅几何计算方式不同。
% ═══════════════════════════════════════════════════════════════════
%
% 输入：
%   rcvPos:         [rcvNum × 2]            接收站二维坐标（米）
%   sig_rcv_batch:  [N_batch × rcvNum × N0] 多个子带的频域信号
%                   也支持 [rcvNum × N0] 单子带输入（自动扩展为 batch=1）
%   fs:             采样率（Hz）
%   init_angle:     搜索中心角度（度）
%   lamda:          角度搜索步长（度）
%   range:          搜索半角度范围（度），搜索区间 = [init_angle-range, init_angle+range]
%
% 输出：
%   mtr_batch:      [N_batch × num_angle]   各子带 DAD 角度谱
%                   单子带输入时返回 [1 × num_angle]
%
% 调用示例（多子带）：
%   mtr = DAD_calculator_gpu_batch(rcvPos, sig_batch, fs, 90, 0.1, 45);
%   [~, best_idx] = max(mtr(1,:));
%   best_aoa = (90 - 45) + (best_idx-1) * 0.1;
%
% 调用示例（单子带，兼容原版接口）：
%   mtr = DAD_calculator_gpu_batch(rcvPos, sig_rcv, fs, 90, 0.1, 45);

vc = 299792458;

%% ═══════════════════════════════════════
%  输入维度处理（兼容单子带 [rcvNum×N0]）
%% ═══════════════════════════════════════
if ndims(sig_rcv_batch) == 2
    % 单子带输入 [rcvNum × N0] → [1 × rcvNum × N0]
    sig_rcv_batch = reshape(sig_rcv_batch, 1, size(sig_rcv_batch,1), size(sig_rcv_batch,2));
    single_mode = true;
else
    single_mode = false;
end

[N_batch, rcvNum, N0] = size(sig_rcv_batch);

% 角度搜索向量（与原版一致）
angle_vec = init_angle - range : lamda : init_angle + range;
num_angle = length(angle_vec);

% 频率轴（fftshift 对应，与原版 -fs/2:fs/nfft:fs/2-1 等价）
f = (-N0/2 : N0/2-1) * (fs / N0);   % [1 × N0]

%% ═══════════════════════════════════════
%  Step 1: 预计算互谱（CPU，数据量小）
%  对应原版：V(m,:) = sig_rcv(m,:) .* exp(...)  →  V*V'
%  这里将 V*V' 分解为对角项（自功率）和非对角项（互谱）
%% ═══════════════════════════════════════

% ── 对角线：各接收站自功率 (rcvNum × N_batch)，与角度无关 ──
diag_vals = zeros(rcvNum, N_batch);
for b = 1:N_batch
    sig_b = squeeze(sig_rcv_batch(b, :, :));   % (rcvNum, N0)
    diag_vals(:, b) = real(sum(abs(sig_b).^2, 2));
end

% ── 非对角线：站对互谱 (N0 × nPairs × N_batch) ──
nPairs  = rcvNum * (rcvNum - 1) / 2;
pair_m1 = zeros(1, nPairs);
pair_m2 = zeros(1, nPairs);
cs_mat  = zeros(N0, nPairs, N_batch);

idx = 0;
for m1 = 1:rcvNum
    for m2 = m1+1:rcvNum
        idx = idx + 1;
        pair_m1(idx) = m1;
        pair_m2(idx) = m2;
        for b = 1:N_batch
            % 互谱 = sig_m1 .* conj(sig_m2)（频域）
            cs_mat(:, idx, b) = squeeze(sig_rcv_batch(b, m1, :)) .* ...
                                conj(squeeze(sig_rcv_batch(b, m2, :)));
        end
    end
end

%% ═══════════════════════════════════════
%  Step 2: 预计算各角度各接收站时延（CPU，仅依赖几何，所有子带共享）
%
%  对应原版：
%    barycenter = mean(rcvPos)
%    aoa_dir = -[cos(θ), sin(θ)]
%    time_delay(m) = dot(aoa_dir, rcvPos(m,:)-barycenter) / vc
%% ═══════════════════════════════════════
barycenter      = mean(rcvPos, 1);                  % [1 × 2]
rcv2baryctr_vec = rcvPos - barycenter;              % [rcvNum × 2]

% taus(k, m)：第 k 个角度下第 m 个接收站的时延
taus = zeros(num_angle, rcvNum);
for k = 1:num_angle
    theta   = angle_vec(k) * pi / 180;
    aoa_dir = -[cos(theta), sin(theta)];            % [1 × 2]
    for m = 1:rcvNum
        taus(k, m) = dot(aoa_dir, rcv2baryctr_vec(m,:)) / vc;
    end
end

%% ═══════════════════════════════════════
%  Step 3: GPU 批量构造相关矩阵
%
%  R_all 维度: [rcvNum × rcvNum × (num_angle × N_batch)]
%  排列顺序: angle 变化快（内层），batch 变化慢（外层）
%  与 DPD_calculator_gpu_batch 中 grid 变化快、batch 变化慢完全对应
%% ═══════════════════════════════════════
g_taus     = gpuArray(taus);           % (num_angle, rcvNum)
g_cs_mat   = gpuArray(cs_mat);         % (N0, nPairs, N_batch)
g_f        = gpuArray(f);              % (1, N0)
g_TWO_PI_F = 2j * pi * g_f;           % (1, N0)

total_pts = num_angle * N_batch;
R_all = complex(zeros(rcvNum, rcvNum, total_pts, 'double', 'gpuArray'));

% ── 对角线：各角度自功率（与角度无关，直接 repmat 扩展）──
for m = 1:rcvNum
    % (num_angle, N_batch)：每个 batch 的自功率在所有角度点上相同
    diag_expand = repmat(gpuArray(diag_vals(m, :)), num_angle, 1);
    R_all(m, m, :) = reshape(diag_expand, 1, 1, []);
end

% ── 非对角线：phase_mat 仅算一次，乘以所有子带的互谱 ──
%
%  R(m1,m2,angle_k) = Σ_f cs(m1,m2,f) · exp(2jπf·(τ_{m1,k}-τ_{m2,k}))
%                   = phase_mat(k,:) · cs_pair(:,b)   （矩乘形式）
%
%  phase_mat: (num_angle, N0)  ← 只与几何相关，所有子带共享（核心优化）
%  cs_pair:   (N0, N_batch)    ← 与子带相关
%  val:        (num_angle, N_batch) ← 一次矩乘算完所有角度和子带

for ip = 1:nPairs
    m1 = pair_m1(ip);
    m2 = pair_m2(ip);

    dtau      = g_taus(:, m1) - g_taus(:, m2);    % (num_angle, 1)
    phase_mat = exp(dtau * g_TWO_PI_F);             % (num_angle, N0)

    % 该站对所有子带的互谱: (N0, N_batch)
    cs_pair = reshape(g_cs_mat(:, ip, :), N0, N_batch);

    % 一次矩乘算完所有角度×所有子带: (num_angle, N_batch)
    val = phase_mat * cs_pair;

    % 写入 R_all，顺序与对角线一致（angle 快，batch 慢）
    R_all(m1, m2, :) = reshape(val,       1, 1, []);
    R_all(m2, m1, :) = reshape(conj(val), 1, 1, []);
end

%% ═══════════════════════════════════════
%  Step 4: pageeig 批量求特征值
%  一次性对 total_pts = num_angle × N_batch 个 [rcvNum×rcvNum] 矩阵
%  做完整特征分解，替换原版逐角度调用 eigs
%% ═══════════════════════════════════════
R_cpu   = gather(R_all);                   % 从 GPU 取回
D       = pageeig(R_cpu);                  % (rcvNum, total_pts)
max_eig = max(abs(D), [], 1);              % (1, total_pts)

% reshape: angle 变化快，batch 变化慢 → (num_angle, N_batch)
max_eig_mat = reshape(max_eig, num_angle, N_batch);

%% ═══════════════════════════════════════
%  Step 5: 整理输出
%% ═══════════════════════════════════════
% 转置为 (N_batch, num_angle)，batch 为行，角度为列
mtr_batch = max_eig_mat.';

% 单子带时维持 (1 × num_angle)，与原版 mtr 形状一致
% （如需与原版完全兼容，可在外部用 [~,idx]=max(mtr_batch); aoa=angle_vec(idx)）

end