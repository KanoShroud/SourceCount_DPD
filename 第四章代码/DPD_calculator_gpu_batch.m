function mtr_batch = DPD_calculator_gpu_batch(rcvPos, sig_rcv_batch, init_pos, edge, lamda, fs)
% DPD_calculator_gpu_batch  一次性计算多个子带的DPD空间谱
%
% 核心优化（相比逐子带调用 DPD_calculator_gpu）：
%   1. gpuArray 只分配一次，避免 N_batch 次重复分配/释放
%   2. 相位补偿矩阵 phase_mat 只算一次（仅依赖几何关系，与子带无关）
%   3. 矩阵乘法 phase_mat * cs_batch 一次算完所有子带
%   4. pageeig 一次性处理所有子带×所有网格点的相关矩阵
%
% 输入：
%   rcvPos:        [rcvNum × 2] 接收站二维坐标
%   sig_rcv_batch: [N_batch × rcvNum × N0] 多个子带的频域信号
%                  也支持 [rcvNum × N0] 单子带输入（自动扩展）
%   init_pos:      [1 × 2] 搜索区域中心
%   edge:          搜索区域半边长 (m)
%   lamda:         网格间距 (m)
%   fs:            采样率 (Hz)
%
% 输出：
%   mtr_batch:     [N_batch × num_x × num_y] 各子带DPD空间谱
%                  单子带输入时返回 [num_x × num_y]

vc = 299792458;

%% ═══════════════════════════════════════
%  输入维度处理
%% ═══════════════════════════════════════
if ndims(sig_rcv_batch) == 2
    % 单子带输入 (rcvNum × N0) → (1 × rcvNum × N0)
    sig_rcv_batch = reshape(sig_rcv_batch, 1, size(sig_rcv_batch,1), size(sig_rcv_batch,2));
    single_mode = true;
else
    single_mode = false;
end

[N_batch, rcvNum, N0] = size(sig_rcv_batch);

x_vec  = init_pos(1)-edge : lamda : init_pos(1)+edge;
y_vec  = init_pos(2)-edge : lamda : init_pos(2)+edge;
num_x  = length(x_vec);
num_y  = length(y_vec);
num_grid = num_x * num_y;

f = (-N0/2 : N0/2-1) * (fs/N0);

%% ═══════════════════════════════════════
%  Step 1: 预计算互谱（CPU，数据量小）
%% ═══════════════════════════════════════

% 对角线: (rcvNum, N_batch)
diag_vals = zeros(rcvNum, N_batch);
for b = 1:N_batch
    sig_b = squeeze(sig_rcv_batch(b, :, :));   % (rcvNum, N0)
    diag_vals(:, b) = real(sum(abs(sig_b).^2, 2));
end

% 非对角线站对: (N0, nPairs, N_batch)
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
            cs_mat(:, idx, b) = squeeze(sig_rcv_batch(b, m1, :)) .* ...
                                conj(squeeze(sig_rcv_batch(b, m2, :)));
        end
    end
end

%% ═══════════════════════════════════════
%  Step 2: 网格时延（CPU，仅依赖几何，所有子带共享）
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
%  Step 3: GPU批量构造相关矩阵
%  R_all 维度: (rcvNum, rcvNum, num_grid * N_batch)
%  排列顺序: grid变化快, batch变化慢
%% ═══════════════════════════════════════
g_taus     = gpuArray(taus);
g_cs_mat   = gpuArray(cs_mat);           % (N0, nPairs, N_batch)
g_f        = gpuArray(f);
g_TWO_PI_F = 2j * pi * g_f;

total_pts = num_grid * N_batch;
R_all = complex(zeros(rcvNum, rcvNum, total_pts, 'double', 'gpuArray'));

% 对角线: 每个子带在所有网格点上的值相同
for m = 1:rcvNum
    % (num_grid, N_batch) → reshape 按列展开 → grid变化快
    diag_expand = repmat(gpuArray(diag_vals(m,:)), num_grid, 1);
    R_all(m, m, :) = reshape(diag_expand, 1, 1, []);
end

% 非对角线: phase_mat 只算一次（仅依赖几何），乘以所有子带的互谱
for ip = 1:nPairs
    m1 = pair_m1(ip);
    m2 = pair_m2(ip);

    dtau      = g_taus(:, m1) - g_taus(:, m2);     % (num_grid, 1)
    phase_mat = exp(dtau * g_TWO_PI_F);              % (num_grid, N0) ← 所有子带共享

    % 取该站对所有子带的互谱: (N0, N_batch)
    cs_pair = reshape(g_cs_mat(:, ip, :), N0, N_batch);

    % 一次矩乘算完所有子带: (num_grid, N0) × (N0, N_batch) → (num_grid, N_batch)
    val = phase_mat * cs_pair;

    % reshape 后写入 R_all，排列顺序与对角线一致
    R_all(m1, m2, :) = reshape(val, 1, 1, []);
    R_all(m2, m1, :) = reshape(conj(val), 1, 1, []);
end

%% ═══════════════════════════════════════
%  Step 4: pageeig 批量求特征值
%% ═══════════════════════════════════════
R_cpu   = gather(R_all);
D       = pageeig(R_cpu);
max_eig = max(abs(D), [], 1);           % (1, total_pts)

% reshape: grid变化快, batch变化慢
max_eig_mat = reshape(max_eig, num_grid, N_batch);   % (num_grid, N_batch)

%% ═══════════════════════════════════════
%  Step 5: 输出
%% ═══════════════════════════════════════
if single_mode
    mtr_batch = reshape(max_eig_mat(:,1), num_y, num_x).';
else
    mtr_batch = zeros(N_batch, num_x, num_y);
    for b = 1:N_batch
        mtr_batch(b, :, :) = reshape(max_eig_mat(:,b), num_y, num_x).';
    end
end

end