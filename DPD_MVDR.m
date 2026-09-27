function [pos,mtr] = DPD_MVDR(rcvPos,sig_rcv,init_pos,edge,lamda,fs,band,fc)
% function [pos,mtr] = DPD_MVDR(rcvPos,sig_rcv,init_pos,edge,lamda,fs)
%--------------------------------------------------------------------------
% DPD Algorithm Based on Minimun-Variance-Distortionless-Response( 2D )
% Input : 
%       rcvPos:     NxM, positions of reciving sensors, each row is a sensor 
%                   position;
%       sig_rcv:    NxK, time region baseband signal;
%       init_pos:   1xM, The midpoint of the grid;
%       edge:       The length of the grid boundary;
%       lamda:      The steps of grid search;
%       fs:         Sample rate;
%       band:       Signal bandwidth;
% Output :
%       pos:        **currently not coded**
%       mtr:        cost function matrix, the dimension depends on the step;
%
% Revised on 2026/03/06 in Xidian University
%--------------------------------------------------------------------------
vc = 299792458;

x_vec = init_pos(1) - edge : lamda : init_pos(1) + edge;
y_vec = init_pos(2) - edge : lamda : init_pos(2) + edge;
search_num = length(x_vec);
mtr = zeros(search_num, search_num);
pos = []; 
[Rcv_num, N_total] = size(sig_rcv);

%信号分段，前提是确保满足有限观测时长频域模型(基于10倍安全裕度)
rdoa_max = calc_rdoa_max(rcvPos);
T = N_total / fs;
J = floor(T / (40*(rdoa_max/vc))); %段数
K = floor(N_total / J); %每段的长度 or fft长度
f = (-floor(K/2) : ceil(K/2)-1) * (fs/K);


sig_fft = zeros(Rcv_num, K, J);
for j = 1:J  
    % 提取第j段信号
    idx_start = (j-1)*K + 1;
    idx_end = j*K;
    sig_segment = sig_rcv(:, idx_start:idx_end);
    
    sig_fft(:, :, j) = fftshift(fft(sig_segment, K, 2), 2);  
end

R_hat_inv = zeros(Rcv_num, Rcv_num, K);
valid_k_idx = find( abs(f) <= band / 2);
for k = valid_k_idx 
    cov_sum = zeros(Rcv_num);
    for j = 1:J
        sig_fft_kj = sig_fft(:, k, j);
        cov_sum = cov_sum + sig_fft_kj * sig_fft_kj'; 
    end
    %协方差矩阵均值
    R_hat = cov_sum / J;
    dynamic_diag_load = (trace(real(R_hat)) / Rcv_num) * 1e-6; % 对角加载
    
    % 加上动态底噪，保证矩阵稳定求逆
    R_hat_loaded = R_hat + dynamic_diag_load * eye(Rcv_num);     
    R_hat_inv(:,:,k) = inv(R_hat_loaded);
end

for x_idx = 1:search_num
    for y_idx = 1:search_num
        currentPos = [x_vec(x_idx),y_vec(y_idx)];
        Lambda_k = zeros(Rcv_num, 1);
        sum_matrix = zeros(Rcv_num, Rcv_num);
        
        for k = valid_k_idx
            for m = 1:Rcv_num
             time_delay =  norm(currentPos - rcvPos(m,:)) / vc;
             Lambda_k(m) = exp(-1j*2*pi*(fc + f(k))*time_delay);
            end
            PhaseMat = conj(Lambda_k) * Lambda_k.';
            sum_matrix = sum_matrix + R_hat_inv(:,:,k) .* PhaseMat;
        end
        
        eig_min = min(eig(sum_matrix + sum_matrix')/2);
        mtr(x_idx, y_idx) = 1 / real(eig_min);
    end
end

[~, max_idx] = max(mtr(:)); 
[best_x_idx, best_y_idx] = ind2sub(size(mtr), max_idx); 
pos = [x_vec(best_x_idx), y_vec(best_y_idx)]; 
% fprintf('MVDR 网格初始坐标: (%.4f, %.4f)搜索定位坐标: (%.4f, %.4f)', init_pos(1), init_pos(2), pos(1), pos(2));

%% 绘图
needPlot = 0;
if needPlot == 1
    figure('Color', 'w'); 
    clf; 
    hold on; 
    
    % ================= 核心修改区 =================
    % 1. 彻底放弃 dB 转换，直接绘制原始的 mtr 矩阵
    % 注意：矩阵需要转置 mtr' 以匹配 x_vec 和 y_vec 的维度
    [~, h_cont] = contourf(x_vec, y_vec, mtr', 100); 
    set(h_cont, 'LineColor', 'none'); 
    
    % 2. 依然使用 jet 色板
    colormap('jet'); 
    
    % 3. 不做任何 clim/caxis 截断！
    % 让 MATLAB 自动将深蓝色映射到 mtr 的最小值，将深红色映射到 mtr 的最大值。
    % ==============================================
    
    % 4. 绘制基站位置 
    h_bs = plot(rcvPos(:,1), rcvPos(:,2), '^w', ...
        'MarkerSize', 10, ...
        'MarkerFaceColor', 'w', ...
        'MarkerEdgeColor', 'k', ...
        'LineWidth', 1.5, ...
        'DisplayName', 'Base Stations');
    
    % 5. 坐标轴与图框美化
    box on; 
    set(gca, 'LineWidth', 1, 'FontSize', 11, 'Layer', 'top'); 
    axis equal; 
    axis([min(x_vec) max(x_vec) min(y_vec) max(y_vec)]);
    xlabel('x [m]', 'FontSize', 11, 'FontWeight', 'normal');
    ylabel('y [m]', 'FontSize', 11, 'FontWeight', 'normal');
    
    % 6. Colorbar 设置
    cb = colorbar;
    set(cb, 'LineWidth', 1, 'FontSize', 11);
    title(cb, 'Raw Power'); % 标题改为原始功率
    
    lgd = legend([h_bs], 'Location', 'best', 'TextColor', 'black', 'FontSize', 11);
    set(lgd, 'Color', 'w', 'EdgeColor', 'k');
    
    hold off;
end
end
