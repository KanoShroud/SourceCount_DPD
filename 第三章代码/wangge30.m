%% wangge30.m  实例级标签可视化
%  适配 main30.m 直接输出的矩阵格式
%  DPD空间谱显示使用和网络一致的 log + z-score 归一化
clc; clear; close all;

%% ═══════════════════════════════════════
%  控制参数
%% ═══════════════════════════════════════
vis_set       = 'test';
vis_trial_idx = 23;
show_dpd      = 1;
show_label    = 1;
show_stats    = 1;

%% ═══════════════════════════════════════
%  加载数据
%% ═══════════════════════════════════════
load(sprintf('%s_data.mat', vis_set));

N_sub      = double(N_sub_val);
max_src    = double(max_src_val);
num_gx     = double(num_grid);
fs         = double(fs_val);
symbolRate = double(symbolRate_val);
BW_actual  = symbolRate * (1 + 0.25 * 1.2);  % 和main27一致
B_win      = double(B_win_val);
B_step     = double(B_step_val);
sub_f_lo   = double(sub_f_lo_val);
sub_f_hi   = double(sub_f_hi_val);
thresh     = double(thresh_val);
edge_v     = double(edge_val);
lamda_v    = double(lamda_val);

% 接收站（固定构型，从配置重建）
R_rcv = 500;
N_rx = 4;
angles_rx = (0:N_rx-1) * 2*pi/N_rx;
rcvPos = [R_rcv*cos(angles_rx)', R_rcv*sin(angles_rx)'];
rcv_num = N_rx;

x_vec = -edge_v : lamda_v : edge_v;
y_vec = -edge_v : lamda_v : edge_v;
[x_grid, y_grid] = meshgrid(x_vec, y_vec);

N_total = size(mtr_sub_all, 1);

%% ═══════════════════════════════════════
%  定位目标样本
%% ═══════════════════════════════════════
if vis_trial_idx > N_total
    error('vis_trial_idx=%d 超出范围（共%d条）', vis_trial_idx, N_total);
end

n_src   = double(src_count_all(vis_trial_idx));
mtr_sub = squeeze(mtr_sub_all(vis_trial_idx, :, :, :));
bm      = squeeze(band_mask_all(vis_trial_idx, :, :));
ig      = squeeze(ignore_mask_all(vis_trial_idx, :, :));
fc_off  = squeeze(fc_offset_all(vis_trial_idx, :));
pt_w    = squeeze(Pt_W_all(vis_trial_idx, :));
spos    = squeeze(src_pos_all(vis_trial_idx, :, :));
avg_snr = double(avg_snr_all(vis_trial_idx));

% ── 计算每个源的平均SNR ──
N_power_W = 10^((-90-30)/10);  % -90dBm
fc_carrier = 5800e6;
snr_per_src = zeros(1, n_src);
for s = 1:n_src
    snr_nodes = zeros(1, rcv_num);
    for m = 1:rcv_num
        d = norm(spos(s,:) - rcvPos(m,:));
        if d < 1, d = 1; end
        PL_dB = PL_free(fc_carrier, d, 0, 0);
        Pr = pt_w(s) * 10^(-PL_dB/10);
        snr_nodes(m) = 10*log10(Pr / N_power_W);
    end
    snr_per_src(s) = mean(snr_nodes);
end

fprintf('=== %s 集 Trial %d ===\n', vis_set, vis_trial_idx);
fprintf('采样率: %.0fMHz  子带: B_win=%.0fMHz  B_step=%.0fMHz  N_sub=%d\n', fs/1e6, B_win/1e6, B_step/1e6, N_sub);
fprintf('搜索区域: ±%.0fm  网格步长: %.0fm\n', edge_v, lamda_v);
fprintf('标签阈值: thresh=%.2f\n', thresh);
fprintf('信源数: %d\n', n_src);

for s = 1:n_src
    dist_to_center = norm(spos(s,:));
    fprintf('  信源%d (slot%d): fc偏移=%.2fMHz | Pt=%.4fW | 位置=[%.0f,%.0f]m | 距中心%.0fm | 平均SNR=%.1fdB\n', ...
            s, s, fc_off(s)/1e6, pt_w(s), spos(s,1), spos(s,2), dist_to_center, snr_per_src(s));
end

for s = 1:max_src
    if s <= n_src
        fprintf('  slot%d band_mask:   [%s]\n', s, num2str(bm(s,:), '%d '));
        fprintf('  slot%d ignore_mask: [%s]\n', s, num2str(ig(s,:), '%d '));
    else
        fprintf('  slot%d: 空槽位\n', s);
    end
end

% 频率关系
if n_src >= 2
    n_cofreq = 0; n_overlap = 0; n_separate = 0;
    for i = 1:n_src-1
        for j = i+1:n_src
            d = abs(fc_off(i) - fc_off(j));
            if d < 1e3,            n_cofreq = n_cofreq + 1;
            elseif d < symbolRate, n_overlap = n_overlap + 1;
            else,                  n_separate = n_separate + 1;
            end
        end
    end
    fprintf('  频率关系: %d对同频, %d对近频重叠, %d对分离\n', n_cofreq, n_overlap, n_separate);
end

%% ═══════════════════════════════════════
%  颜色定义
%% ═══════════════════════════════════════
colors_src = {'b','r','g','m','c'};
slot_colors = [0.2 0.4 0.8; 0.8 0.2 0.2; 0.2 0.7 0.3];

%% ═══════════════════════════════════════
%  覆盖率计算
%% ═══════════════════════════════════════
src_cov_main = zeros(N_sub, n_src);
src_cov_roll = zeros(N_sub, n_src);
for s = 1:n_src
    fc_s = fc_off(s);
    mainlobe_lo = fc_s - symbolRate/2;
    mainlobe_hi = fc_s + symbolRate/2;
    rolloff_lo  = fc_s - BW_actual/2;
    rolloff_hi  = fc_s + BW_actual/2;
    for k = 1:N_sub
        ov_main = max(0, min(mainlobe_hi, sub_f_hi(k)) - max(mainlobe_lo, sub_f_lo(k)));
        ov_all  = max(0, min(rolloff_hi, sub_f_hi(k)) - max(rolloff_lo, sub_f_lo(k)));
        src_cov_main(k, s) = ov_main / B_win;
        src_cov_roll(k, s) = (ov_all - ov_main) / B_win;
    end
end

%% ═══════════════════════════════════════
%  子带DPD空间谱（和网络一致的 log + z-score）
%% ═══════════════════════════════════════
if show_dpd
    % ── log + z-score 归一化（和网络预处理完全一致）──
    all_log = log(double(mtr_sub) + 1);      % (N_sub, gx, gy)
    mu_all  = mean(all_log(:));
    sd_all  = std(all_log(:), 1) + 1e-6;
    all_zscore = (all_log - mu_all) / sd_all;

    fprintf('\nDPD z-score统计: mu=%.4f  sd=%.4f\n', mu_all, sd_all);
    fprintf('z-score范围: [%.2f, %.2f]\n', min(all_zscore(:)), max(all_zscore(:)));

    n_cols = ceil(sqrt(N_sub));
    n_rows = ceil(N_sub / n_cols);
    figure('Color','w','Name',sprintf('[%s] Trial%d 各子带DPD空间谱 (z-score)', vis_set, vis_trial_idx));

    for k = 1:N_sub
        mtr_k    = double(squeeze(mtr_sub(k,:,:)));
        mtr_norm = squeeze(all_zscore(k,:,:));

        subplot(n_rows, n_cols, k);
        surf(x_grid, y_grid, mtr_norm'); shading interp; colorbar;
        xlabel('x(m)','FontSize',6); ylabel('y(m)','FontSize',6);
        set(gca,'FontSize',6);

        f_lo = sub_f_lo(k)/1e6;
        f_hi = sub_f_hi(k)/1e6;

        slot_strs = {};
        for s = 1:n_src
            if bm(s,k) == 1
                slot_strs{end+1} = sprintf('S%d:1', s);
            elseif ig(s,k) == 1
                slot_strs{end+1} = sprintf('S%d:IG', s);
            end
        end
        if isempty(slot_strs)
            t_str = sprintf('W%d [%.0f~%.0f] 空', k, f_lo, f_hi);
            t_color = [0.6 0.6 0.6];
        else
            t_str = sprintf('W%d [%.0f~%.0f] %s', k, f_lo, f_hi, strjoin(slot_strs, ' '));
            if any(bm(:,k) == 1), t_color = [0 0.6 0];
            else, t_color = [0.8 0.6 0]; end
        end
        title(t_str, 'Color', t_color, 'FontSize', 7, 'FontWeight', 'bold');

        hold on;
        for s = 1:n_src
            if bm(s, k) == 1
                plot3(spos(s,1), spos(s,2), max(mtr_norm(:))+0.5, colors_src{s}, ...
                      'Marker','*', 'MarkerSize',12, 'LineWidth',2);
            end
        end
        [~, idx_peak] = max(mtr_k(:));
        [ix, iy] = ind2sub(size(mtr_k), idx_peak);
        plot3(x_vec(ix), y_vec(iy), max(mtr_norm(:))+0.5, 'wx', 'MarkerSize',10, 'LineWidth',2);
        plot3(rcvPos(:,1), rcvPos(:,2), repmat(max(mtr_norm(:))+0.5, rcv_num, 1), 'r^', ...
              'MarkerSize',5, 'MarkerFaceColor','r');
        theta_circle = linspace(0, 2*pi, 100);
        plot3(R_rcv*cos(theta_circle), R_rcv*sin(theta_circle), ...
              repmat(max(mtr_norm(:))+0.5, 1, 100), 'r--', 'LineWidth', 0.5);
        hold off;
    end
    sgtitle(sprintf('[%s] Trial%d  %d源 (网络视角: log+z-score)', vis_set, vis_trial_idx, n_src), 'FontSize',12);
end

%% ═══════════════════════════════════════
%  标签总览图
%% ═══════════════════════════════════════
if show_label
    figure('Color','w','Name',sprintf('[%s] Trial%d 标签总览', vis_set, vis_trial_idx), ...
           'Position',[100 100 1100 900]);

    %% 第1行：频谱-子带对应
    subplot(5,1,1);
    hold on;
    for k = 1:N_sub
        f_lo = sub_f_lo(k)/1e6; f_hi = sub_f_hi(k)/1e6;
        has_signal = any(bm(:,k) == 1);
        has_ignore = any(ig(:,k) == 1);
        if has_signal, c = [0.8 1.0 0.8];
        elseif has_ignore, c = [1.0 0.95 0.7];
        else, c = [0.92 0.92 0.92]; end
        patch([f_lo f_hi f_hi f_lo],[0 0 1 1], c, ...
              'FaceAlpha',0.5, 'EdgeColor',[0.3 0.3 0.3], 'LineWidth',0.5);
        text((f_lo+f_hi)/2, 0.06, sprintf('W%d', k), 'HorizontalAlignment','center', 'FontSize',6);
    end
    for s = 1:n_src
        ml_lo = (fc_off(s) - symbolRate/2) / 1e6;
        ml_hi = (fc_off(s) + symbolRate/2) / 1e6;
        y_pos = 0.35 + s * 0.15;
        plot([ml_lo ml_hi], [y_pos y_pos], colors_src{s}, 'LineWidth', 4, ...
             'DisplayName', sprintf('Slot%d [%.1f,%.1f]MHz', s, ml_lo, ml_hi));
        sig_lo = (fc_off(s) - BW_actual/2) / 1e6;
        sig_hi = (fc_off(s) + BW_actual/2) / 1e6;
        plot([sig_lo ml_lo], [y_pos y_pos], colors_src{s}, 'LineWidth',1.5, 'LineStyle',':', 'HandleVisibility','off');
        plot([ml_hi sig_hi], [y_pos y_pos], colors_src{s}, 'LineWidth',1.5, 'LineStyle',':', 'HandleVisibility','off');
    end
    hold off;
    xlim([-fs/2/1e6, fs/2/1e6]); ylim([0 1]);
    xlabel('频率 (MHz)','FontSize',11); set(gca,'YTick',[]);
    title(sprintf('[%s] Trial%d  %d源  频谱-子带对应', vis_set, vis_trial_idx, n_src), 'FontSize',11);
    legend('Location','best','FontSize',7);

    %% 第2行：band_mask 逐槽位
    subplot(5,1,2);
    hold on;
    bar_width = 0.7 / max_src;
    for s = 1:max_src
        x_offset = (s - (max_src+1)/2) * bar_width;
        for k = 1:N_sub
            if s <= n_src
                if bm(s,k) == 1
                    bar(k + x_offset, 1, bar_width, 'FaceColor', slot_colors(s,:), 'EdgeColor','none');
                elseif ig(s,k) == 1
                    bar(k + x_offset, 0.5, bar_width, 'FaceColor', slot_colors(s,:), 'FaceAlpha',0.3, 'EdgeColor','none');
                end
            end
        end
    end
    hold off;
    xlabel('子带编号','FontSize',11); ylabel('占用','FontSize',11);
    title('band\_mask（实色=1  半透明=ignore） 蓝=slot1 红=slot2 绿=slot3', 'FontSize',10);
    set(gca,'XTick',1:N_sub,'YTick',[0 0.5 1]); ylim([0 1.3]); grid on;

    %% 第3行：覆盖率分解
    subplot(5,1,3);
    hold on;
    bar_width2 = 0.7 / max(n_src, 1);
    for s = 1:n_src
        x_offset = (s - (n_src+1)/2) * bar_width2;
        bar((1:N_sub) + x_offset, src_cov_main(:,s), bar_width2, ...
             'FaceColor', slot_colors(s,:), 'EdgeColor','none');
        bar((1:N_sub) + x_offset, src_cov_main(:,s) + src_cov_roll(:,s), ...
             bar_width2, 'FaceColor', slot_colors(s,:), 'FaceAlpha',0.3, 'EdgeColor','none');
    end
    yline(thresh, 'r--', sprintf('thresh=%.2f', thresh), 'LineWidth',1.2, 'FontSize',8, 'Color',[0.8 0 0]);
    hold off;
    xlabel('子带编号','FontSize',11); ylabel('覆盖率','FontSize',11);
    title('逐信源覆盖率（实色=主瓣  半透明=含滚降  红虚线=阈值）','FontSize',10);
    if n_src > 0
        leg_str = arrayfun(@(s) sprintf('Slot%d (fc=%.1fMHz)', s, fc_off(s)/1e6), 1:n_src, 'UniformOutput',false);
        legend(leg_str{:}, 'Location','best','FontSize',7);
    end
    set(gca,'XTick',1:N_sub);
    if n_src > 0, ylim([0, max(sum(src_cov_main + src_cov_roll, 2)) + 0.2]); end
    grid on;

    %% 第4行：空间分布
    subplot(5,1,4);
    hold on;
    theta_circle = linspace(0, 2*pi, 200);
    plot(R_rcv*cos(theta_circle), R_rcv*sin(theta_circle), 'k--', 'LineWidth', 1);
    plot(rcvPos(:,1), rcvPos(:,2), 'r^', 'MarkerSize',10, 'MarkerFaceColor','r', 'DisplayName','接收站');
    for s = 1:n_src
        dist_c = norm(spos(s,:));
        plot(spos(s,1), spos(s,2), 'o', 'Color', colors_src{s}, ...
             'MarkerSize', 10, 'MarkerFaceColor', colors_src{s}, 'LineWidth', 2, ...
             'DisplayName', sprintf('Slot%d d=%.0fm Pt=%.3fW SNR=%.1fdB', s, dist_c, pt_w(s), snr_per_src(s)));
    end
    plot([-edge_v edge_v edge_v -edge_v -edge_v], [-edge_v -edge_v edge_v edge_v -edge_v], 'b:', 'LineWidth',0.5);
    hold off;
    axis equal; grid on;
    xlabel('x (m)','FontSize',11); ylabel('y (m)','FontSize',11);
    title('信源与接收站空间分布','FontSize',11);
    legend('Location','best','FontSize',7);

    %% 第5行：文字汇总
    subplot(5,1,5);
    axis off;
    txt = {};
    txt{end+1} = sprintf('【实例级标签】src_count=%d  max_src=%d', n_src, max_src);
    txt{end+1} = sprintf('子带: B_win=%.0fMHz  B_step=%.0fMHz  N_sub=%d  阈值=%.2f', B_win/1e6, B_step/1e6, N_sub, thresh);
    txt{end+1} = sprintf('接收站: %d个，R=%.0fm  搜索: ±%.0fm  网格: %.0fm', rcv_num, R_rcv, edge_v, lamda_v);
    if n_src > 0, txt{end+1} = sprintf('最弱源平均SNR: %.1fdB', avg_snr); end
    txt{end+1} = '';
    for s = 1:n_src
        dist_c = norm(spos(s,:));
        txt{end+1} = sprintf('Slot%d: fc偏移=%.1fMHz  位置=[%.0f,%.0f]m  距中心%.0fm  Pt=%.3fW  SNR=%.1fdB', ...
            s, fc_off(s)/1e6, spos(s,1), spos(s,2), dist_c, pt_w(s), snr_per_src(s));
        bm_str = ''; ig_str = '';
        for k = 1:N_sub
            if bm(s,k) == 1, bm_str = [bm_str, sprintf(' W%d', k)]; end
            if ig(s,k) == 1, ig_str = [ig_str, sprintf(' W%d', k)]; end
        end
        if ~isempty(bm_str), txt{end+1} = sprintf('       占据子带:%s', bm_str); end
        if ~isempty(ig_str), txt{end+1} = sprintf('       ignore子带:%s', ig_str); end
    end
    for s = n_src+1:max_src, txt{end+1} = sprintf('Slot%d: 空', s); end
    text(0.05, 0.95, strjoin(txt, '\n'), 'FontSize',9, 'FontName','FixedWidth', ...
         'VerticalAlignment','top', 'Interpreter','none');
end

%% ═══════════════════════════════════════
%  全局统计
%% ═══════════════════════════════════════
if show_stats
    fprintf('\n===== 全局统计（%d条样本）=====\n', N_total);

    src_counts = double(src_count_all);
    max_n = max(src_counts);
    fprintf('\n信源数量分布:\n');
    for c = 0:max_n
        cnt = sum(src_counts == c);
        if cnt > 0, fprintf('  %d源: %d条 (%.1f%%)\n', c, cnt, 100*cnt/N_total); end
    end

    % SNR和距离
    all_dist = []; all_snr_v = []; all_pt_v = [];
    for i = 1:N_total
        n = src_counts(i);
        for s = 1:n
            all_dist(end+1) = norm(squeeze(src_pos_all(i,s,:)));
            all_pt_v(end+1) = Pt_W_all(i,s);
        end
    end
    has_src = (src_counts > 0);
    valid_snr = double(avg_snr_all(has_src));
    if ~isempty(valid_snr)
        fprintf('\nSNR范围: [%.1f, %.1f] dB\n', min(valid_snr), max(valid_snr));
        fprintf('信源距离范围: [%.0f, %.0f] m\n', min(all_dist), max(all_dist));
        fprintf('发射功率范围: [%.3f, %.3f] W\n', min(all_pt_v), max(all_pt_v));
    end

    % 子带标签
    total_band1 = sum(band_mask_all(:) == 1);
    total_ignore = sum(ignore_mask_all(:) == 1);
    total_band0 = numel(band_mask_all) - total_band1 - total_ignore;
    total_all = numel(band_mask_all);
    fprintf('\n子带标签分布 (共 %d 个槽位×子带):\n', total_all);
    fprintf('  标1: %d (%.1f%%)\n', total_band1, 100*total_band1/total_all);
    fprintf('  标0: %d (%.1f%%)\n', total_band0, 100*total_band0/total_all);
    fprintf('  ignore: %d (%.1f%%)\n', total_ignore, 100*total_ignore/total_all);

    % 每源子带数
    bands_per_src = [];
    for i = 1:N_total
        n = src_counts(i);
        for s = 1:n
            bands_per_src(end+1) = sum(band_mask_all(i,s,:) == 1);
        end
    end
    if ~isempty(bands_per_src)
        fprintf('\n每个信源占据子带数: 平均%.1f  最小%d  最大%d\n', ...
                mean(bands_per_src), min(bands_per_src), max(bands_per_src));
    end

    % 频率重叠
    fprintf('\n===== 频率重叠模式统计 =====\n');
    n_zero=0; n_one=0; n_cofreq=0; n_near=0; n_mix=0; n_sep=0;
    for i = 1:N_total
        n = src_counts(i);
        if n==0, n_zero=n_zero+1; continue; end
        if n==1, n_one=n_one+1; continue; end
        fc_n = fc_offset_all(i, 1:n);
        dists = [];
        for ii=1:n-1, for jj=ii+1:n, dists(end+1)=abs(fc_n(ii)-fc_n(jj)); end, end
        nc=sum(dists<1e3); nn=sum(dists>=1e3 & dists<symbolRate); nf=sum(dists>=symbolRate);
        if nf==length(dists), n_sep=n_sep+1;
        elseif nc>0 && nn==0, n_cofreq=n_cofreq+1;
        elseif nn>0 && nc==0, n_near=n_near+1;
        else, n_mix=n_mix+1; end
    end
    fprintf('  0源:%d  1源:%d  含同频:%d  含近频:%d  混合:%d  全分离:%d\n', ...
            n_zero, n_one, n_cofreq, n_near, n_mix, n_sep);

    %% 可视化
    figure('Color','w','Name','全局统计','Position',[100 100 1600 500]);

    subplot(1,5,1);
    src_hist = zeros(1, max_n+1);
    for c=0:max_n, src_hist(c+1)=sum(src_counts==c); end
    bar(0:max_n, src_hist, 0.6, 'FaceColor',[0.3 0.6 0.9]);
    xlabel('信源数量'); ylabel('样本数'); title('信源数分布');
    set(gca,'XTick',0:max_n); grid on;

    subplot(1,5,2);
    if ~isempty(valid_snr)
        histogram(valid_snr, 30, 'FaceColor',[0.3 0.7 0.5]);
        xlabel('最弱源SNR (dB)'); ylabel('样本数'); title('SNR分布'); grid on;
    end

    subplot(1,5,3);
    if ~isempty(all_dist)
        histogram(all_dist, 30, 'FaceColor',[0.5 0.5 0.8]);
        xlabel('信源距离 (m)'); ylabel('信源数'); title('距离分布'); grid on;
    end

    subplot(1,5,4);
    if ~isempty(bands_per_src)
        histogram(bands_per_src, 'BinMethod','integers', 'FaceColor',[0.3 0.7 0.5]);
        xlabel('占据子带数'); ylabel('信源数'); title('每源子带数'); grid on;
    end

    subplot(1,5,5);
    overlap_data = [n_zero, n_one, n_cofreq, n_near, n_mix, n_sep];
    overlap_labels = {'0源','1源','含同频','含近频','混合','全分离'};
    valid_idx = overlap_data > 0;
    if any(valid_idx)
        pie(overlap_data(valid_idx), overlap_labels(valid_idx));
        title('频率重叠模式');
    end
end