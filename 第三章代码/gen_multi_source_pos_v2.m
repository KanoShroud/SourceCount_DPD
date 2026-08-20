function src_pos = gen_multi_source_pos_v2(rcvPos, n_src, dist_range, min_dist_src, max_iter)
% 在搜索区域内随机生成多个信源位置（信源可在接收站凸包内外）
%
% 约束：
%   1. 信源到中心点的距离在 dist_range 范围内
%   2. 信源之间距离 >= min_dist_src
%
% 输入：
%   rcvPos       : [rcv_num × 2] 接收站坐标
%   n_src        : 需要生成的信源数量
%   dist_range   : [min_dist, max_dist] 信源到原点的距离范围（米）
%   min_dist_src : 信源之间最小间距（米），默认300
%   max_iter     : 最大迭代次数，默认50000
%
% 输出：
%   src_pos : [n_src × 2] 各信源坐标

if nargin < 4, min_dist_src = 300;   end
if nargin < 5, max_iter     = 50000; end

dist_min = dist_range(1);
dist_max = dist_range(2);

src_pos = zeros(n_src, 2);
placed  = 0;

for iter = 1:max_iter
    % 随机距离和角度（极坐标均匀采样）
    % 注意：均匀面积采样需要 sqrt(rand)
    r     = sqrt(dist_min^2 + (dist_max^2 - dist_min^2) * rand());
    theta = 2 * pi * rand();
    x = r * cos(theta);
    y = r * sin(theta);

    % 约束：与已放置的信源距离 >= min_dist_src
    if placed > 0
        dist_to_src = sqrt((src_pos(1:placed,1) - x).^2 + ...
                           (src_pos(1:placed,2) - y).^2);
        if min(dist_to_src) < min_dist_src
            continue;
        end
    end

    % 所有约束满足，记录该位置
    placed = placed + 1;
    src_pos(placed, :) = [x, y];

    if placed == n_src
        return;
    end
end

% 超出最大迭代次数
warning('gen_multi_source_pos_v2: 迭代%d次后仅放置%d/%d个信源。', ...
        max_iter, placed, n_src);

% 未放置的信源填充为随机位置
for s = placed+1 : n_src
    r     = sqrt(dist_min^2 + (dist_max^2 - dist_min^2) * rand());
    theta = 2 * pi * rand();
    src_pos(s, :) = [r*cos(theta), r*sin(theta)];
end

end