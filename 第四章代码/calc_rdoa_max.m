function rdoa_max = calc_rdoa_max(pos)
% CALC_RDOA_MAX 计算基站群的最大孔径（最大基线距离）
% 该参数用于确定 DPD 算法中的分段长度 K
%
% 输入:
%   pos: [Rcv_num x 2] 或 [Rcv_num x 3] 的基站位置矩阵
%        每一行代表一个基站的坐标 (x, y) 或 (x, y, z)
%
% 输出:
%   rdoa_max: 基站对之间的最大欧几里得距离 (单位: 米)
%             对应论文中提到的 "largest separation between the base stations" 

    % 获取基站数量
    [L, ~] = size(pos);
    
    if L < 2
        rdoa_max = 0;
        warning('基站数量小于 2，无法计算最大间距。');
        return;
    end

    % 方式 1: 使用 pdist 函数 (需要 Statistics and Machine Learning Toolbox)
    % pdist 计算所有点对之间的欧氏距离
    if exist('pdist', 'file')
        distances = pdist(pos, 'euclidean');
        rdoa_max = max(distances);
        
    else
        % 方式 2: 双重循环计算 (不需要任何工具箱，通用性强)
        rdoa_max = 0;
        for i = 1:L-1
            for j = i+1:L
                % 计算第 i 个和第 j 个基站的距离
                d_ij = norm(pos(i,:) - pos(j,:));
                
                % 更新最大值
                if d_ij > rdoa_max
                    rdoa_max = d_ij;
                end
            end
        end
    end
    
    fprintf('计算完毕: 基站间最大距离 (rdoa_max) = %.2f 米\n', rdoa_max);
end