function dataLen = calc_dataLen(len, fs, symbolRate, margin)
% CALC_DATALEN 自适应计算Gen_basesig所需的dataLen
%
% 输入：
%   len        : 实际需要的采样点数
%   fs         : 目标采样率（Hz）
%   symbolRate : 符号率（Hz），即Txobj.symbolRate_V
%   margin     : 额外安全余量（点数），默认500
%
% 输出：
%   dataLen    : 传给Gen_basesig的Frame_len

if nargin < 4, margin = 500; end

sps0 = 8;                        % Gen_basesig内部固定的中间过采样倍数
Fs0  = symbolRate * sps0;        % 中间采样率

if Fs0 == fs
    % 无需resample，内部只有overlap_retention的瞬态（很短）
    transient = 100;
else
    % resample(base0, p, q, 256) 的滤波器长度 = 2*256*max(p,q)+1
    [p, q]    = rat(fs / Fs0, 1e-6);
    filt_len  = 2 * 256 * max(p, q) + 1;

    if fs > Fs0
        % 升采样：瞬态在目标采样率fs下
        transient = ceil(filt_len / 2);
    else
        % 降采样：瞬态在中间采样率Fs0下，换算到fs
        transient = ceil((filt_len / 2) * (fs / Fs0));
    end
end

dataLen = len + transient + margin;

fprintf(['calc_dataLen: symbolRate=%.3fMHz  Fs0=%.3fMHz  ' ...
         'transient=%d点  margin=%d  dataLen=%d\n'], ...
         symbolRate/1e6, Fs0/1e6, transient, margin, dataLen);
end