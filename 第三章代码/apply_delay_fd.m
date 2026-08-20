function sig_out = apply_delay_fd(sig, tau, fs)
% 在频域对基带信号施加时延 tau（秒），支持任意小数时延
%   sig  : 1×N 复数基带信号
%   tau  : 时延（秒）
%   fs   : 采样率（Hz）
N   = length(sig);
f   = (-N/2 : N/2-1) * (fs / N);
Sig_F         = fftshift(fft(sig));
Sig_F_delayed = Sig_F .* exp(-1j * 2*pi * f * tau);
sig_out       = ifft(ifftshift(Sig_F_delayed));
end