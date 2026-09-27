% HRDPD
% Reference: High Resolution Direct Position Determination of Radio Frequency Sources
% written by Tom Tirer
clear; clc;

M = 7; % number of array elements (ULA)
J = 20; % number of intervals (intervals of a signal from same localization)
K = 64; % number of used frequencies
SNR = 30; % [dB]
Q = 4; % number of emitters
L = 4; % number of arrays
array = (0:M-1)*.5; % array elements location in wavelength
fmax = 150*10^3;
signals_correlation = 0;

x0 = [-1.5,1.5,0,0]; % true x coordinate of emitters [km]
y0 = [-50,-50,-51,-49]; % true y coordinate of emitters [km]
x_array = [100,100,100,100]; % x coordinate of arrays [km]
y_array = [-150,-50,50,150]; % y coordinate of arrays [km]
phi_array = [90,90,90,90]; % [deg] counterclock-wise rotation of the ULA array baseline with respect to the x axis

C = 3*10^5; % [km/sec]

x_array = x_array(1:L);
y_array = y_array(1:L);
x0 = x0(1:Q);
y0 = y0(1:Q);

w = (0:1:K-1)*2*pi*fmax/K;

S0_all_temp = (randn(Q,K*J) + 1i*randn(Q,K*J))./sqrt(2); % Signal Fourier Coefficients
sigmaMat = toeplitz([1,repmat(signals_correlation,1,Q-1)]);
[U_sigma,D_sigma,~] = svd(sigmaMat); % V_sigma=U_sigma
rootSigmaMat = U_sigma*(D_sigma.^0.5)*U_sigma';
S0_all_temp = rootSigmaMat * S0_all_temp;

S0_all = reshape(S0_all_temp,Q,K,J);
for q=1:1:Q
    for j=1:1:J
        S0_all(q,:,j) = S0_all(q,:,j)./norm(S0_all(q,:,j)); % Normalization of signal of every emitter in every interval
    end
end

X = zeros(M,K,J,L);
    
% create measurements
for l=1:1:L
    alpha = 0.995+0.1*(randn(Q,1) + 1i*randn(Q,1))./sqrt(2); % E[|alpha^2|]=1
    
    theta0 = atan2(y0-y_array(l),x0-x_array(l))*180/pi - phi_array(l); % true signal AOA [deg]
    tau0 = (1/C) * ((x0-x_array(l)).^2+(y0-y_array(l)).^2).^0.5;
    
    a0 = M^(-.5)*exp(-1i*2*pi*array'*cos(theta0*pi/180)); % true steering vector
    
    fac = 10^(SNR/20)*M^(.5);
            
    for j=1:1:J
        Noise = (randn(M,K) + 1i*randn(M,K))./sqrt(2); %noise
        Noise = Noise./sqrt(K);       
        X(:,:,j,l) = fac * a0 * diag(alpha) * (S0_all(:,:,j).*exp(-1i*tau0'*w)) + Noise;        
    end
end

%%
x_candidate = [-2.5:0.125:2.5];
y_candidate = [-52.5:0.125:-47.5];

Q_ML = zeros(length(y_candidate),length(x_candidate));
Q_HR = zeros(length(y_candidate),length(x_candidate));
r = X(:,:,:,1:L); %(M,K,J,L)

r2 = permute(r,[1 4 3 2]);
r_bar = reshape(r2,M*L,J,K);
inv_R_hat = zeros(M*L,M*L,K); % for all frequencies

tic

for k=1:1:K % invert all R_hat beforehand
    R_hat = r_bar(:,:,k)*r_bar(:,:,k)'/J;
    inv_R_hat(:,:,k) = inv(R_hat+diag(0.001*ones(1,M*L))); % adding diagonal loading
end

for x_ind=1:1:length(x_candidate)
    for y_ind=1:1:length(y_candidate)
        
        x = x_candidate(x_ind);
        y = y_candidate(y_ind);
        D_ML = zeros(L,L);
        D_HR = zeros(L,L);
        a_tilda = zeros(M*L,1);
        
        for k=1:1:K
            R_hat = r_bar(:,:,k)*r_bar(:,:,k)'/J;
            
            disp(['x_ind=',num2str(x_ind),', y_ind=',num2str(y_ind),', k=',num2str(k)]);
            
            for l=1:1:L
                theta = atan2(y-y_array(l),x-x_array(l)) - phi_array(l)*pi/180;
                tau = (1/C) * ((x-x_array(l)).^2+(y-y_array(l)).^2).^0.5;
                a = exp(-1i*2*pi*array'*cos(theta));
                a_tilda((l-1)*M+1:l*M) = a * exp(-1i*w(k)*tau);
            end
            
            Lambda = diag(a_tilda);
            B = kron(eye(L),ones(M,1));
            
            D_ML = D_ML + B'*Lambda'*R_hat*Lambda*B;
            D_HR = D_HR + B'*Lambda'*inv_R_hat(:,:,k)*Lambda*B;
        end
        
        Q_ML(y_ind,x_ind) = max(real(eig(D_ML)));
        Q_HR(y_ind,x_ind) = max(real(eig(inv(D_HR+diag(0.001*ones(1,L))))));
    end
end
toc

figure; imagesc(x_candidate,y_candidate,Q_ML); title('Q_S_M_L'); set(gca,'YDir','normal');

figure; imagesc(x_candidate,y_candidate,Q_HR); title('Q_H_R'); set(gca,'YDir','normal');

