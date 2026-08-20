function [ConvResult2] = overlap_retention(sig,Fs,DK)

nfft = 16384;
FilterOrder = 1000;
if DK/Fs>=1
    
% coef1 = 1;
ConvResult2 = sig;
else
coef1 = DK/Fs;

fir = fir1(FilterOrder,coef1);
FirLength = length(fir);

hk=fft(fir,nfft);
SigLength = nfft + 1 -FirLength ;
GroupNum = ceil(length(sig)/SigLength);

if(GroupNum * SigLength - length(sig) < 1000)
sig2 = zeros(GroupNum*SigLength+1,1);
for i=1 : length(sig)
sig2(i) = sig(i);
end
else
    sig2 = sig;
end
GroupNum2 = ceil(length(sig2)/SigLength);


SigTotal = zeros(GroupNum2*SigLength,1);
for i=1 : length(sig2)
SigTotal(i) = sig2(i);
end

ConvResult = zeros(1,length(SigTotal));
for k=0:GroupNum2-1
x1=zeros(1,nfft);
x1(nfft-SigLength+1:nfft)=SigTotal(k*SigLength+1:(k+1)*SigLength);
if k~=0
x1(1:nfft-SigLength)=SigTotal(k*SigLength-nfft+SigLength+1:k*SigLength);
end
x1k=fft(x1,nfft);
yk=x1k.*hk;
y2=ifft(yk);
ConvResult(k*SigLength+1:(k+1)*SigLength)=y2(nfft-SigLength+1:nfft);
end
length1 = floor(FirLength/2);
ConvResult2 = ConvResult(length1+1:length(sig)+length1);
end
end