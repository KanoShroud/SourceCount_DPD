%本函数文件用来构建自由空间路径损耗模型，输出PL，单位为dB。
%Gt,Gr单位为dbi
function PL=PL_free(fc,dist,Gt,Gr)
vc = 299792458;
lamda=vc /fc; %fc代表载波频率[Hz]
tmp=lamda./(4*pi*dist); %dist代表基站和用户之间的距离
PL=-20*log10(tmp)-Gt-Gr;