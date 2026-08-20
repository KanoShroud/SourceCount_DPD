function [Bw,base] = Gen_basesig(Frame_len,Fs,txId)
global Txobj
% 函数 Gen_basesig 生成基带信号
% 输入参数：
%   Frame_len - 数据长度
%   Fs - 采样频率
%   Fd - 符号速率
%   mod_type - 调制类型
%   arfa - 滚降系数
%   shaping_type - 成形类型
%   mod_depth - 调制深度
%   conttype - 连续相位类型
% 输出参数：
%   Bw - 带宽
%   base - 基带信号
x = find(txId == Txobj.txId_V);
Fd = Txobj.symbolRate_V(x);
mod_type = Txobj.modType_V(x);
arfa = Txobj.arfa_V(x);
shaping_type = Txobj.shapingType_V(x);
mod_depth = Txobj.modDepth_V(x);
conttype = Txobj.contPhase_V(x);
multiplexingType = Txobj.multiplexingType_V(x);
%%
modulationTypes = categorical(["BPSK", "QPSK","PI/4DQPSK","OQPSK","8PSK","16PSK" ,...
    "16QAM","32QAM","64QAM","128QAM","256QAM","512QAM","1024QAM","16APSK","32APSK",...
    "2ASK","4ASK","2FSK","4FSK","8FSK","MSK","GMSK"]);
type = find(modulationTypes == mod_type);
sps  = Fs / Fd ;%码元（符号）个数

%%
switch multiplexingType
    case "NONE"
        
    if(type<=find(modulationTypes=="32APSK"))
        sps0  = 8;   %%第一次过采样
        Frame_len0 = Frame_len/sps*sps0; %%第一次输出点数
        Fs0 = Fd*sps0;
        base0 = BaseBand_QPA(ceil(Frame_len0*1.05),sps0,arfa,shaping_type,modulationTypes(type));%%基带信号生成
        Bw = Fd*(1+arfa*1.2);%因为使用滤波器所以引入了滚降系数
        %%Fs0 ~= Fs: 如果初始采样频率 Fs0 不等于目标采样频率 Fs，则需要对信号进行处理以适配目标采样频率。
        if(Fs0~=Fs)
            base0 = overlap_retention(base0,Fs0,Bw);
            base  = resample(base0,Fs,Fs0,256);
        else
            base  = overlap_retention(base0,Fs,Bw);
        end
        if(length(base)<Frame_len)
            base  = [base,zeros(1,Frame_len-length(base))];
        else
            base  = base(1:Frame_len);
        end
    elseif modulationTypes(type)=="2FSK"||modulationTypes(type)=="4FSK"||modulationTypes(type)=="8FSK"
        sps0  = 16;   %%第一次过采样
        Frame_len0 = Frame_len/sps*sps0; %%第一次输出点数
        Fs0 = Fd*sps0;
        M =  double(extract(string(modulationTypes(type)),1));
        Bw = ((M-1)*mod_depth+2)*Fd;    % ???????????????????不理解
        if(Bw>=Fs)
            mod_depth = (Fs/Fd-2)/(M-1);% ???????????????????不理解
        end
        [Bw,base0] = BaseBand_AFM(ceil(Frame_len0*1.05),Fd,sps0,arfa,shaping_type,modulationTypes(type),mod_depth,conttype);
        if(Fs0~=Fs)
            base0 = overlap_retention(base0,Fs0,Bw);
            base  = resample(base0,Fs,Fs0,256);
        else
            base  = overlap_retention(base0,Fs,Bw);
        end
        if(length(base)<Frame_len)
            base  = [base,zeros(1,Frame_len-length(base))];
        else
            base  = base(1:Frame_len);
        end
    elseif modulationTypes(type)=="2ASK"||modulationTypes(type)=="4ASK"
        sps0  = 8;                       %%第一次过采样
        Frame_len0 = Frame_len/sps*sps0; %%第一次输出点数
        Fs0 = Fd*sps0;
        [~,base0] = BaseBand_AFM(ceil(Frame_len0*1.05),Fd,sps0,arfa,shaping_type,modulationTypes(type),mod_depth,conttype);
        Bw = Fd*(1+arfa*1.2);
        if(Fs0~=Fs)
            base0 = overlap_retention(base0,Fs0,Bw);
            base  = resample(base0,Fs,Fs0,256);
        else
            base  = overlap_retention(base0,Fs,Bw);
        end
        if(length(base)<Frame_len)
            base  = [base,zeros(1,Frame_len-length(base))];
        else
            base  = base(1:Frame_len);
        end
    elseif modulationTypes(type)=="MSK"||modulationTypes(type)=="GMSK"
        sps0  = 8;                       %% 第一次过采样
        Frame_len0 = Frame_len/sps*sps0; %% 第一次输出点数
        Fs0 = Fd*sps0;
        [Bw,base0] = BaseBand_AFM(ceil(Frame_len0*1.05),Fd,sps0,arfa,shaping_type,modulationTypes(type),mod_depth,conttype);
        if(Fs0~=Fs)
            base0 = overlap_retention(base0,Fs0,Bw);
            base  = resample(base0,Fs,Fs0,256);
        else
            base  = overlap_retention(base0,Fs,Bw);
        end
        if(length(base)<Frame_len)
            base  = [base,zeros(1,Frame_len-length(base))];
        else
            base  = base(1:Frame_len);
        end
    end

  case "CDMA"
    [PN_type,pnGenPolyCoeffs,pnGenInitState] = Get_CDMA('CDMA.json',node_id) ;
    if(modulationTypes(type) == "BPSK" || modulationTypes(type) == "QPSK" )
        sps0  = 8;                       %% 第一次过采样
        Frame_len0 = Frame_len/sps*sps0; %% 第一次输出点数
        Fs0 = Fd*sps0;
        base0 = BaseBand_CDMA(ceil(Frame_len0*1.05),modulationTypes(type), PN_type, pnGenPolyCoeffs,pnGenInitState,shaping_type, arfa, sps0,Fs0);
        Bw = Fd*(1+arfa*1.2);
        if(Fs0~=Fs)
            base0 = overlap_retention(base0,Fs0,Bw);
            base  = resample(base0,Fs,Fs0,256);
        else
            base  = overlap_retention(base0,Fs,Bw);
        end
        if ( length(base) <Frame_len )
            base  = [base,zeros(1,Frame_len-length(base))];
        else
            base  = base(1:Frame_len);
        end
    else
        disp("CDMA只有BPSK、QPSK两种调制方式！")
    end


    case "TDMA"

    case "OFDM" 

end

% 对信号进行归一化处理，使得信号的平均幅度为 1。
base = base/mean(abs(base));
end

