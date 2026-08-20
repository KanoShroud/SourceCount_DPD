function [base_sig] = BaseBand_QPA(point0,sps,beta,shape,modType)
bit_lenmin = ceil(point0/sps);
if(modType == "OQPSK")
    bitdata =  randi([0,3],1,bit_lenmin);
    base_sig = OqpskModulator(bitdata,sps,beta,shape);
else
    switch modType
        case "BPSK"
            bitdata =  randi([0,1],1,bit_lenmin);
            mod_data = BpskModulator(bitdata);
        case "QPSK"
            bitdata =  randi([0,3],1,bit_lenmin);
            mod_data = QpskModulator(bitdata);
        case "PI/4DQPSK"
            bitdata =  randi([0,3],1,bit_lenmin);
            mod_data = PI4DqpskModulator(bitdata);
        case "8PSK"
            bitdata =  randi([0,7],1,bit_lenmin);
            mod_data = Psk8Modulator(bitdata);
        case "16PSK"
            bitdata =  randi([0,15],1,bit_lenmin);
            mod_data = Psk16Modulator(bitdata);
        case "8QAM"
            bitdata =  randi([0,7],1,bit_lenmin);
            mod_data = QAM8Modulator(bitdata);
        case "16QAM"
            bitdata =  randi([0,15],1,bit_lenmin);
            mod_data = QAM16Modulator(bitdata);
        case "32QAM"
            bitdata =  randi([0,31],1,bit_lenmin);
            mod_data = QAM32Modulator(bitdata);
        case "64QAM"
            bitdata =  randi([0,63],1,bit_lenmin);
            mod_data = QAM64Modulator(bitdata);
        case "128QAM"
            bitdata =  randi([0,127],1,bit_lenmin);
            mod_data = QAM128Modulator(bitdata);
        case "256QAM"
            bitdata =  randi([0,255],1,bit_lenmin);
            mod_data = QAM256Modulator(bitdata);
        case "512QAM"
            bitdata =  randi([0,511],1,bit_lenmin);
            mod_data = QAM512Modulator(bitdata);       
        case "1024QAM"
            bitdata =  randi([0,1023],1,bit_lenmin);
            mod_data = QAM1024Modulator(bitdata);
        case "16APSK"
            gama = 3.15;
            R1 =sqrt(16/(4+12*gama*gama));
            R2 =R1*gama;
            radii = [R1 R2];
            bitdata =  randi([0,15],1,bit_lenmin);
            mod_data = Apsk16Modulator(bitdata,radii);
        case "32APSK"
            gama1 =2.72;
            gama2 = 4.87;
            R1 =sqrt(32/(4+12*gama1*gama1+16*gama2*gama2));
            R2 =R1 * gama1;
            R3 =R1 * gama2;
            radii = [R1 R2 R3];
            bitdata =  randi([0,31],1,bit_lenmin);
            mod_data = Apsk32Modulator(bitdata,radii);
    end

    if(shape == "RaisedCos")
        %shape = "Normal";
        rctFilt = comm.RaisedCosineTransmitFilter(...
        'Shape',                 "Normal", ...
        'RolloffFactor',         beta, ...
        'FilterSpanInSymbols',    sps, ...
        'OutputSamplesPerSymbol', sps);
    elseif(shape == "RootRaisedCos")
        %shape = "Square root";
        rctFilt = comm.RaisedCosineTransmitFilter(...
        'Shape',                 "Square root", ...
        'RolloffFactor',         beta, ...
        'FilterSpanInSymbols',    sps, ...
        'OutputSamplesPerSymbol', sps);
    else
        %shape = "Square root";
        rctFilt = comm.RaisedCosineTransmitFilter(...
        'Shape',                 "Square root", ...
        'RolloffFactor',         beta, ...
        'FilterSpanInSymbols',    sps, ...
        'OutputSamplesPerSymbol', sps);
    end

    yo=(rctFilt(mod_data.')).';
    base_sig= yo(sps*sps/2+1:length(yo));% 基带信号;
end

end


function mod_data = BpskModulator(msg )
M =2;
mod_data = pskmod(msg,M,0);

end
function mod_data = QpskModulator(msg )
M =4;
mod_data = pskmod(msg,M,pi/4);

end
function mod_data = PI4DqpskModulator(msg )
M =4;
mod_data = dpskmod(msg,M,pi/4);
end

function mod_data = OqpskModulator(msg,sps,beta,shape)
if(shape == "RaisedCos")
    %shape = "Normal raised cosine";
    oqpskMod = comm.OQPSKModulator('BitInput', false,  'PulseShape', ...
    "Normal raised cosine",'RolloffFactor',beta,'FilterSpanInSymbols',sps,...
    'SamplesPerSymbol',sps);
elseif(shape == "RootRaisedCos")
    %shape ="Root raised cosine";
    oqpskMod = comm.OQPSKModulator('BitInput', false,  'PulseShape', ...
    "Root raised cosine",'RolloffFactor',beta,'FilterSpanInSymbols',sps,...
    'SamplesPerSymbol',sps);
else
    %shape ="Root raised cosine";
    oqpskMod = comm.OQPSKModulator('BitInput', false,  'PulseShape', ...
    "Root raised cosine",'RolloffFactor',beta,'FilterSpanInSymbols',sps,...
    'SamplesPerSymbol',sps);
end

mod_data = oqpskMod(msg.').';
end

function mod_data = Psk8Modulator(msg )
M =8;
mod_data = pskmod(msg,M,0);

end

function mod_data = Psk16Modulator(msg )
M =16;
mod_data = pskmod(msg,M,0);

end

function mod_data = QAM16Modulator(msg )
M =16;
mod_data = qammod(msg,M);

end
function mod_data = QAM8Modulator(msg )
M =8;
mod_data = qammod(msg,M);

end
function mod_data = QAM32Modulator(msg )
M =32;
mod_data = qammod(msg,M);

end
function mod_data = QAM64Modulator(msg )
M =64;
mod_data = qammod(msg,M);

end
function mod_data = QAM128Modulator(msg )
M =128;
mod_data = qammod(msg,M);

end
function mod_data = QAM256Modulator(msg )
M =256;
mod_data = qammod(msg,M);

end
function mod_data = QAM512Modulator(msg )
M =512;
mod_data = qammod(msg,M);

end
function mod_data = QAM1024Modulator(msg )
M =1024;
mod_data = qammod(msg,M);

end
function mod_data = Apsk16Modulator(msg ,radii)
M = [4 12];
mod_data = apskmod(msg,M,radii);

end
function mod_data = Apsk32Modulator(msg ,radii)
M = [4 12 16];
mod_data = apskmod(msg,M,radii);

end