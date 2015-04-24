%Shift up by how many semitones
function [ Y ] = pitchShift( S, Fs, W, H, semitones )
    bins = 0:size(S, 2) - 1;
    freqs = Fs*bins/size(S, 2);
    freqs = freqs*2^(-semitones/12);
    bins = (freqs/Fs)*size(S, 2);
    
    [OrigX, OrigY] = meshgrid(bins, 1:size(S, 1));
    [FinalX, FinalY] = meshgrid(0:size(S, 2) - 1, 1:size(S, 1));
    SShift = interp2(OrigX, OrigY, S, FinalX, FinalY);
    Y = griffinLimInverse( SShift, W, H, 10 );
end

