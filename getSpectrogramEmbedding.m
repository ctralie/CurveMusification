function [ DelaySeries, SampleDelays, hopSize, winSize ] = getSpectrogramEmbedding( soundfilename, windowSizeSec, hopSizeSec, filePrefix )
    [X, Fs] = audioread(soundfilename);
    X = mean(X, 2);
    hopSize = round(Fs*hopSizeSec);
    HopsPerWin = round(Fs*windowSizeSec/hopSize);
    winSize = HopsPerWin*hopSize;
    
    NWindows = floor((length(X) - winSize)/winSize);
    S = cell(NWindows, 1);
    
    for ii = 1:NWindows
        fprintf(1, '%i of %i\n', ii, NWindows);
        x = X((ii-1)*winSize + (1:2*winSize-hopSize));
        thisS = abs(spectrogram(x, winSize, winSize - hopSize))';
        cutoff = round(size(thisS, 2)*4000/Fs);
        S{ii} = thisS(:, 2:cutoff);
    end

    DelaySeries = cell2mat(S);
    DelaySeries = bsxfun(@minus, mean(DelaySeries, 1), DelaySeries);
    
    %Perform a random projection on the mean-centered point cloud down
    %to 3 dimensions
    U = randn(size(DelaySeries, 2), 3);
    U = bsxfun(@times, sqrt(sum(U.*U)), U);
    DelaySeries = DelaySeries*U;
    SampleDelays = (0:size(DelaySeries, 1))*hopSize/Fs;
    
    save(sprintf('%sSTFT.mat', filePrefix), 'DelaySeries', 'SampleDelays', 'soundfilename', 'Fs');
end

