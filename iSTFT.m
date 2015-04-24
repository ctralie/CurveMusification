function [ X ] = iSTFT( S, W, H, winfunc )
    %S: N x 1 Audio Signal, W: Window Size: H HopSize, winfunc: Window
    %function
    %First put back the entire redundant STFT
    if mod(W, 2) == 0
        %Even Case
        S2 = S(:, 2:end-1);
        S = [S fliplr(conj(S2))];
    else
        %Odd Case
        S2 = S(:, 2:end);
        S = [S fliplr(conj(S2))];
    end
    
    %Figure out how long the reconstructed signal actually is
    N = W + H*(size(S, 1) - 1);
    X = zeros(N, 1);
    
    %Setup the window
    Q = W/H;
    if Q - floor(Q) > 0
        fprintf(1, 'Warning: Window size is not integer multiple of hop size\n');
    end
    if nargin < 4
        %Use half sine by default
        winfunc = @(W) sin(pi*(0:W-1)/(W-1));
        win = winfunc(W);
        win = win/(Q/2);
    end
    
    %Do overlap/add synthesis
    for ii = 1:size(S, 1)
        i1 = 1 + (ii-1)*H;
        i2 = i1 + W - 1;
        X(i1:i2) = X(i1:i2) + ( win(:)'.*ifft(S(ii, :)) )';
    end
end

