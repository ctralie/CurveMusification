function [ X ] = griffinLimInverse( S, W, H, NIters )
    %Assume signal came from STFT using half sine window and that S
    %is magnitude only
    A = S;
    for ii = 1:NIters
        fprintf(1, 'Iteration %i of %i\n', ii, NIters);
        A = STFT(iSTFT(A, W, H), W, H);
        Norm = sqrt(A.*conj(A));
        Norm(Norm < eps) = 1;
        A = S.*(A./Norm);
    end
    X = iSTFT(A, W, H);
end

