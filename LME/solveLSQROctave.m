load('LapVars.mat');
L = sparse(double(I+1), double(J+1), double(V), double(M), double(N));
Z = L \ y;
% LTL = L'*L;
% LTy = L'*y;
% R = chol(LTL, 'lower');
% Z = R\(R'\LTy);

dlmwrite('LapY.txt', Z, ' ');
