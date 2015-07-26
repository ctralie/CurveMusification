filename = 'WeWantARock.wav';
[X, Fs] = audioread(filename);
NSecs = 1.5;
if length(size(X)) > 1
    X = mean(X, 2);
end
NSamples = round(NSecs*Fs);
X = X(1:NSamples);
WLen = round(Fs/2);

Y = zeros(NSamples-WLen+1, WLen);
for ii = 1:WLen
    ii
    Y(ii, :) = X(ii:ii+WLen-1);
end

[~, S, PCs] = fsvd(Y, 500);
Z = Y*PCs;
Proj = Z*PCs';

XNew = zeros(size(X));
NContrib = zeros(size(X));
for ii = 1:size(Proj, 1)
    ii
    idx = ii:ii+WLen-1;
    XNew(idx) = XNew(idx) + Proj(ii, :)';
    NContrib(idx) = NContrib(idx) + 1;
end
XNew = bsxfun(@times, 1./NContrib, XNew);