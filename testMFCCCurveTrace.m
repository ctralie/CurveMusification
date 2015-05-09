filename = 'MJ.mp3';

NY = 2000;
K = 100;
t = linspace(0, 1, NY);
Y = 0.5*[cos(2*pi*t(:)) sin(4*pi*t(:))];

[X, S] = getMFCCTempoWindow(filename, 0.5);
X = bsxfun(@minus, mean(X), X);
X = bsxfun(@times, 1./sqrt(sum(X.^2, 2)), X);
[~, X] = pca(X);
XProj = X(:, 1:2);

[XOut, idx] = traceCurve(Y, XProj, K, 0);

plot(XProj(:, 1), XProj(:, 2), 'r.');
hold on;
plot(XOut(:, 1), XOut(:, 2), 'b');

[SX, Fs] = audioread(filename);
if size(SX, 2) > 1
    SX = mean(SX, 2);
end

winLen = round(Fs*0.01);
XOut = zeros(NY*winLen, 1);
for ii = 1:length(idx)
    XOut((ii-1)*winLen + (1:winLen)) = SX(idx(ii):idx(ii)+winLen-1);
end
audiowrite('out.wav', XOut, Fs);