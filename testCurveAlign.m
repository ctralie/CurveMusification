% W = 16384;
% H = 32;
W = 512;
H = 256;
HEmbed = 1;
[X, Fs] = audioread('AddictedToLoveClip.wav');
S = STFT(X, W, HEmbed);
S = abs(S);
NBins = size(S, 2);
%S = S(:, 1:2600);

%Figure out the approximate scale of the spectrogram point cloud
Scale = bsxfun(@minus, mean(S, 1), S);
Scale = 0.5*max(sqrt(sum(Scale.^2, 2)));

t = linspace(0, 1, size(S, 1)/(H/HEmbed));
X = Scale*[cos(2*pi*t(:)) sin(2*pi*t(:))];

SOut = traceCurve(X, S, 10, 0);

Z = [SOut; S];
[~, Y] = pca(Z);
clf;
N = size(SOut, 1);
plot3(Y(1:N, 1), Y(1:N, 2), Y(1:N, 3), 'b.');
hold on;
plot3(Y(N+1:end, 1), Y(N+1:end, 2), Y(N+1:end, 3), 'r.');

Y = griffinLimInverse(SOut, W, H, 10);
audiowrite('CircleTry1.wav', Y, Fs);