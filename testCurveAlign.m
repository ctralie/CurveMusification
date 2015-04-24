W = 16384;
H = 32;
[X, Fs] = audioread('AddictedToLoveClip.wav');
S = STFT(X, W, H);
S = abs(S);
NBins = size(S, 2);
S = S(:, 1:150);

%Figure out the approximate scale of the spectrogram point cloud
Scale = bsxfun(@minus, mean(S, 1), S);
Scale = 2*max(sqrt(sum(Scale.^2, 2)));

t = linspace(0, 1, size(S, 1));

X = Scale*[cos(2*pi*t(:)) sin(2*pi*2*t(:))];
X = [X zeros(size(X, 1), size(S, 2) - 2)];
X = X + repmat(mean(S, 1), [size(X, 1), 1]);

[T, ~, iters] = FICP(X, S, 0.5, 10);
X = [X ones(size(X, 1), 1)];
X = (T*X')';
X = X(:, 1:end-1);
X(X < 0) = 0;

X = [X zeros(size(X, 1), NBins - size(X, 2))];
Y = griffinLimInverse(X, W, H, 10);
audiowrite('CircleTry1.wav', Y, Fs);