NY = 200;
t = linspace(0, 1, NY);
Y = [cos(2*pi*t(:)) sin(4*pi*t(:))];

K = 100;
NX = 10000;
X = 4*(rand(NX, 2) - 0.5);

XOut = traceCurve(Y, X, K, 0);