import numpy as np
import librosa
import matplotlib.pyplot as plt
import scipy
import scipy.io as sio
from sklearn.decomposition import PCA
from SlidingWindow import *

def makePentatonicSliding(X, FsIn, FsOut, f0 = 220):
    d = X.shape[1]
    #Step 1: Setup base frequencies
    fs = [f0]
    for i in range(1, d):
        if i%2 == 0:
            fs.append(fs[-1]*2**(1.0/6.0))
        else:
            fs.append(fs[-1]*2**(1.0/4.0))
    fs = np.array(fs)
    print fs

    #Step 2: Compute amplitudes of sinusoids
    Y = X - np.min(X, 0)[None, :]
    Y = Y/np.max(Y)

    fac = FsOut / FsIn
    idx1 = np.arange(X.shape[0])
    idx2 = np.arange(fac*X.shape[0])/float(fac)
    noteidx = np.arange(d)
    f = scipy.interpolate.interp2d(noteidx, idx1, Y, kind='linear')
    A = f(noteidx, idx2)

    #Step 3: Make all sinusoids
    t = np.arange(A.shape[0])/float(FsOut)
    S = A*np.cos(2*np.pi*t[:, None]*fs[None, :])
    x = np.sum(S, 1)
    x = x/np.max(x)
    return x

if __name__ == '__main__':
    #Step 1: Setup the signal
    T1 = 10 #The period of the first sine in number of samples
    T2 = T1*np.pi #The period of the second sine in number of samples
    NPeriods = 15 #How many periods to go through, relative to the second sinusoid
    N = T1*3*NPeriods #The total number of samples
    t = np.arange(N)*np.pi/3 #Time indices
    x = np.cos(2*np.pi*(1.0/T1)*t) #The first sinusoid
    x += np.cos(2*np.pi*(1.0/T2)*t) #The second sinusoid
    #x = x + 0.05*np.random.randn(len(x))


    dim = 30
    Tau = 1
    dT = 0.5
    X = getSlidingWindow(x, dim, Tau, dT)
    X = X - np.mean(X, 1)[:, None]
    X = X/np.sqrt(np.sum(X**2, 1))[:, None]
    pca = PCA(n_components = 10)
    Y = pca.fit_transform(X)

    Fs = 22050
    x = makePentatonicSliding(Y, 30, Fs)

    sio.wavfile.write("out.wav", Fs, x)

    plt.plot(x)
    plt.show()
