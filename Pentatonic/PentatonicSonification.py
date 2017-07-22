import numpy as np
import matplotlib
#matplotlib.use("Agg")
import librosa
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy
import scipy.io as sio
from sklearn.decomposition import PCA
from SlidingWindow import *
import subprocess
import os

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
    print "X.shape[0] = ", X.shape[0]
    print "fac*X.shape[0] = ", fac*X.shape[0]
    
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


#A class for doing animation of the sliding window
class SignalAnimator(animation.FuncAnimation):
    def __init__(self, fig, x, FsIn, Win):
        self.fig = fig
        self.x = x
        self.FsIn = FsIn
        Win = int(Win)
        self.Win = Win
        
        Win = int(Win)
        c = plt.get_cmap('Spectral')
        C = c(np.array(np.round(np.linspace(0, 255, len(x))), dtype=np.int32))
        self.C = C[:, 0:3]
        self.xmin = np.min(x) - 0.2*(np.max(x) - np.min(x))
        self.xmax = np.max(x) + 0.2*(np.max(x) - np.min(x))

        ax1 = fig.add_subplot(111)
        self.ax1 = ax1

        #Original curve
        self.origCurve, = ax1.plot(x, 'k')
        ax1.hold(True)
        self.windowPlot, = ax1.plot([0], [x[0]], c = C[0, :])
        self.leftLim, = ax1.plot([0, 0], [self.xmin, self.xmax], c=C[0, :])
        self.rightLim, = ax1.plot([Win, Win], [self.xmin, self.xmax], c=C[0, :], lineWidth=2)

        #Setup animation thread
        W = len(x) - Win + 1
        animation.FuncAnimation.__init__(self, fig, func = self._draw_frame, frames = W, interval = 10)

        #Write movie
        FFMpegWriter = animation.writers['ffmpeg']
        metadata = dict(title='Sliding Window Sonification',
                        comment='Check it out!')
        writer = FFMpegWriter(fps=FsIn, metadata=metadata, bitrate = 30000)
        self.save("temp.mp4", writer = writer)

    def _draw_frame(self, i):
        print i
        self.windowPlot.set_xdata(np.arange(i, i+self.Win))
        self.windowPlot.set_ydata(self.x[i:i+self.Win])
        self.windowPlot.set_color(self.C[i, :])
        self.leftLim.set_xdata([i, i])
        self.leftLim.set_ydata([self.xmin, self.xmax])
        self.leftLim.set_color(self.C[i, :])
        self.rightLim.set_xdata([i+self.Win-1, i+self.Win-1])
        self.rightLim.set_ydata([self.xmin, self.xmax])
        self.rightLim.set_color(self.C[i, :])


def makeSyncedVideo(x, FsIn, xout, FsOut, Win, filename):
    sio.wavfile.write("temp.wav", FsOut, xout)
    
    fig = plt.figure()
    ani = SignalAnimator(fig, x, FsIn, Win)
    
    subprocess.call(["avconv", "-i", "temp.mp4", "-i", "temp.wav", "-b", "30000k", filename])        
    os.remove("temp.mp4")
    os.remove("temp.wav")


if __name__ == '__main__':
    #Step 1: Setup the signal
    T1 = 10 #The period of the first sine in number of samples
    T2 = T1*2 #The period of the second sine in number of samples
    NPeriods = 15 #How many periods to go through, relative to the second sinusoid
    N = T1*3*NPeriods #The total number of samples
    t = np.arange(N)*np.pi/3 #Time indices
    x = np.cos(2*np.pi*(1.0/T1)*t) #The first sinusoid
    x += np.cos(2*np.pi*(1.0/T2)*t) #The second sinusoid
    x = x + 2*np.random.randn(len(x))
    #x = 5*np.arange(400)


    dim = 20
    Tau = 1
    dT = 1
    X = getSlidingWindow(x, dim, Tau, dT)
    X = X - np.mean(X, 1)[:, None]
    X = X/np.sqrt(np.sum(X**2, 1))[:, None]
    pca = PCA(n_components = 10)
    Y = pca.fit_transform(X)

    FsOut = 22050
    FsIn = 25
    xout = makePentatonicSliding(Y, FsIn, FsOut)
    makeSyncedVideo(x, FsIn, xout, FsOut, dim*Tau, "noisycost_cos2t.avi")
    
