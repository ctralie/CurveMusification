import librosa
import numpy as np
import scipy.io as sio
from scipy import sparse
import scipy.sparse.linalg as slinalg
import matplotlib.pyplot as plt
from pylab import cm
from oct2py import octave
octave.addpath('nmflib') #Needed to use NMF library
from VideoTools import *
import scipy.interpolate as interp
import subprocess

class GTzanEnvelopeDatabase(object):
    def __init__(self, winSize, hopSize, genres, NDictElems):
        [self.winSize, self.hopSize] = [winSize, hopSize]
        Xs = np.array([])
        for genre in genres:
            for i in range(100):
                filename = "genres/%s/%s.%.5i.au"%(genre, genre, i)
                print "Processing ", filename
                [XAudio, Fs] = librosa.load(filename)
                S = librosa.core.stft(XAudio, winSize, hopSize)
                M = librosa.filters.mel(Fs, winSize)
                X = M.dot(np.abs(S))
                if Xs.size == 0:
                    Xs = X
                else:
                    Xs = np.concatenate((Xs, X), 1)
        self.Xs = Xs
        print "Doing NMF..."
        [W, H, errs, vout] = octave.nmf_euc_orth(Xs, NDictElems)
        print "Finished NMF"
        sio.savemat("Xs%i.mat"%NDictElems, {"Xs":Xs, "W":W, "H":H})    

class EnvelopeSound(object):
    def __init__(self):
        self.filename = ""
        self.XAudio = np.array([])
        self.Fs = 22050
        self.winSize = 16384
        self.hopSize = 512
        self.fmax = 8000
        self.S = np.array([[]]) #Spectrogram
        self.M = np.array([[]]) #Mel filterbank
        self.X = np.array([[]]) #Mel spectrogram

    def loadAudio(self, filename):
        self.filename = filename
        fileparts = filename.split(".")
        if not fileparts[-1] == "wav":
            os.remove("temp.wav")
            subprocess.call(["avconv", "-i", filename, "temp.wav"])
            #self.XAudio, self.Fs = librosa.load("temp.wav")
            self.Fs, self.XAudio = sio.wavfile.read("temp.wav")
            self.XAudio = self.XAudio.T
        else:
            #self.XAudio, self.Fs = librosa.load(filename)
            self.Fs, self.XAudio = sio.wavfile.read(filename)
            self.XAudio = self.XAudio.T
            print self.XAudio.shape
        print "Fs = %i"%self.Fs
        self.XAudio = librosa.core.to_mono(self.XAudio)
        #Convert to a format that can be played by pygame
        if os.path.exists("temp.ogg"):
            os.remove("temp.ogg")
        subprocess.call(["avconv", "-i", filename, "temp.ogg"])

    def makeWhiteNoiseSignal(self, Fs, NSamples):
        self.Fs = Fs
        self.XAudio = 0.1*np.random.randn(NSamples)
        self.XAudio[self.XAudio > 1] = 1
        self.XAudio[self.XAudio < -1] = -1

    def processSpecgram(self, winSize, hopSize):
        [self.winSize, self.hopSize] = [winSize, hopSize]
        self.S = librosa.core.stft(self.XAudio, winSize, hopSize)
        self.M = librosa.filters.mel(self.Fs, winSize)
        self.X = self.M.dot(np.abs(self.S))

    def getSampleDelay(self, i):
        if i == -1:
            i = self.Y.shape[0]-1
        return float(i)*self.hopSize/self.Fs

    #Export to web interface
    def exportToLoopDitty():
        Y = self.Y
        #Output information text file
        fout = open("%s.txt"%outprefix, "w")
        for i in range(Y.shape[0]):
            fout.write("%g,%g,%g,%g,"%(Y[i, 0], Y[i, 1], Y[i, 2], i*float(self.hopSize)/self.Fs))
        fout.write("%g"%(np.sum(lam[0:3])/np.sum(lam)))
        fout.close()
        #Output audio information
        sio.wavfile.write("%s.wav"%outprefix, Fs, XAudio)
        
    def invertNewMelSpectrum(self, XNew):
        #Step 1: Create a new STFT with a scaled envelope
        MEnergy = np.sum(self.M, 0)
        idxpass = (MEnergy == 0)
        MEnergy[idxpass] = 1
        Ratio = XNew/(self.X + 0j)
        Ratio[self.X == 0] = 1
        SNew = np.zeros(self.S.shape) + 0j
        for i in range(SNew.shape[1]):
            for k in range(self.M.shape[0]):
                SNew[:, i] += ((self.M[k, :].flatten())*(self.S[:, i].flatten()))*Ratio[k, i]
        SNew = SNew/MEnergy[:, None] #Normalize Mel-spaced envelope
        SNew[idxpass, :] = self.S[idxpass, :] #Passband (
        #Step 2: Perform Griffin Lim iterative phase retrieval
        y = librosa.core.istft(SNew, self.hopSize, self.winSize)
        y = y/np.max(abs(y)) #Prevent clippling (TODO: something smarter?)
        return y

def getVideoResampled(filename, NPCs, DelayWin, hopSize, winSize, Fs, framerate):
    print "Processing video..."
    (I, IDims) = loadCVVideo(filename)
    (data_subspace, s_I) = tde_tosubspace(I, I.shape[1])
    s_I_mean = tde_mean(s_I,DelayWin)
    (X, S) = tde_rightsvd(s_I,DelayWin,s_I_mean)
    X = X*S[None, :]
    M = X.shape[0]
    hopSec = float(hopSize)/Fs
    idx = np.arange(M)/float(framerate)
    NHops = np.ceil((idx[-1]-float(winSize-hopSize)/Fs)/hopSec)
    print "NHops = ", NHops
    idxx = np.arange(NHops)*hopSec
    Y = np.zeros((len(idxx), NPCs))
    for i in range(NPCs):
        Y[:, i] = interp.spline(idx, X[:, i], idxx)
    print "Finished processing video"
    #Apply a random rotation
    Y = Y.T
    (_, _, R) = np.linalg.svd(np.random.randn(NPCs, NPCs))
    Y = Y - np.mean(Y, 1)[:, None]
    Y = R.dot(Y)
    #Put in nonnegative orthant
    Y = Y - np.min(Y, 1)[:, None]
    return (Y, idxx)

if __name__ == '__main__':
#    d = GTzanEnvelopeDatabase(2048, 256, ['hiphop'], 20)
    
    #Load spectral NMF dictionary
    d = sio.loadmat("Xs20_8192.mat")
    W = d['W']
    H = d['H']
    idx = np.argsort(np.var(H, 1))
    W = W[:, idx]
    NW = W.shape[1]
    
    #Load video
    Fs = 22050
    hopSize = 128
    winSize = 8192
    NPCs = NW
    DelayWin = 10
    framerate = 30.0
    videoname = 'Videos/surveillance_walking.mp4'
    outname = 'walking'
    #videoname = 'Videos/jumpingjackscropped.avi'
    #outname = 'jumpingjacks'
    #videoname = 'Videos/BUGroundTruth.ogg'
    #videoname = 'Videos/baby.avi'
    #outname = 'baby'
    (Y, idxx) = getVideoResampled(videoname, NPCs, DelayWin, hopSize, winSize, Fs, framerate)
    plt.imshow(Y.T, interpolation = 'none', aspect = 'auto')
    plt.show()
    
    
    s = EnvelopeSound()
    s.makeWhiteNoiseSignal(Fs, int(np.round(Fs*idxx[-1])))
    s.processSpecgram(winSize, hopSize)
    

    XNew = W.dot(Y)
    XNew = XNew/np.max(XNew)
    plt.subplot(121)
    plt.imshow(W, aspect = 'auto', interpolation = 'none')
    plt.axis('off')
    plt.title('NMF Dictionary')
    plt.subplot(122)
    plt.imshow(XNew, aspect = 'auto', interpolation = 'none')
    plt.axis('off')
    plt.title('Synthesized Mel Spectrogram')
    plt.show()
    y = s.invertNewMelSpectrum(XNew)
    sio.wavfile.write("%s.wav"%outname, Fs, y)
    subprocess.call(["avconv", "-i", "%s.wav"%outname, "-i", videoname, "-qscale", "100", "%s.ogg"%outname])
