import librosa
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from OpenGL.GL import *
from OpenGL.arrays import vbo
from pylab import cm
from Cameras3D import *

class LaplacianSound(object):
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
        self.Y = np.array([[]]) #PCA on mel spectrogram
        self.PCs = np.array([[]]) #Principal components on mel spectrogram
        self.YBuf = None #Vertex buffer for PCA on mel spectrogram
        self.YColorsBuf = None #Vertex colors
        self.varExplained = 0.0 #Variance explained

    def loadAudio(self, filename):
        self.filename = filename
        self.XAudio, self.Fs = librosa.load(filename)
        self.XAudio = librosa.core.to_mono(self.XAudio)        

    def processSpecgram(self, winSize, hopSize, pfmax):
        [self.winSize, self.hopSize, self.fmax] = [winSize, hopSize, pfmax]
        self.S = librosa.core.stft(self.XAudio, winSize, hopSize)
        self.M = librosa.filters.mel(self.Fs, winSize, fmax = pfmax)
        self.X = self.M.dot(np.abs(self.S))

    def doPCA(self):
        #Do PCA on mel-spaced STFT and update vertex and color buffers
        X = self.X.T - np.mean(self.X.T, 0)
        D = (X.T).dot(X)/X.shape[0]
        (lam, eigvecs) = np.linalg.eig(D)
        lam = np.abs(lam)
        self.varExplained = np.sum(lam[0:3])/np.sum(lam)
        self.PCs = eigvecs[:, 0:3]
        self.Y = X.dot(self.PCs)
        if self.YBuf:
            self.YBuf.delete()
        self.YBuf = vbo.VBO(np.array(self.Y, dtype=np.float32))
        if self.YColorsBuf:
            self.YColorsBuf.delete()
        cmConvert = cm.get_cmap('jet')
        YColors = cmConvert(np.linspace(0, 1, self.Y.shape[0]))[:, 0:3]
        self.YColorsBuf = vbo.VBO(np.array(YColors, dtype=np.float32))
    
    def getBBox(self):
        bbox = BBox3D()
        bbox.fromPoints(self.Y)
        return bbox
    
    def bindBuffers(self):
        self.YBuf.bind()
        glEnableClientState(GL_VERTEX_ARRAY)
        glVertexPointerf(self.YBuf)
        self.YColorsBuf.bind()
        glEnableClientState(GL_COLOR_ARRAY)
        glColorPointerf(self.YColorsBuf)
    
    def unbindBuffers(self):
        self.YBuf.unbind()
        self.YColorsBuf.unbind()
        glDisableClientState(GL_VERTEX_ARRAY)
        glDisableClientState(GL_COLOR_ARRAY)

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
