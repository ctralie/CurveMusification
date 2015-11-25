import librosa
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from OpenGL.GL import *
from OpenGL.arrays import vbo
from pylab import cm
from Cameras3D import *
import subprocess
import os

POINT_SIZE = 10

def splitIntoRGBA(val):
    A = (0xff000000&val)>>24
    R = (0x00ff0000&val)>>16
    G = (0x0000ff00&val)>>8
    B = (0x000000ff&val)
    return [R, G, B, A]

def extractFromRGBA(R, G, B, A):
    return ((A<<24)&0xff000000) | ((R<<16)&0x00ff0000) | ((G<<8)&0x0000ff00) | (B&0x000000ff)

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
        self.IndexDisplayList = -1 #Used to help select laplacian anchors

    def loadAudio(self, filename):
        self.filename = filename
        self.XAudio, self.Fs = librosa.load(filename)
        self.XAudio = librosa.core.to_mono(self.XAudio)
        #Convert to a format that can be played by pygame
        os.remove("temp.ogg")
        subprocess.call(["avconv", "-i", filename, "temp.ogg"])

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
    
    def updateIndexDisplayList(self):
        if self.IndexDisplayList != -1: #Deallocate previous display list
            glDeleteLists(self.IndexDisplayList, 1)
        self.IndexDisplayList = glGenLists(1)
        print "Updating index display list"
        glNewList(self.IndexDisplayList, GL_COMPILE)
        glDisable(GL_LIGHTING)
        N = self.Y.shape[0]
        #First draw all of the faces with index N+1 so that occlusion is
        #taken into proper consideration
        [R, G, B, A] = splitIntoRGBA(N+2)
        glColor4ub(R, G, B, A)
        glPointSize(POINT_SIZE)
        glBegin(GL_POINTS)
        for i in range(0, N):
            P = self.Y[i, :]
            [R, G, B, A] = splitIntoRGBA(i+1)
            glColor4ub(R, G, B, A)
            glVertex3f(P[0], P[1], P[2])
        glEnd()
        glEndList()
