import librosa
import numpy as np
import scipy.io as sio
import scipy.linalg
from scipy import sparse
import matplotlib.pyplot as plt
from OpenGL.GL import *
from OpenGL.arrays import vbo
from pylab import cm
from Cameras3D import *
import subprocess
import os
this_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, this_path + '/../ext/libigl/python/')
sys.path.insert(0, this_path + '/../ext/lib/')
import igl

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
        self.texID = None #Texture pointer for spectrogram image

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

    def updateSpecgramTexture(self):
        W = self.X.shape[1]
        H = self.X.shape[0]
        #x = librosa.core.logamplitude(self.X.flatten())
        x = self.X.flatten()
        x = x - np.min(x)
        x = x/np.max(x)
        cmap = plt.get_cmap('jet')
        C = np.array(np.round(255.0*cmap(x)), dtype=np.uint8).flatten()
        if not self.texID:   
            self.texID = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.texID)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, 3, W, H, 0, GL_RGBA, GL_UNSIGNED_BYTE, C)

    def processSpecgram(self, winSize, hopSize, pfmax):
        [self.winSize, self.hopSize, self.fmax] = [winSize, hopSize, pfmax]
        self.S = librosa.core.stft(self.XAudio, winSize, hopSize)
        self.M = librosa.filters.mel(self.Fs, winSize, fmax = pfmax)
        self.X = self.M.dot(np.abs(self.S))
        self.updateSpecgramTexture()

    
    def drawSpecgram(self, width, height, CurrIdx):
        if not self.texID:
            return
        glDisable(GL_DEPTH_TEST)
        glDisable(GL_LIGHTING)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluOrtho2D(0, width, 0, height)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        
        #Draw spectrogram
        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, self.texID)
        glBegin(GL_QUADS)
        H = 150
        glTexCoord2f(1.0, 0.0); glVertex2i(width, 0)
        glTexCoord2f(1.0,1.0); glVertex2i(width, H)
        glTexCoord2f(0.0, 1.0); glVertex2i(0, H)
        glTexCoord2f(0.0,0.0); glVertex2i(0, 0)
        glEnd()
        glDisable(GL_TEXTURE_2D)
        
        #Draw line segment marker
        glColor3f(1, 0, 0)
        pos = float(width)*CurrIdx/self.X.shape[1]
        glBegin(GL_LINES)
        glVertex2f(pos, 0)
        glVertex2f(pos, H)
        glEnd()
        glEnable(GL_LIGHTING)
        glEnable(GL_DEPTH_TEST)
    
    def doPCA(self, dims = 3):
        #Do PCA on mel-spaced STFT and update vertex and color buffers
        X = self.X.T - np.mean(self.X.T, 0)
        D = (X.T).dot(X)/X.shape[0]
        (lam, eigvecs) = np.linalg.eig(D)
        lam = np.abs(lam)
        idx = np.argsort(-lam)
        lam = lam[idx]
        eigvecs = eigvecs[:, idx]
        self.varExplained = np.sum(lam[0:dims])/np.sum(lam)
        self.PCs = eigvecs[:, 0:dims]
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
            if P.size == 3:
                glVertex3f(P[0], P[1], P[2])
            else:
                glVertex2f(P[0], P[1])
        glEnd()
        glEndList()
    
    def doLaplacianWarp(self, anchorsIdx, anchors, anchorWeights, dims = 3, weighted = True):
        #https://ensiwiki.ensimag.fr/index.php/Alexandre_Ribard_:_Laplacian_Curve_Editing_--_Detail_Preservation
        #Step 1: Create laplacian matrix
        N = self.Y.shape[0] #Number of vertices
        M = N - 1 #Number of edges
        L = np.zeros((N, N))
        I = np.zeros(M*2)
        J = np.zeros(M*2)
        
        V = np.ones(M*2)
        weighted = False
        if weighted:
            Ds = np.sqrt(np.sum((self.Y[1:, :] - self.Y[0:-1, :])**2, 1))
            V = np.concatenate((Ds[:, None], Ds[:, None]), 0).flatten()
        I[0:M] = np.arange(0, N-1)
        J[0:M] = np.arange(1, N)
        I[M:2*M] = np.arange(1, N)
        J[M:2*M] = np.arange(0, N-1)
        L = sparse.coo_matrix((V, (I, J)), shape=(N, N)).tocsr()
        W = np.array(L.sum(1)).flatten()
        L = sparse.dia_matrix((W, 0), L.shape) - L
        W[0] = W[0]*2.0
        W[-1] = W[-1]*2.0
        D = sparse.dia_matrix((1.0/W, 0), L.shape)
        #L = D.dot(L)
        #print L.todense()
        deltaCoords = L.dot(self.Y)
        coo = L.tocoo()
        coo = np.vstack((coo.row, coo.col, coo.data)).T
        coo = igl.eigen.MatrixXd(np.array(coo, dtype=np.float64))
        LE = igl.eigen.SparseMatrixd()
        LE.fromCOO(coo)
        
        #Step 2: Add anchors
        #TODO: Add first and last points as anchors?
        Q = (LE.transpose())*LE
        #Now add in sparse constraints
        diagTerms = igl.eigen.SparseMatrixd(N, N)
        # anchor points
        for a in anchorsIdx:
            diagTerms.insert(a, a, anchorWeights)
        Q = Q + diagTerms
        Q.makeCompressed()
        solver = igl.eigen.SimplicialLLTsparse(Q)
        
        #Step 3: Solve for new positions
        y = np.array(LE*igl.eigen.MatrixXd(np.array(deltaCoords[:, 0:dims], dtype=np.float64)))
        y[anchorsIdx] += anchorWeights*anchors      
        y = igl.eigen.MatrixXd(y)       
        ret = solver.solve(y)
        YOrig = np.array(self.Y)
        self.Y = np.array(ret)
        plt.plot(self.Y[:, 0], self.Y[:, 1], '.')
        plt.show()
        
        self.YBuf.delete()
        self.YBuf = vbo.VBO(np.array(self.Y, dtype=np.float32))
        self.updateIndexDisplayList()
        
        #Step 4: Change mel spectrum
        #TODO: Change sound
        #Subtract away original 3 components and put these in their place
        XNew = self.X - (YOrig.dot(self.PCs.T)).T
        XNew += (self.Y.dot(self.PCs.T)).T
        self.XAudio = self.invertNewMelSpectrum(XNew)
        self.X = XNew
        self.updateSpecgramTexture()
        sio.wavfile.write("temp.wav", self.Fs, self.XAudio)
        if os.path.exists("temp.ogg"):
            os.remove("temp.ogg")
        subprocess.call(["avconv", "-i", "temp.wav", "temp.ogg"])
        
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
