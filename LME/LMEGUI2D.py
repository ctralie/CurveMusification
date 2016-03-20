#Based off of http://wiki.wxpython.org/GLCanvas
#Lots of help from http://wiki.wxpython.org/Getting%20Started
from OpenGL.GL import *
from OpenGL.arrays import vbo
import wx
from wx import glcanvas

from Cameras3D import *
from LaplacianSound import *
from struct import *
from sys import exit, argv
import random
import numpy as np
import scipy.io as sio
from pylab import cm
import os
import math
import time
from time import sleep
from pylab import cm
import matplotlib.pyplot as plt

import pygame

DEFAULT_SIZE = wx.Size(1200, 800)
DEFAULT_POS = wx.Point(10, 10)


#GUI States
(STATE_NORMAL, STATE_CHOOSEVERTICES, STATE_CHOOSELAPLACEVERTICES) = (0, 1, 2)
#Laplacian substates
(SUBSTATE_NONE, CHOOSELAPLACE_WAITING, CHOOSELAPLACE_PICKVERTEX) = (0, 1, 2)

def saveImageGL(mvcanvas, filename):
    view = glGetIntegerv(GL_VIEWPORT)
    img = wx.EmptyImage(view[2], view[3] )
    pixels = glReadPixels(0, 0, view[2], view[3], GL_RGB,
                     GL_UNSIGNED_BYTE)
    img.SetData( pixels )
    img = img.Mirror(False)
    img.SaveFile(filename, wx.BITMAP_TYPE_PNG)

def saveImage(canvas, filename):
    s = wx.ScreenDC()
    w, h = canvas.size.Get()
    b = wx.EmptyBitmap(w, h)
    m = wx.MemoryDCFromDC(s)
    m.SelectObject(b)
    m.Blit(0, 0, w, h, s, 70, 0)
    m.SelectObject(wx.NullBitmap)
    b.SaveFile(filename, wx.BITMAP_TYPE_PNG)

class MeshViewerCanvas(glcanvas.GLCanvas):
    def initCircle(self):
        NPoints = 200
        t = np.linspace(0, 2*np.pi, int(NPoints*1.5))[0:NPoints]
        Y = np.zeros((len(t), 2))
        Y[:, 0] = 400 + 200*np.cos(t)
        Y[:, 1] = 400 + 200*np.sin(t)
        cmConvert = cm.get_cmap('jet')
        YColors = cmConvert(np.linspace(0, 1.0, Y.shape[0]))[:, 0:3]
        if self.sound.YBuf:
            self.sound.YBuf.delete()
        self.sound.Y = Y
        self.sound.YBuf = vbo.VBO(np.array(self.sound.Y, dtype=np.float32))
        if self.sound.YColorsBuf:
            self.sound.YColorsBuf.delete()
        self.sound.YColorsBuf = vbo.VBO(np.array(YColors, dtype=np.float32))
    
    def __init__(self, parent):
        attribs = (glcanvas.WX_GL_RGBA, glcanvas.WX_GL_DOUBLEBUFFER, glcanvas.WX_GL_DEPTH_SIZE, 24)
        glcanvas.GLCanvas.__init__(self, parent, -1, attribList = attribs)    
        self.context = glcanvas.GLContext(self)
        
        self.GUIState = STATE_CHOOSEVERTICES
        self.GUISubstate = SUBSTATE_NONE
        
        self.parent = parent
        #Camera state variables
        self.size = self.GetClientSize()
        
        #Main state variables
        self.MousePos = [0, 0]
        self.initiallyResized = False
        
        self.bbox = BBox3D()
        random.seed()
        
        #Sound Variables
        self.sound = LaplacianSound()
        self.initCircle()
        
        #State variables for laplacian mesh operations
        self.laplacianConstraints = {} #Elements will be key-value pairs (idx, Point3D(new position))
        self.laplaceCurrentIdx = -1
        
        #Other GUI Variables
        self.DrawEdges = True
        
        self.GLinitialized = False
        #GL-related events
        wx.EVT_ERASE_BACKGROUND(self, self.processEraseBackgroundEvent)
        wx.EVT_SIZE(self, self.processSizeEvent)
        wx.EVT_PAINT(self, self.processPaintEvent)
        #Mouse Events
        wx.EVT_LEFT_DOWN(self, self.MouseDown)
        wx.EVT_LEFT_UP(self, self.MouseUp)
        wx.EVT_RIGHT_DOWN(self, self.MouseDown)
        wx.EVT_RIGHT_UP(self, self.MouseUp)
        wx.EVT_MIDDLE_DOWN(self, self.MouseDown)
        wx.EVT_MIDDLE_UP(self, self.MouseUp)
        wx.EVT_MOTION(self, self.MouseMotion)        

    #######Laplacian mesh menu handles
    def doLaplacianMeshSelectVertices(self, evt):
        if self.sound.YBuf:
            self.sound.updateIndexDisplayList()
            self.GUIState = STATE_CHOOSELAPLACEVERTICES
            self.GUISubstate = CHOOSELAPLACE_WAITING
            self.Refresh()
    
    def doSelectVertices(self, evt):
        self.GUIState = STATE_CHOOSEVERTICES
        self.Refresh()
    
    def clearLaplacianMeshSelection(self, evt):
        self.laplacianConstraints.clear()
        self.Refresh()
    
    def doLaplacianSolveWithConstraints(self, evt):
        anchorWeights = 1#np.mean(np.std(self.sound.Y, 0))
        anchors = np.zeros((len(self.laplacianConstraints), 2))
        i = 0
        anchorsIdx = []
        for anchor in self.laplacianConstraints:
            anchorsIdx.append(anchor)
            anchors[i, :] = self.laplacianConstraints[anchor]
            i += 1
        self.sound.doLaplacianWarp(anchorsIdx, anchors, anchorWeights, 2)
        self.Refresh()
    
    def processEraseBackgroundEvent(self, event): pass #avoid flashing on MSW.

    def setup2DProjectionMatrix(self):
        #Set up projection matrix
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluOrtho2D(0.0, float(self.size.width), 0.0, float(self.size.height))
        
    def processSizeEvent(self, event):
        self.size = self.GetClientSize()
        self.SetCurrent(self.context)
        glViewport(0, 0, self.size.width, self.size.height)
        self.setup2DProjectionMatrix()


    def processPaintEvent(self, event):
        dc = wx.PaintDC(self)
        self.SetCurrent(self.context)
        if not self.GLinitialized:
            self.initGL()
            self.GLinitialized = True
        self.repaint()

    def drawStandard(self, PointSize = 3):
        glDisable(GL_LIGHTING)
        glPointSize(PointSize)
        NPoints = self.sound.Y.shape[0]
        
        self.sound.bindBuffers()
        if self.DrawEdges:
            if NPoints % 2 == 0:
                glDrawArrays(GL_LINES, 0, NPoints)
                glDrawArrays(GL_LINES, 1, NPoints-2)
            else:
                glDrawArrays(GL_LINES, 0, NPoints-1)
                glDrawArrays(GL_LINES, 1, NPoints-1)
        glDrawArrays(GL_POINTS, 0, NPoints)
        self.sound.unbindBuffers()

    def repaint(self):
        #Set up projection matrix
        self.setup2DProjectionMatrix()
        
        #Set up modelview matrix
        glClearColor(0.0, 0.0, 0.0, 0.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    
        if self.GUIState == STATE_NORMAL or self.GUIState == STATE_CHOOSEVERTICES:
            if self.sound.Y.size > 0:
                self.drawStandard()   
        elif self.GUIState == STATE_CHOOSELAPLACEVERTICES:
            self.drawStandard()
            if self.GUISubstate == CHOOSELAPLACE_WAITING:
                glDisable(GL_LIGHTING)
                glPointSize(POINT_SIZE)
                glBegin(GL_POINTS)
                for idx in self.laplacianConstraints:
                    P = self.sound.Y[idx, :]
                    glColor3f(0, 1, 0)
                    glVertex2f(P[0], P[1])
                    P = self.laplacianConstraints[idx]
                    if idx == self.laplaceCurrentIdx:
                        glColor3f(1, 0, 0)
                    else:
                        glColor3f(0, 0, 1)
                    glVertex2f(P[0], P[1])
                glEnd()
                glColor3f(1, 1, 0)
                glBegin(GL_LINES)
                for idx in self.laplacianConstraints:
                    P1 = self.sound.Y[idx, :]
                    P2 = self.laplacianConstraints[idx]
                    glVertex2f(P1[0], P1[1])
                    glVertex2f(P2[0], P2[1])
                glEnd()
                self.drawStandard(POINT_SIZE)
                
            elif self.GUISubstate == CHOOSELAPLACE_PICKVERTEX:
                glClearColor(0.0, 0.0, 0.0, 0.0)
                glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
                glCallList(self.sound.IndexDisplayList)
                pixel = glReadPixels(self.MousePos[0], self.MousePos[1], 1, 1, GL_RGBA, GL_UNSIGNED_BYTE)
                [R, G, B, A] = [int(pixel.encode("hex")[i*2:(i+1)*2], 16) for i in range(4)]
                idx = extractFromRGBA(R, G, B, 0) - 1
                if idx >= 0 and idx < self.sound.Y.shape[0]:
                    if idx in self.laplacianConstraints:
                        #De-select if it's already selected
                        self.laplaceCurrentIdx = -1
                        self.laplacianConstraints.pop(idx, None)
                    else:
                        self.laplacianConstraints[idx] = np.array(self.sound.Y[idx, :])
                        self.laplaceCurrentIdx = idx
                self.GUISubstate = CHOOSELAPLACE_WAITING
                self.Refresh()
            
        self.SwapBuffers()
    
    def initGL(self):        
        glLightModelfv(GL_LIGHT_MODEL_AMBIENT, [0.2, 0.2, 0.2, 1.0])
        glLightModeli(GL_LIGHT_MODEL_LOCAL_VIEWER, GL_TRUE)
        glLightfv(GL_LIGHT0, GL_DIFFUSE, [1.0, 1.0, 1.0, 1.0])
        glEnable(GL_LIGHT0)
        glLightfv(GL_LIGHT1, GL_DIFFUSE, [0.5, 0.5, 0.5, 1.0])
        glEnable(GL_LIGHT1)
        glEnable(GL_NORMALIZE)
        glEnable(GL_LIGHTING)
        glEnable(GL_DEPTH_TEST)

    def handleMouseStuff(self, x, y):
        #Invert y from what the window manager says
        y = self.size.height - y
        self.MousePos = [x, y]

    def MouseDown(self, evt):
        state = wx.GetMouseState()
        x, y = evt.GetPosition()
        if self.GUIState == STATE_CHOOSELAPLACEVERTICES:
            if state.ShiftDown():
                #Pick vertex for laplacian mesh constraints
                self.GUISubstate = CHOOSELAPLACE_PICKVERTEX
        elif self.GUIState == STATE_CHOOSEVERTICES:
            P = np.array([[float(x), self.size.height-float(y)]])
            if self.sound.Y.size == 0:
                self.sound.Y = P
            else:
                self.sound.Y = np.concatenate((self.sound.Y, P), 0)
            idx = self.sound.Y.shape[0]
            cmConvert = cm.get_cmap('jet')
            YColors = cmConvert(np.arange(idx))[:, 0:3]
            if self.sound.YBuf:
                self.sound.YBuf.delete()
            self.sound.YBuf = vbo.VBO(np.array(self.sound.Y, dtype=np.float32))
            if self.sound.YColorsBuf:
                self.sound.YColorsBuf.delete()
            cmConvert = cm.get_cmap('jet')
            YColors = cmConvert(np.arange(self.sound.Y.shape[0]))[:, 0:3]
            self.sound.YColorsBuf = vbo.VBO(np.array(YColors, dtype=np.float32))
        self.CaptureMouse()
        self.handleMouseStuff(x, y)
        self.Refresh()
    
    def MouseUp(self, evt):
        x, y = evt.GetPosition()
        self.handleMouseStuff(x, y)
        self.ReleaseMouse()
        self.Refresh()

    def MouseMotion(self, evt):
        state = wx.GetMouseState()
        x, y = evt.GetPosition()
        [lastX, lastY] = self.MousePos
        self.handleMouseStuff(x, y)
        dX = self.MousePos[0] - lastX
        dY = self.MousePos[1] - lastY
        if evt.Dragging():
            idx = self.laplaceCurrentIdx
            if self.GUIState == STATE_CHOOSELAPLACEVERTICES and state.ControlDown() and self.laplaceCurrentIdx in self.laplacianConstraints:
                xnew = float(x)
                ynew = float(self.size.height-y)
                self.laplacianConstraints[idx] = np.array([xnew, ynew])
        self.Refresh()

class MeshViewerFrame(wx.Frame):
    (ID_LoadSound, ID_SAVESCREENSHOT, ID_SELECTVERTICES, ID_SELECTLAPLACEVERTICES, ID_CLEARLAPLACEVERTICES, ID_SOLVEWITHCONSTRAINTS) = (1, 2, 3, 4, 5, 6)
    
    def __init__(self, parent, id, title, pos=DEFAULT_POS, size=DEFAULT_SIZE, style=wx.DEFAULT_FRAME_STYLE, name = 'GLWindow'):
        style = style | wx.NO_FULL_REPAINT_ON_RESIZE
        super(MeshViewerFrame, self).__init__(parent, id, title, pos, size, style, name)
        #Initialize the menu
        self.CreateStatusBar()
        
        self.size = size
        self.pos = pos
        print "MeshViewerFrameSize = %s, pos = %s"%(self.size, self.pos)
        self.glcanvas = MeshViewerCanvas(self)
        
        #####File menu
        filemenu = wx.Menu()
        menuSaveScreenshot = filemenu.Append(MeshViewerFrame.ID_SAVESCREENSHOT, "&Save Screenshot", "Save a screenshot of the GL Canvas")
        self.Bind(wx.EVT_MENU, self.OnSaveScreenshot, menuSaveScreenshot)
        
        menuExit = filemenu.Append(wx.ID_EXIT,"E&xit"," Terminate the program")
        self.Bind(wx.EVT_MENU, self.OnExit, menuExit)
        
        #####Laplacian Mesh Menu
        laplacianMenu = wx.Menu()
        menuSelectVertices = laplacianMenu.Append(MeshViewerFrame.ID_SELECTVERTICES, "&Select Vertices", "Select Vertices")
        self.Bind(wx.EVT_MENU, self.glcanvas.doSelectVertices, menuSelectVertices)
        menuSelectLaplaceVertices = laplacianMenu.Append(MeshViewerFrame.ID_SELECTLAPLACEVERTICES, "&Select Laplace Vertices", "Select Laplace Vertices")
        self.Bind(wx.EVT_MENU, self.glcanvas.doLaplacianMeshSelectVertices, menuSelectLaplaceVertices)
        menuClearLaplaceVertices = laplacianMenu.Append(MeshViewerFrame.ID_CLEARLAPLACEVERTICES, "&Clear vertex selection", "Clear Vertex Selection")
        self.Bind(wx.EVT_MENU, self.glcanvas.clearLaplacianMeshSelection, menuClearLaplaceVertices)
        menuSolveWithConstraints = laplacianMenu.Append(MeshViewerFrame.ID_SOLVEWITHCONSTRAINTS, "&Solve with Constraints", "Solve with Constraints")
        self.Bind(wx.EVT_MENU, self.glcanvas.doLaplacianSolveWithConstraints, menuSolveWithConstraints)
        
        # Creating the menubar.
        menuBar = wx.MenuBar()
        menuBar.Append(filemenu,"&File") # Adding the "filemenu" to the MenuBar
        menuBar.Append(laplacianMenu,"&MeshLaplacian") # Adding the "filemenu" to the MenuBar
        self.SetMenuBar(menuBar)  # Adding the MenuBar to the Frame content.
        
        self.rightPanel = wx.BoxSizer(wx.VERTICAL)
        
        
        #Finally add the two main panels to the sizer
        self.sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.sizer.Add(self.glcanvas, 2, wx.EXPAND)
        self.sizer.Add(self.rightPanel, 0, wx.EXPAND)
        
        self.SetSizer(self.sizer)
        self.Layout()
        #self.SetAutoLayout(1)
        #self.sizer.Fit(self)
        self.Show()

    def OnSaveScreenshot(self, evt):
        dlg = wx.FileDialog(self, "Choose a file", ".", "", "*", wx.SAVE)
        if dlg.ShowModal() == wx.ID_OK:
            filename = dlg.GetFilename()
            dirname = dlg.GetDirectory()
            filepath = os.path.join(dirname, filename)
            saveImageGL(self.glcanvas, filepath)
        dlg.Destroy()
        return

    def OnExit(self, evt):
        self.Close(True)
        return

class MeshViewer(object):
    def __init__(self, filename = None, ts = False, sp = "", ra = 0):
        app = wx.App()
        frame = MeshViewerFrame(None, -1, 'MeshViewer')
        frame.Show(True)
        app.MainLoop()
        app.Destroy()

if __name__ == '__main__':
    viewer = MeshViewer()
