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
(STATE_NORMAL, STATE_PLAYINGVIDEO) = (0, 1)
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
    def __init__(self, parent):
        attribs = (glcanvas.WX_GL_RGBA, glcanvas.WX_GL_DOUBLEBUFFER, glcanvas.WX_GL_DEPTH_SIZE, 24)
        glcanvas.GLCanvas.__init__(self, parent, -1, attribList = attribs)    
        self.context = glcanvas.GLContext(self)
        
        self.GUIState = STATE_NORMAL
        self.GUISubstate = SUBSTATE_NONE
        
        self.parent = parent
        #Camera state variables
        self.size = self.GetClientSize()
        self.camera = MousePolarCamera(self.size.width, self.size.height)
        
        #Main state variables
        self.MousePos = [0, 0]
        self.initiallyResized = False
        
        self.bbox = BBox3D()
        random.seed()
        
        #Sound Variables
        self.sound = LaplacianSound()
        self.timeSlider = None
        self.Playing = False
        self.PlayIDX = 0  #Index that's being played in OpenGL
        self.timeOffset = 0 #Used to help keep track of sound position as user jumps around
        
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
        #self.initGL()

    def updateParams(self, evt):
        winSize = int(self.winSizeTxt.GetValue())
        hopSize = int(self.hopSizeTxt.GetValue())
        fmax = int(self.fmaxTxt.GetValue())
        winSize = int(self.winSizeTxt.GetValue())
        hopSize = int(self.hopSizeTxt.GetValue())
        fmax = int(self.fmaxTxt.GetValue())
        self.PlayIdx = 0
        self.timeOffset = 0
        self.timeSlider.SetValue(0)
        self.sound.processSpecgram(winSize, hopSize, fmax)
        self.sound.doPCA()
        self.bbox = self.sound.getBBox()
        self.viewFromFront(None)
        self.Refresh()

    def OnLoadSound(self, evt):
        dlg = wx.FileDialog(self, "Choose a file", ".", "", "*", wx.OPEN)
        if dlg.ShowModal() == wx.ID_OK:
            self.externalFile = False
            filename = dlg.GetFilename()
            dirname = dlg.GetDirectory()
            print "Loading %s...."%filename
            filepath = os.path.join(dirname, filename)
            self.sound.loadAudio(filepath)
            self.updateParams(None)
            print "Finished loading"
        dlg.Destroy()
        return

    def OnPlay(self, evt):
        if self.Playing:
            return
        if self.sound.Y.size > 0:
            self.Playing = True
            pygame.mixer.quit()
            pygame.mixer.init(frequency = self.sound.Fs)
            pygame.mixer.music.load("temp.ogg")
            print "Playing at ", self.timeOffset
            pygame.mixer.music.play(0, self.timeOffset)
            self.Refresh()
    
    def OnPause(self, evt):
        if self.sound.Y.size == 0:
            return
        pygame.mixer.music.stop()
        self.Playing = False
        self.timeOffset += float(pygame.mixer.music.get_pos()) / 1000.0
    
    #The user can change the position in the song
    def SliderMove(self, evt):
        pos = evt.GetPosition()
        time = self.sound.getSampleDelay(-1)*float(pos)/1000.0
        if self.Playing:
            print time
            pygame.mixer.music.play(0, time)
        self.timeOffset = time
        self.PlayIDX = 0

    def viewFromFront(self, evt):
        self.camera.centerOnBBox(self.bbox, theta = -math.pi/2, phi = math.pi/2)
        self.Refresh()
    
    def viewFromTop(self, evt):
        self.camera.centerOnBBox(self.bbox, theta = -math.pi/2, phi = 0)
        self.Refresh()
    
    def viewFromSide(self, evt):
        self.camera.centerOnBBox(self.bbox, theta = -math.pi, phi = math.pi/2)
        self.Refresh()
    
    def processEraseBackgroundEvent(self, event): pass #avoid flashing on MSW.

    def processSizeEvent(self, event):
        self.size = self.GetClientSize()
        self.SetCurrent(self.context)
        glViewport(0, 0, self.size.width, self.size.height)
        if not self.initiallyResized:
            #The canvas gets resized once on initialization so the camera needs
            #to be updated accordingly at that point
            self.camera = MousePolarCamera(self.size.width, self.size.height)
            self.camera.centerOnBBox(self.bbox, math.pi/2, math.pi/2)
            self.initiallyResized = True

    def processPaintEvent(self, event):
        dc = wx.PaintDC(self)
        self.SetCurrent(self.context)
        if not self.GLinitialized:
            self.initGL()
            self.GLinitialized = True
        self.repaint()

    def repaint(self):
        #Set up projection matrix
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        farDist = 3*self.bbox.getDiagLength()
        nearDist = farDist/50.0
        gluPerspective(180.0*self.camera.yfov/np.pi, float(self.size.x)/self.size.y, nearDist, farDist)
        
        #Set up modelview matrix
        self.camera.gotoCameraFrame()    
        glClearColor(0.0, 0.0, 0.0, 0.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    
        if self.sound.Y.size > 0:
            glDisable(GL_LIGHTING)
            glColor3f(1, 0, 0)
            glPointSize(3)
            NPoints = self.sound.Y.shape[0]
            CurrPoint = NPoints-1
            if self.Playing:
                dT = self.timeOffset + float(pygame.mixer.music.get_pos()) / 1000.0
                sliderPos = int(np.round(1000*dT/(self.sound.getSampleDelay(-1))))
                self.timeSlider.SetValue(sliderPos)
                while dT > self.sound.getSampleDelay(self.PlayIDX):
                    self.PlayIDX += 1
                    if self.PlayIDX == NPoints - 1:
                        self.Playing = False
                CurrPoint = self.PlayIDX+1
                self.Refresh()
            if CurrPoint >= NPoints:
                CurrPoint = NPoints-1
            
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
            
            glPointSize(8)
            glBegin(GL_POINTS)
            glColor3f(1, 1, 1)
            x = self.sound.Y[CurrPoint, :]
            glVertex3f(x[0], x[1], x[2])
            glEnd()
            
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
            #Translate/rotate shape
            if evt.MiddleIsDown():
                self.camera.translate(dX, dY)
            elif evt.RightIsDown():
                self.camera.zoom(-dY)#Want to zoom in as the mouse goes up
            elif evt.LeftIsDown():
                self.camera.orbitLeftRight(dX)
                self.camera.orbitUpDown(dY)
        self.Refresh()

class MeshViewerFrame(wx.Frame):
    (ID_LoadSound, ID_SAVESCREENSHOT) = (1, 2)
    
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
        menuLoadSound = filemenu.Append(MeshViewerFrame.ID_LoadSound, "&Load Song", "Load Song")
        self.Bind(wx.EVT_MENU, self.glcanvas.OnLoadSound, menuLoadSound)
        menuSaveScreenshot = filemenu.Append(MeshViewerFrame.ID_SAVESCREENSHOT, "&Save Screenshot", "Save a screenshot of the GL Canvas")
        self.Bind(wx.EVT_MENU, self.OnSaveScreenshot, menuSaveScreenshot)
        
        menuExit = filemenu.Append(wx.ID_EXIT,"E&xit"," Terminate the program")
        self.Bind(wx.EVT_MENU, self.OnExit, menuExit)
        
        
        # Creating the menubar.
        menuBar = wx.MenuBar()
        menuBar.Append(filemenu,"&File") # Adding the "filemenu" to the MenuBar
        self.SetMenuBar(menuBar)  # Adding the MenuBar to the Frame content.
        
        self.rightPanel = wx.BoxSizer(wx.VERTICAL)
        
        #Buttons to go to a default view
        viewPanel = wx.BoxSizer(wx.HORIZONTAL)
        topViewButton = wx.Button(self, -1, "Top")
        self.Bind(wx.EVT_BUTTON, self.glcanvas.viewFromTop, topViewButton)
        viewPanel.Add(topViewButton, 0, wx.EXPAND)
        sideViewButton = wx.Button(self, -1, "Side")
        self.Bind(wx.EVT_BUTTON, self.glcanvas.viewFromSide, sideViewButton)
        viewPanel.Add(sideViewButton, 0, wx.EXPAND)
        frontViewButton = wx.Button(self, -1, "Front")
        self.Bind(wx.EVT_BUTTON, self.glcanvas.viewFromFront, frontViewButton)
        viewPanel.Add(frontViewButton, 0, wx.EXPAND)
        self.rightPanel.Add(wx.StaticText(self, label="Views"), 0, wx.EXPAND)
        self.rightPanel.Add(viewPanel, 0, wx.EXPAND)
        
        
        #Parameter Choices
        self.rightPanel.Add(wx.StaticText(self, label="\nParameters"), 0, wx.EXPAND)
        ParamPanel = wx.GridSizer(rows=4, cols=2, hgap=5, vgap=5)
        ParamPanel.Add(wx.StaticText(self, label="hopSize"), 0, wx.ALIGN_LEFT)
        self.glcanvas.hopSizeTxt = wx.TextCtrl(self)
        self.glcanvas.hopSizeTxt.SetValue("512")
        ParamPanel.Add(self.glcanvas.hopSizeTxt, 0, wx.ALIGN_LEFT)
        ParamPanel.Add(wx.StaticText(self, label="winSize"), 0, wx.ALIGN_LEFT)
        self.glcanvas.winSizeTxt = wx.TextCtrl(self)
        self.glcanvas.winSizeTxt.SetValue("16384")
        ParamPanel.Add(self.glcanvas.winSizeTxt, 0, wx.ALIGN_LEFT)        
        ParamPanel.Add(wx.StaticText(self, label="fmax"), 0, wx.ALIGN_LEFT)
        self.glcanvas.fmaxTxt = wx.TextCtrl(self)
        self.glcanvas.fmaxTxt.SetValue("8000")
        ParamPanel.Add(self.glcanvas.fmaxTxt, 0, wx.ALIGN_LEFT)
        self.rightPanel.Add(ParamPanel, 0, wx.EXPAND)
        
        updateParamsButton = wx.Button(self, -1, "Update Parameters")
        self.Bind(wx.EVT_BUTTON, self.glcanvas.updateParams, updateParamsButton)
        self.rightPanel.Add(updateParamsButton, 0, wx.EXPAND)
        
        #PlayPause
        self.rightPanel.Add(wx.StaticText(self, label="\nPlay/Pause"), 0, wx.EXPAND)
        PlayPausePanel = wx.GridSizer(rows=2, cols=2, hgap=2, vgap=2)
        playButton = wx.Button(self, -1, "Play")
        self.Bind(wx.EVT_BUTTON, self.glcanvas.OnPlay, playButton)
        PlayPausePanel.Add(playButton, 0, wx.EXPAND)
        pauseButton = wx.Button(self, -1, "Pause")
        self.Bind(wx.EVT_BUTTON, self.glcanvas.OnPause, pauseButton)
        PlayPausePanel.Add(pauseButton, 0, wx.EXPAND)
        PlayPausePanel.Add(wx.StaticText(self, label='Time'))
        self.glcanvas.TimeTxt = wx.TextCtrl(self)
        self.glcanvas.TimeTxt.SetValue("0")
        PlayPausePanel.Add(self.glcanvas.TimeTxt, flag=wx.LEFT, border=5)    
        self.rightPanel.Add(PlayPausePanel, 0, wx.EXPAND)
        
        
        #Add the scroll bar to choose the time of the song
        glCanvasSizer = wx.BoxSizer(wx.VERTICAL)
        glCanvasSizer.Add(self.glcanvas, 2, wx.EXPAND)
        self.glcanvas.timeSlider = wx.Slider(self, -1, 0, 0, 1000)
        glCanvasSizer.Add(self.glcanvas.timeSlider, 0, wx.EXPAND)
        self.glcanvas.timeSlider.Bind(wx.EVT_COMMAND_SCROLL, self.glcanvas.SliderMove)    
        
        #Finally add the two main panels to the sizer
        self.sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.sizer.Add(glCanvasSizer, 2, wx.EXPAND)
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
