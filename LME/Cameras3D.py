from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import numpy as np
import math

EPS = 1e-12

def normalizeVec(V):
    return V/np.sqrt(np.sum(V**2))

class Plane3D(object):
    #P0 is some point on the plane, N is the normal
    def __init__(self, P0, N):
        self.P0 = np.array(P0)
        self.N = normalizeVec(N)
        self.resetEquation()

    def resetEquation(self):
        self.D = -self.P0.dot(self.N)

    def initFromEquation(self, A, B, C, D):
        N = np.array([A, B, C])
        self.P0 = (-D/N.dot(N))*np.array([A, B, C])
        self.N = normalizeVec(N)
        self.resetEquation()

    def distFromPlane(self, P):
        return self.N.dot(P) + self.D

    def __str__(self):
        return "Plane3D: %g*x + %g*y + %g*z + %g = 0"%(self.N[0], self.N[1], self.N[2], self.D)


class Line3D(object):
    def __init__(self, P0, V):
        self.P0 = np.array(P0)
        self.V = np.array(V)

    def intersectPlane(self, plane):
        P0 = plane.P0
        N = plane.N
        P = self.P0
        V = self.V
        if abs(N.dot(V)) < EPS:
            return None
        t = (P0.dot(N) - N.dot(P)) / (N.dot(V))
        intersectP = P + t*V
        return [t, intersectP]
    
    def intersectOtherLineRet_t(self, other):
        #Solve for (s, t) in the equation P0 + t*V0 = P1+s*V1
        #This is three equations (x, y, z components) in 2 variables (s, t)
        #User cramer's rule and the fact that there is a linear
        #dependence that only leaves two independent equations
        #(add the last two equations together)
        #[a b][t] = [e]
        #[c d][s]    [f]
        P0 = self.P0
        V0 = self.V
        P1 = other.P0
        V1 = other.V
        a = V0[0]+V0[2]
        b = -(V1[0]+V1[2])
        c = V0[1] + V0[2]
        d = -(V1[1]+V1[2])
        e = P1[0] + P1[2] - (P0[0] + P0[2])
        f = P1[1] + P1[2] - (P0[1] + P0[2])
        #print "[%g %g][t] = [%g]\n[%g %g][s]   [%g]"%(a, b, e, c, d, f)
        detDenom = a*d - c*b
        #Lines are parallel or skew
        if abs(detDenom) < EPS:
            return None
        detNumt = e*d - b*f
        detNums = a*f - c*e
        t = float(detNumt) / float(detDenom)
        s = float(detNums) / float(detDenom)
        #print "s = %g, t = %g"%(s, t)
        return (t, P0 + t*V0)
    
    def intersectOtherLine(self, other):
        ret = self.intersectOtherLineRet_t(other)
        if ret:
            return ret[1]
        return None
    
    def __str__(self):
        return "Line3D: %s + t%s"%(self.P0, self.V)


class Ray3D(object):
    def __init__(self, P0, V):
        self.P0 = np.array(P0)
        self.V = normalizeVec(V)
        self.line = Line3D(self.P0, self.V)
    
    def Copy(self):
        return Ray3D(self.P0, self.V)
    
    def Transform(self, matrix):
        self.P0 = mulHomogenous(matrix, self.P0.flatten())
        self.V = matrix[0:3, 0:3].dot(self.V)
        self.V = normalizeVec(self.V)
    
    def intersectPlane(self, plane):
        intersection = self.line.intersectPlane(plane)
        if intersection:
            if intersection[0] < 0:
                return None
            return intersection
    
    def intersectMeshFace(self, face):
        facePlane = face.getPlane()
        intersection = self.intersectPlane(facePlane)
        if not intersection:
            return None
        [t, intersectP] = intersection
        #Now check to see if the intersection is within the polygon
        #Do this by verifying that intersectP is on the same side
        #of each segment of the polygon
        verts = face.getVerticesPos()
        if verts.shape[0] < 3:
            return None
        lastCross = np.cross(verts[1, :]-verts[0, :], intersectP - verts[1, :])
        lastCross = normalizeVec(lastCross)
        for i in range(1, verts.shape[0]):
            v0 = verts[i, :]
            v1 = verts[(i+1)%verts.shape[0]]
            cross = np.cross(v1 - v0, intersectP - v1)
            cross = normalizeVec(cross)
            if cross.dot(lastCross) < EPS: #The intersection point is on the outside of the polygon
                return None
            lastCross = cross
        return [t, intersectP]

    def __str__(self):
        return "Ray3D: %s + t%s"%(self.P0, self.V)

class BBox3D(object):
    def __init__(self, b = np.array([[-1, -1, -1], [1, 1, 1]], dtype = 'float32')):
        self.b = b
    
    def getDiagLength(self):
        dB = self.b[1, :] - self.b[0, :]
        return np.sqrt(dB.dot(dB))
    
    def getCenter(self):
        return np.mean(self.b, 0)
    
    def addPoint(self, P):
        self.b[0, :] = np.min((P, self.b[0, :]), 0)
        self.b[1, :] = np.max((P, self.b[1, :]), 0)
    
    def fromPoints(self, Ps):
        self.b[0, :] = np.min(Ps, 0)
        self.b[1, :] = np.max(Ps, 0)
    
    def Union(self, other):
        self.b[0, :] = np.min(self.b[0, :], other.b[0, :])
        self.b[1, :] = np.max(self.b[1, :], other.b[1, :])
    
    def __str__(self):
        coords = self.b.T.flatten()
        ranges = (self.b[1, :] - self.b[0, :]).flatten()
        return "BBox3D: [%g, %g] x [%g, %g] x [%g, %g],  Range (%g x %g x %g)"%tuple(coords.tolist() + ranges.tolist())

#This function pushes a matrix onto the stack that puts everything
#in the frame of a camera which is centered at position "P",
#is pointing towards "t", and has vector "r" to the right
#t - towards vector
#u - up vector
#r - right vector
#P - Camera center
def gotoCameraFrame(t, u, r, P):
	rotMat = np.array([ [r[0], u[0], -t[0], 0], [r[1], u[1], -t[1], 0], [r[2], u[2], -t[2], 0], [0, 0, 0, 1] ])
	rotMat = rotMat.T
	transMat = np.array([ [1, 0, 0, -P[0]], [0, 1, 0, -P[1]], [0, 0, 1, -P[2]], [0, 0, 0, 1] ])
	#Translate first then rotate
	mat = rotMat.dot(transMat)
	#OpenGL is column major and mine are row major so take transpose
	mat = mat.T
	glMatrixMode(GL_MODELVIEW)
	glLoadIdentity()
	glMultMatrixd(mat.flatten())
	
def getModelviewMatrix(t, u, r, P):
	rotMat = np.array([ [r[0], u[0], -t[0], 0], [r[1], u[1], -t[1], 0], [r[2], u[2], -t[2], 0], [0, 0, 0, 1] ])
	rotMat = rotMat.T
	transMat = np.array([ [1, 0, 0, -P[0]], [0, 1, 0, -P[1]], [0, 0, 1, -P[2]], [0, 0, 0, 1] ])
	#Translate first then rotate
	mat = rotMat.dot(transMat)
	#OpenGL is column major and mine are row major so take transpose
	mat = mat.T
	return mat.flatten()    

def getPerspectiveMatrix(yfov, aspect, near, far):
    f = 1.0/math.tan(yfov/2)
    nf = 1/(near - far)
    mat = np.zeros(16)
    mat[0] = f/aspect
    mat[5] = f
    mat[10] = (far + near)*nf
    mat[11] = -1
    mat[14] = (2*far*near)*nf
    return mat

class MousePolarCamera(object):
	#Coordinate system is defined as in OpenGL as a right
	#handed system with +z out of the screen, +x to the right,
	#and +y up
	#phi is CCW down from +y, theta is CCW away from +z
	def __init__(self, pixWidth, pixHeight, yfov = 0.75):
		self.pixWidth = pixWidth
		self.pixHeight = pixHeight
		self.yfov = yfov
		self.nearDist = 0.1
		self.farDist = 10.0
		self.center = np.array([0, 0, 0])
		self.R = 1
		self.theta = 0
		self.phi = 0 
		self.updateVecsFromPolar()
	
	def setNearFar(self, nearDist, farDist):
	    self.nearDist = nearDist
	    self.farDist = farDist
	
	def centerOnBBox(self, bbox, theta = -math.pi/2, phi = math.pi/2):
		self.center = bbox.getCenter()
		self.R = bbox.getDiagLength()*1.5
		self.theta = theta
		self.phi = phi
		self.updateVecsFromPolar()		

	def centerOnPoints(self, X):
		bbox = BBox3D()
		bbox.fromPoints(X)
		self.centerOnBBox(bbox)

	def updateVecsFromPolar(self):
		[sinT, cosT, sinP, cosP] = [math.sin(self.theta), math.cos(self.theta), math.sin(self.phi), math.cos(self.phi)]
		#Make the camera look inwards
		#i.e. towards is -dP(R, phi, theta)/dR, where P(R, phi, theta) is polar position
		self.towards = np.array([-sinP*cosT, -cosP, sinP*sinT])
		self.up = np.array([-cosP*cosT, sinP, cosP*sinT])
		self.eye = self.center - self.R*self.towards

	def gotoCameraFrame(self):
		gotoCameraFrame(self.towards, self.up, np.cross(self.towards, self.up), self.eye)
	
	def getModelviewMatrix(self):
	    return getModelviewMatrix(self.towards, self.up, np.cross(self.towards, self.up), self.eye)
	
	def getPerspectiveMatrix(self):
	    return getPerspectiveMatrix(self.yfov, float(self.pixWidth)/self.pixHeight, nearDist, farDist)
	
	def orbitUpDown(self, dP):
		dP = 1.5*dP/float(self.pixHeight)
		self.phi = self.phi+dP
		self.updateVecsFromPolar()
	
	def orbitLeftRight(self, dT):
		dT = 1.5*dT/float(self.pixWidth)
		self.theta = self.theta-dT
		self.updateVecsFromPolar()
	
	def zoom(self, rate):
		rate = rate / float(self.pixHeight)
		self.R = self.R*pow(4, rate)
		self.updateVecsFromPolar()
	
	def translate(self, dx, dy):
		length = np.sqrt((self.center-self.eye)**2)*math.tan(self.yfov);
		dx = length*dx / float(self.pixWidth)
		dy = length*dy / float(self.pixHeight)
		r = np.cross(self.towards, self.up)
		self.center = self.center - dx*r - dy*self.up
		self.updateVecsFromPolar()
