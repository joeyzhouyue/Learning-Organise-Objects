from openravepy import *
from random import randint
import math, time
from numpy import *
from openravepy import misc
if not __openravepy_build_doc__:
    from openravepy import *
    from numpy import *
import time, threading
import numpy as np
import numpy.matlib
from numpy import linalg as LA
from rank_nullspace import rank, nullspace
from scipy.spatial import ConvexHull
from scipy.spatial import cKDTree

def CompConvexHull(pp,np, nb,mu):
	#compute the wrench hull
	W1= ContactPrimitive(pp[0,:], np[0,:], mu, nb)
	W2= ContactPrimitive(pp[1,:], np[1,:], mu, nb)
	W3= ContactPrimitive(pp[2,:], np[2,:], mu, nb)
	MW= MinkSum(W1, W2)
	MW=MinkSum(MW, W3)
	#MW = concatenate((MW,W1,W2,W3), axis=0)   #equal to add [0,0,0,0,0,0] into the set #?
	#print MW
	hull = ConvexHull(MW)
	#K = hull.simplices
	#hp,K = CompHyperPlane(MW, K); # This is not necessay, we can treat the degenerated case in the maximal force computation.
	return hull.equations, hull.simplices

def unique_rows(a):
	a = np.ascontiguousarray(a)
	unique_a = np.unique(a.view([('', a.dtype)]*a.shape[1]))
	return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))

def MinkSum(A,B):
	#Given two array A and B, return there every pariwise sum
	C=zeros((shape(A)[0]*shape(B)[0],shape(A)[1]))
	k=0
	for i in range(shape(A)[0]):
		for j in range(shape(B)[0]):
			C[k,:] = A[i,:]+B[j,:]
			k=k+1
	return unique_rows(C)


def CompHyperPlane(V,K):
	# compute the outwards normal direction
	#V: vertex; K: vertex index for each facet
	ns = zeros((shape(K)[0],shape(V)[1]+1)) # normal of each hyperplane and offset
	NP=[] #invalid hyperplane
	nbCol= shape(V)[1]
	for i in range(shape(K)[0]):
		tmp = zeros((shape(V[K[i,:],:])[0],nbCol+1))
		tmp[:,0:nbCol] = V[K[i,:],:]
		tmp[:,nbCol]= ones((shape(V[K[i,:],:])[0]))
		tmp1= nullspace(tmp)
		tmp1 = transpose(tmp1)
		if shape(tmp1)[0]== 0:
			print "Oops! Not Valid......"
		if shape(tmp1)[0]>1:
			ns[i,:] = tmp1[0,:]
			NP.append(i)
		else:
			ns[i,:] = tmp1[0,:]

		#check the normal and it is outwards direction:
		"""
		If V is a normal, b is an offset, and x is a point inside the convex hull, then Vx+b <0.
		In the newer version of scipy, this hyperplane is given as an output.
		"""
		if dot(ns[i,0:6],V[K[i,0],:])<0:
			ns[i,:] = -ns[i,:]
	ns = numpy.delete(ns,NP,0)
	K = numpy.delete(K, NP,0)
	return ns, K

def ContactPrimitive(p,np, mu, nb):
	W = zeros((6,nb))
	ns = nullspace(np)
	X=transpose(ns[:,0])
	Z=np
	Y =cross(Z, X)
	R=ones((3,3))
	R[0:3,0] = X
	R[0:3,1] = Y
	R[0:3,2] = Z
	for i in range(nb):
		W[0,i]= mu*cos(float(i)/nb*2.0*pi)
		W[1,i]= mu*sin(float(i)/nb*2*pi)
		W[2,i]= 1
		W[0:3,i] = dot(R,W[0:3,i])
		W[3:6,i] = cross(p,W[0:3,i])
	return transpose(W)
def VirtualFrame(p1, p2, p3):
	T = eye(4,4)
	T[0:3,0] = (p3-p1)/LA.norm(p3-p1)
	T[0:3,2] = numpy.cross(p2-p1, T[0:3,0])
	T[0:3,2] = T[0:3,2]/LA.norm(T[0:3,2])
	T[0:3,1] = numpy.cross(T[0:3,2],T[0:3,0]) 
	T[0:3,3] = (p1+p2+p3)/3
	return T

# Manifold Learning
def GenerateSurface(h, phi, xxrange,yyrange,nb=5,noise=0):
	"This is a function to generate a surface in given range with noise or not"
	x=numpy.linspace(xxrange[0],xxrange[1],nb)
	y=numpy.linspace(yyrange[0],yyrange[1],nb)
	xx, yy = meshgrid(x, y)
	v = np.cos(xx*phi-1)+np.sin(yy*phi)
	zz = -h*sin(v)/v+1.3+2*noise*random.random(shape(xx))-noise
	return np.vstack([np.hstack(xx), np.hstack(yy),np.hstack(zz)]).T, xx,yy,zz
def ReadTxt(filename):
	fh = open( filename)
	data = []
	for line in fh.readlines():
		y = [value for value in line.split()]
		data.append( y )
		fh.close()
	data = np.asarray(data)
	data = data.astype(np.float)
	return data

def GetTangentVec(x):
	theta1= ReadTxt('Mfd_theta1.txt')
	theta2= ReadTxt('Mfd_theta2.txt')
	theta3= ReadTxt('Mfd_theta3.txt')
	sigma= ReadTxt('Mfd_sigma.txt')
	mu= ReadTxt('Mfd_mu.txt')
	f=zeros((5,1))
	for i in range(5):
		f[i,0]=np.exp(-(LA.norm(x-array([[mu[0,i],mu[1,i],mu[2,i]]])))**2/(2*sigma[0,i]))
	H1=dot(theta1,f)
	H2=dot(theta2,f)
	H3=dot(theta3,f)
	H = np.hstack([H1,H2,H3])
	ind=0
	for ind in range(shape(H)[0]):
		H[ind,:]=H[ind,:]/LA.norm(H[ind,:])
		ind=ind+1
	return H.T
def GetProjection(x,eps=0.001):
	#compute the projection of x on the manifold for suface
	dataTraining= ReadTxt('Mfd_training.txt')
	dist2=np.sum((dataTraining-x)**2,axis=1)
	id = np.argmin(dist2)
	pt = dataTraining[id,:]
	tmp0 = 1
	tmp1 = 0
	while abs(tmp1-tmp0)>eps:
		tmp0 = LA.norm(x-pt)
		H = GetTangentVec(pt)
		ptmp = dot(x-pt,H)
		pt=pt+0.02*dot(H,ptmp)
		tmp1 = LA.norm(x-pt)
	return pt

def PlotFrame(env, T, framescale):
	h1=env.drawlinestrip(points=array((T[0:3,3],T[0:3,3]+T[0:3,0]*framescale)),linewidth=5.0, colors=array(((1,0,0,0.5))))
	h2=env.drawlinestrip(points=array((T[0:3,3],T[0:3,3]+T[0:3,1]*framescale)),linewidth=5.0, colors=array(((0,1,0,0.5))))
	h3=env.drawlinestrip(points=array((T[0:3,3],T[0:3,3]+T[0:3,2]*framescale)),linewidth=5.0, colors=array(((0,0,1,0.5))))
	return h1,h2,h3
def GenerateCurve(ttrange,nb=100,noise=0.0):
	t  = np.linspace(ttrange[0],ttrange[1],nb)
	#yy = 0.3*sin(t*4)**3
	#xx = 0.3*cos(t*4)**2-0.95
	#zz = 0.3*sin(t*4)*cos(t*4)+0.6
	yy = np.linspace(-0.15,3.14/15-0.15,nb)
	zz = 0.2*sin((yy+0.15)*15)+0.6+random.random(shape(yy))*noise
	xx=-0.1*sin((yy+0.15)*50)-0.55+random.random(shape(yy))*noise
	return np.vstack([np.hstack(xx), np.hstack(yy),np.hstack(zz)]).T
def GetTangentVecCurve(x):
	theta1= ReadTxt('MfdCurve_theta1.txt')
	theta2= ReadTxt('MfdCurve_theta2.txt')
	theta3= ReadTxt('MfdCurve_theta3.txt')
	sigma= ReadTxt('MfdCurve_sigma.txt')
	mu= ReadTxt('MfdCurve_mu.txt')
	f=zeros((5,1))
	for i in range(5):
		f[i,0]=np.exp(-(LA.norm(x-array([[mu[0,i],mu[1,i],mu[2,i]]])))**2/(2*sigma[0,i]))
	H1=dot(theta1,f)
	H2=dot(theta2,f)
	H3=dot(theta3,f)
	H = np.hstack([H1,H2,H3])
	ind=0
	for ind in range(shape(H)[0]):
		H[ind,:]=H[ind,:]/LA.norm(H[ind,:])
		ind=ind+1
	return H.T	
def GetProjCurve(x,eps=0.001):
	#compute the projection of x on the manifold for curve
	# All these function should be optimized before publish
	dataTraining= ReadTxt('MfdCurve_training.txt')
	dist2=np.sum((dataTraining-x)**2,axis=1)
	id = np.argmin(dist2)
	pt = dataTraining[id,:]
	tmp0 = 1
	tmp1 = 0
	while abs(tmp1-tmp0)>eps:
		tmp0 = LA.norm(x-pt)
		H = GetTangentVecCurve(pt)
		ptmp = dot(x-pt,H)
		pt=pt+0.02*dot(H,ptmp)
		tmp1 = LA.norm(x-pt)
	return pt
def GenerateCurveHand(r=0.01,nb=100,noise=0.0):
	# generate close curve for the Allegro hand
	theta = np.linspace(0,3.14*2,nb)
	z = r*(1+0.5*sin(theta/2)**2)*cos(theta)
	y = r*(1+sin(theta))*sin(theta)
	x = ones(shape(z))+noise
	return np.vstack([np.hstack(x), np.hstack(y),np.hstack(z)]).T

########################################
####   Funtions for Allegro Hand.  #####
########################################
def GetTangentVecCurveHand(x):
	theta1= ReadTxt('MAllegroCurve_theta1.txt')
	theta2= ReadTxt('MAllegroCurve_theta2.txt')
	theta3= ReadTxt('MAllegroCurve_theta3.txt')
	sigma= ReadTxt('MAllegroCurve_sigma.txt')
	mu= ReadTxt('MAllegroCurve_mu.txt')
	f=zeros((8,1))  # The nb of rbf kernel
	for i in range(8):
		f[i,0]=np.exp(-(LA.norm(x-array([[mu[0,i],mu[1,i],mu[2,i]]])))**2/(2*sigma[0,i]))
	H1=dot(theta1,f)
	H2=dot(theta2,f)
	H3=dot(theta3,f)
	H = np.hstack([H1,H2,H3])
	ind=0
	for ind in range(shape(H)[0]):
		H[ind,:]=H[ind,:]/LA.norm(H[ind,:])
		ind=ind+1
	return H.T

def GetProjCurveHand(x,eps=0.001):
	#compute the projection of x on the manifold for curve of Allegro Hand
	dataTraining= ReadTxt('MAllegroCurve_training.txt')
	dist2=np.sum((dataTraining-x)**2,axis=1)
	id = np.argmin(dist2)
	pt = dataTraining[id,:]
	tmp0 = 1
	tmp1 = 0
	while abs(tmp1-tmp0)>eps:
		tmp0 = LA.norm(x-pt)
		H = GetTangentVecCurveHand(pt)
		ptmp = dot(x-pt,H)
		pt=pt+0.02*dot(H,ptmp)
		tmp1 = LA.norm(x-pt)
	return pt





