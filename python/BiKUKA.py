#!/usr/bin/python
"""
This function is to simulate the manifold learning approach for bimanual manipulation
"""
from openravepy import *
import random
import math, time
import numpy as np
from openravepy import misc
if not __openravepy_build_doc__:
    from openravepy import *
    from numpy import *
import time, threading
from collections import Counter
from numpy.linalg import inv
from scipy.optimize import *
from math import radians
import rospy
from copy import copy, deepcopy
from GraspFun import *
import pdb

env = Environment() # create openrave environment
env.SetDebugLevel(DebugLevel.Fatal)
env.SetViewer('qtcoin') # attach viewer (optional)
robot = env.ReadRobotURI('../robots/lwr_dual.robot.xml') #real
env.Add(robot,True)

ind =0
indlist=[1,2,4,6,8,10,12,41,42,44,46,48,50,52]

print shape(robot.GetLinks())
for link in robot.GetLinks():

	if ind in indlist:
		for geom in link.GetGeometries():
			geom.SetDiffuseColor([1,0.4, 0, 0.1])
			geom.SetTransparency(0.5)
	else:
		if (ind<14 and ind >0) or (ind>41and ind<54):
			for geom in link.GetGeometries():
				geom.SetDiffuseColor([0.862745,0.862745,0.862745,0.2])
				geom.SetTransparency(0.2)
	ind = ind+1
print "Number of joints:------", repr(robot.GetActiveDOF())

#left
pdb.set_trace()
robot.SetDOFValues([radians(10)],[7])
robot.SetDOFValues([radians(40)],[8])
robot.SetDOFValues([radians(45)],[9])
robot.SetDOFValues([radians(45)],[10])
robot.SetDOFValues([radians(10)],[11])
robot.SetDOFValues([radians(40)],[12])
robot.SetDOFValues([radians(45)],[13])
robot.SetDOFValues([radians(45)],[14])
robot.SetDOFValues([radians(10)],[15])
robot.SetDOFValues([radians(40)],[16])
robot.SetDOFValues([radians(45)],[17])
robot.SetDOFValues([radians(45)],[18])
robot.SetDOFValues([radians(-100)],[19])
robot.SetDOFValues([radians(0)],[20])
robot.SetDOFValues([radians(-10)],[21])
robot.SetDOFValues([radians(50)],[22])
#right
robot.SetDOFValues([radians(10)],[30])
robot.SetDOFValues([radians(40)],[31])
robot.SetDOFValues([radians(45)],[32])
robot.SetDOFValues([radians(45)],[33])
robot.SetDOFValues([radians(10)],[34])
robot.SetDOFValues([radians(40)],[35])
robot.SetDOFValues([radians(45)],[36])
robot.SetDOFValues([radians(45)],[37])
robot.SetDOFValues([radians(10)],[38])
robot.SetDOFValues([radians(40)],[39])
robot.SetDOFValues([radians(45)],[40])
robot.SetDOFValues([radians(45)],[41])
robot.SetDOFValues([radians(-100)],[42])
robot.SetDOFValues([radians(0)],[43])
robot.SetDOFValues([radians(-10)],[44])
robot.SetDOFValues([radians(50)],[45])

env.UpdatePublishedBodies()

#
framescale=0.2
handles=[]
T=array([[1, 0,0 , 0],[0, 1 ,0 , 0],[0, 0, 1 , 0],[0.015, 0.015, 0 , 1]])
h1=env.drawlinestrip(points=array((T[0:3,3],T[0:3,3]+T[0:3,0]*framescale)),linewidth=5.0, colors=array(((1,0,0,0.5))))
h2=env.drawlinestrip(points=array((T[0:3,3],T[0:3,3]+T[0:3,1]*framescale)),linewidth=5.0, colors=array(((0,1,0,0.5))))
h3=env.drawlinestrip(points=array((T[0:3,3],T[0:3,3]+T[0:3,2]*framescale)),linewidth=5.0, colors=array(((0,0,1,0.5))))
handles.append(h1)
handles.append(h2)
handles.append(h3)
env.UpdatePublishedBodies()

# create a new object (cylinder) to grasp
with env:
	infocylinder = KinBody.Link.GeometryInfo()
	infocylinder._type = KinBody.Link.GeomType.Cylinder
	infocylinder._t[0,3] = 0.0
	infocylinder._vGeomData = [0.32,0.05]
	infocylinder._bVisible = True
	infocylinder._fTransparency = 0.6
	infocylinder._vDiffuseColor = [0.411765, 0.411765, 0.411765]
	k3 = RaveCreateKinBody(env,'')
	k3.InitFromGeometries([infocylinder])
	k3.SetName('tmpcylinder')
	#ObjInBase = array([[1, 0,0 , -0.5],[0, 1 ,0 , 0],[0, 0, 1 , 0.5],[0, 0, 0 , 1]])
	ObjInBase = array([[0,0,1,0],[0, 1 ,0 , 0],[-1,0,0,0],[-0.55, -0.15, 0.6 , 1]]).T
	k3.SetTransform(ObjInBase)
	env.Add(k3,True)
	ObjFrame=k3.GetTransform()
	framescale=0.15
	h1,h2,h3= PlotFrame(env, ObjFrame, framescale)
	env.UpdatePublishedBodies()
env.UpdatePublishedBodies()


#set grasps
GRightInObj =  array([[0,-1,0,0],[1,0,0,0],[0,0,1,0],[0.0,0.35,-0.1,1]]).T
GRightInBase = dot(ObjFrame,GRightInObj)
"""
GRightInBase = array([[ 0.        , -0.        , -1.        , -0.55000025],
       [ 0.        ,  1.        , -0.        ,  0.30547652],
       [ 1.        ,  0.        ,  0.        ,  0.47831321],
       [ 0.        ,  0.        ,  0.        ,  1.        ]])
"""
h4,h5,h6= PlotFrame(env, GRightInBase, 0.1)

GLeftInObj =  array([[0,-1,0,0],[1,0,0,0],[0,0,1,0],[0.0,-0.35,-0.1,1]]).T #here
GLeftInBase = dot(ObjFrame,GLeftInObj)


h7,h8,h9= PlotFrame(env, GLeftInBase, 0.1)
env.UpdatePublishedBodies()
raw_input('press enter to use the IK to reach grasping points')


robot = env.GetRobots()[0] # get the first robot
robotLeftBase = robot.GetLinks()[1].GetTransform()
"""
h1,h2,h3= PlotFrame(env, robotLeftBase, 0.2)
handles.append(h1)
handles.append(h2)
handles.append(h3)
"""
robotRightBase = robot.GetLinks()[37].GetTransform()
"""
h1,h2,h3= PlotFrame(env, robotRightBase, 0.2)
handles.append(h1)
handles.append(h2)
handles.append(h3)
"""
env.UpdatePublishedBodies() 

pdb.set_trace()

maniLeft = robot.SetActiveManipulator("l_lwr")
print maniLeft.GetArmIndices() 
ikmodelLeft = databases.inversekinematics.InverseKinematicsModel(robot=robot, freeindices=[4], iktype=IkParameterization.Type.Transform6D)
if not ikmodelLeft.load():
    ikmodelLeft.autogenerate()

maniRight = robot.SetActiveManipulator("r_lwr")
print maniRight.GetArmIndices()
ikmodelRight = databases.inversekinematics.InverseKinematicsModel(robot=robot,freeindices=[27], iktype=IkParameterization.Type.Transform6D)
if not ikmodelRight.load():
	ikmodelRight.autogenerate()

solLeft = maniLeft.FindIKSolution(GLeftInBase,IkFilterOptions.IgnoreJointLimits)
solRight = maniRight.FindIKSolution(GRightInBase,IkFilterOptions.IgnoreJointLimits)
CurrJntLeft = robot.GetActiveDOFValues()[0:7]
CurrJntRight = robot.GetActiveDOFValues()[23:30]
tmpLeft = solLeft-CurrJntLeft
tmpRight= solRight-CurrJntRight
while LA.norm(tmpLeft)>0.25 or LA.norm(tmpRight)>0.25:
	#print LA.norm(tmpLeft), LA.norm(tmpRight)
	cmdLeft =CurrJntLeft+tmpLeft*0.1
	cmdRight =CurrJntRight+tmpRight*0.1	
	robot.SetDOFValues(cmdLeft, maniLeft.GetArmIndices())
	robot.SetDOFValues(cmdRight,maniRight.GetArmIndices())
	env.UpdatePublishedBodies()
	time.sleep(0.1)
	CurrJntLeft = robot.GetActiveDOFValues()[0:7]
	CurrJntRight = robot.GetActiveDOFValues()[23:30]
	tmpLeft = solLeft-CurrJntLeft
	tmpRight= solRight-CurrJntRight
env.UpdatePublishedBodies() 


# raw_input('press enter to manipulte the object')
ttrange = array([-1.57,1.57])
ppcurve1 = GenerateCurve(ttrange,nb=40,noise=0.01)
ppcurve2 = GenerateCurve(ttrange,nb=40,noise=0.01)
ppcurve3 = GenerateCurve(ttrange,nb=40,noise=0.01)
ind=0
while ind in range(shape(ppcurve1)[0]):
	#ppcurve[ind,0]=ppcurve[ind,0]-0.05
	handles.append(env.plot3(points=ppcurve1[ind,:],pointsize=0.01,colors=array(((0.517647,0.439216,1))),drawstyle = 1))
	handles.append(env.plot3(points=ppcurve2[ind,:],pointsize=0.01,colors=array(((0.517647,0.439216,1))),drawstyle = 1))
	handles.append(env.plot3(points=ppcurve3[ind,:],pointsize=0.01,colors=array(((0.517647,0.439216,1))),drawstyle = 1))
	ind=ind+1
	env.UpdatePublishedBodies()
	time.sleep(0.05)
numpy.savetxt('BiKUKACurve.txt',np.vstack([ppcurve1,ppcurve2,ppcurve3]))

raw_input('press enter to manipulte the object')

ppcurve = GenerateCurve(ttrange,nb=100)

"""
# This is the part without any disturbance
with env:
	ind=0
	while ind in range(shape(ppcurve)[0]): 
		ObjInBase[0:3,3] = ppcurve[ind,:]
		k3.SetTransform(ObjInBase)
		ObjFrame=k3.GetTransform()
		GLeftInBase = dot(ObjFrame,GLeftInObj)
		GRightInBase = dot(ObjFrame,GRightInObj)
		solLeft = maniLeft.FindIKSolution(GLeftInBase,IkFilterOptions.IgnoreJointLimits)
		solRight = maniRight.FindIKSolution(GRightInBase,IkFilterOptions.IgnoreJointLimits)	
			
		# CurrJntLeft = robot.GetActiveDOFValues()[0:7]
		# CurrJntRight = robot.GetActiveDOFValues()[23:30]
		# tmpLeft = solLeft-CurrJntLeft
		# tmpRight= solRight-CurrJntRight
		# print LA.norm(tmpRight)
		# if LA.norm(tmpLeft)>1:
		# 	cmdLeft = CurrJntLeft
		# else:
		# 	cmdLeft = solLeft
		# if LA.norm(tmpRight)>1:
		# 	cmdRight = CurrJntRight
		# else:
		# 	cmdRight = solRight

		robot.SetDOFValues(solLeft, maniLeft.GetArmIndices())
		robot.SetDOFValues(solRight,maniRight.GetArmIndices())
		h1,h2,h3= PlotFrame(env, ObjFrame, 0.1)
		h4,h5,h6= PlotFrame(env, GRightInBase, 0.1)
		h7,h8,h9= PlotFrame(env, GLeftInBase, 0.1)
		# print solLeft
		# print solRight 
		ind=ind+1
		env.UpdatePublishedBodies()
		time.sleep(0.01)
raw_input('press enter to manipulte the object')
"""
# This is the part with disturbance but without any learned manifold adaptation
indpert =randint(20,80)
if indpert in arange(35,45):
	indpert = 50 
print indpert
indpert = 60
v = raw_input('with projection or not(y/n):\n')
print v
bPert = True
with env:
	ind=0
	while ind in range(shape(ppcurve)[0]):
		pdb.set_trace();
		if ind == indpert and bPert:
			timepert=0
			bPert =False
			#dirpert = random.random((3))
			dirpert = GetTangentVecCurve(ppcurve[ind,:])
			dirpert=np.squeeze(dirpert)
			print dirpert
			dirpert[1]=dirpert[1]+0.1
			dirpert[2]=dirpert[2]+0.3
			dirpert = dirpert/LA.norm(dirpert)
			ObjInBase[0:3,3] = ppcurve[ind,:]
			while timepert<15:
				ObjInBase[0:3,3] = ObjInBase[0:3,3]+0.01*dirpert
				handles.append(env.plot3(points=ObjInBase[0:3,3],pointsize=0.01,colors=array(((1,0,0))),drawstyle = 1))
				k3.SetTransform(ObjInBase)
				ObjFrame=k3.GetTransform()
				GLeftInBase = dot(ObjFrame,GLeftInObj)
				GRightInBase = dot(ObjFrame,GRightInObj)
				solLeft = maniLeft.FindIKSolution(GLeftInBase,IkFilterOptions.IgnoreJointLimits)
				solRight = maniRight.FindIKSolution(GRightInBase,IkFilterOptions.IgnoreJointLimits)	
				robot.SetDOFValues(solLeft, maniLeft.GetArmIndices())
				robot.SetDOFValues(solRight,maniRight.GetArmIndices())
				h1,h2,h3= PlotFrame(env, ObjFrame, 0.1)
				h4,h5,h6= PlotFrame(env, GRightInBase, 0.1)
				h7,h8,h9= PlotFrame(env, GLeftInBase, 0.1)
				timepert=timepert+1
				#print timepert
				env.UpdatePublishedBodies()
				time.sleep(0.01)
			#after perturbation, move in strait line to target(*depends on planner)
			if v == 'n':
				ind=ind+30	
				while LA.norm(ObjInBase[0:3,3]-ppcurve[ind,:])>0.01:
					ObjInBase[0:3,3] = ObjInBase[0:3,3]+(ppcurve[ind,:]-ObjInBase[0:3,3])*0.1
					handles.append(env.plot3(points=ObjInBase[0:3,3],pointsize=0.01,colors=array(((1,0,0))),drawstyle = 1))
					k3.SetTransform(ObjInBase)
					ObjFrame=k3.GetTransform()
					GLeftInBase = dot(ObjFrame,GLeftInObj)
					GRightInBase = dot(ObjFrame,GRightInObj)
					solLeft = maniLeft.FindIKSolution(GLeftInBase,IkFilterOptions.IgnoreJointLimits)
					solRight = maniRight.FindIKSolution(GRightInBase,IkFilterOptions.IgnoreJointLimits)	
					robot.SetDOFValues(solLeft, maniLeft.GetArmIndices())
					robot.SetDOFValues(solRight,maniRight.GetArmIndices())
					h1,h2,h3= PlotFrame(env, ObjFrame, 0.1)
					h4,h5,h6= PlotFrame(env, GRightInBase, 0.1)
					h7,h8,h9= PlotFrame(env, GLeftInBase, 0.1)
					env.UpdatePublishedBodies()
					time.sleep(0.02)
			else:
				# start projection
				pt = GetProjCurve(ObjInBase[0:3,3],eps=0.001)
				handles.append(env.plot3(points=pt,pointsize=0.02,colors=array(((1,0.270588,0))),drawstyle = 1))
				while LA.norm(ObjInBase[0:3,3]-pt)>0.01:
					ObjInBase[0:3,3] = ObjInBase[0:3,3]+(pt-ObjInBase[0:3,3])*0.1
					handles.append(env.plot3(points=ObjInBase[0:3,3],pointsize=0.01,colors=array(((0.6,0.6,0.6))),drawstyle = 1))
					k3.SetTransform(ObjInBase)
					ObjFrame=k3.GetTransform()
					GLeftInBase = dot(ObjFrame,GLeftInObj)
					GRightInBase = dot(ObjFrame,GRightInObj)
					solLeft = maniLeft.FindIKSolution(GLeftInBase,IkFilterOptions.IgnoreJointLimits)
					solRight = maniRight.FindIKSolution(GRightInBase,IkFilterOptions.IgnoreJointLimits)	
					robot.SetDOFValues(solLeft, maniLeft.GetArmIndices())
					robot.SetDOFValues(solRight,maniRight.GetArmIndices())
					h1,h2,h3= PlotFrame(env, ObjFrame, 0.1)
					h4,h5,h6= PlotFrame(env, GRightInBase, 0.1)
					h7,h8,h9= PlotFrame(env, GLeftInBase, 0.1)
					env.UpdatePublishedBodies()
					time.sleep(0.02)
				# Here it should use replanning on the manifold;
				# Here we just reuse the previous planning since we know this is only 1D; 
				dist2=np.sum((ppcurve-pt)**2,axis=1)
				ind = np.argmin(dist2)
				#raw_input('press enter to manipulte the object')
		else:
			ObjInBase[0:3,3] = ppcurve[ind,:]
			handles.append(env.plot3(points=ObjInBase[0:3,3],pointsize=0.015,colors=array(((1,0,0))),drawstyle = 1))
			k3.SetTransform(ObjInBase)
			ObjFrame=k3.GetTransform()
			GLeftInBase = dot(ObjFrame,GLeftInObj)
			GRightInBase = dot(ObjFrame,GRightInObj)
			solLeft = maniLeft.FindIKSolution(GLeftInBase,IkFilterOptions.IgnoreJointLimits)
			solRight = maniRight.FindIKSolution(GRightInBase,IkFilterOptions.IgnoreJointLimits)	

			robot.SetDOFValues(solLeft, maniLeft.GetArmIndices())
			robot.SetDOFValues(solRight,maniRight.GetArmIndices())
			h1,h2,h3= PlotFrame(env, ObjFrame, 0.1)
			h4,h5,h6= PlotFrame(env, GRightInBase, 0.1)
			h7,h8,h9= PlotFrame(env, GLeftInBase, 0.1)
		# print solLeft
		# print solRight 
		ind=ind+1
		env.UpdatePublishedBodies()
		time.sleep(0.02)
raw_input('press enter to manipulte the object')