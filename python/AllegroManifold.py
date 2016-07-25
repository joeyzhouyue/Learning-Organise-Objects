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
env.SetViewer('qtcoin') # attach viewer (optional)
env.Load('../robots/allegroTac.robot.xml') # load 16 dof allegro hand
robot = env.GetRobots()[0] # get the first robot
handles = []
T=array([[1, 0,0 , 0],[0, 1 ,0 , 0],[0, 0, 1 , 0],[0.0, 0.0, 0 , 1]]).T
framescale=0.03
h1,h2,h3= PlotFrame(env, T, framescale)
env.UpdatePublishedBodies()
with env:
	infocylinder = KinBody.Link.GeometryInfo()
	infocylinder._type = KinBody.Link.GeomType.Cylinder
	infocylinder._t[0,3] = 0.0
	infocylinder._vGeomData = [0.03,0.016]
	infocylinder._bVisible = True
	infocylinder._fTransparency = 0.5
	infocylinder._vDiffuseColor = [0.419608,0.556863,0.137255]
	k3 = RaveCreateKinBody(env,'')
	k3.InitFromGeometries([infocylinder])
	k3.SetName('tmpcylinder')
	#ObjInBase = array([[1, 0,0 , -0.5],[0, 1 ,0 , 0],[0, 0, 1 , 0.5],[0, 0, 0 , 1]])
	ObjInBase = array([[0,0,-1,0],[0, 1 ,0 , 0],[1,0,0,0],[0.16, -0.052, 0.025, 1]]).T
	k3.SetTransform(ObjInBase)
	env.Add(k3,True)
	ObjFrame=k3.GetTransform()
	ObjTmp= np.eye(4)
	ObjTmp[2,3]=0.0055 # for plot frame
	ObjTmp = dot(ObjFrame,ObjTmp)
	framescale=0.02
	h4,h5,h6= PlotFrame(env, ObjTmp, framescale)
	env.UpdatePublishedBodies()
# set the finger joints
ObjMotion = matrixFromAxisAngle([0,1,0],radians(-20))
with env:
	ObjInBase=dot(ObjInBase,ObjMotion)
	k3.SetTransform(ObjInBase)
	env.Add(k3,True)
	ObjFrame=k3.GetTransform()
	ObjTmp= np.eye(4)
	ObjTmp[2,3]=0.00 # for plot frame
	ObjTmp = dot(ObjFrame,ObjTmp)
	framescale=0.01
	h4,h5,h6= PlotFrame(env, ObjTmp, framescale)
	robot.SetDOFValues([radians(-10)],[0])
	robot.SetDOFValues([radians(60)],[1])
	robot.SetDOFValues([radians(15)],[2])
	robot.SetDOFValues([radians(16)],[3])

	robot.SetDOFValues([radians(-7)],[4])
	robot.SetDOFValues([radians(60)],[5])
	robot.SetDOFValues([radians(15)],[6])
	robot.SetDOFValues([radians(25)],[7])

	robot.SetDOFValues([radians(-80)],[12])
	robot.SetDOFValues([radians(0)],[13])
	robot.SetDOFValues([radians(11)],[14])
	robot.SetDOFValues([radians(18)],[15])

	env.UpdatePublishedBodies()
	pCurve = GenerateCurveHand(r=0.01,nb=100)
	pCurve = pCurve+(ObjFrame[0:3,3]-pCurve[0,:])

	numpy.savetxt('AllegroCurve.txt',pCurve)

	handles.append(env.plot3(points = pCurve, pointsize=0.0006, colors=array(((0.541176,0.168627,0.886275))),drawstyle=1))
	env.UpdatePublishedBodies()
	T1 = robot.GetLinks()[6].GetTransform()
	T2 = robot.GetLinks()[12].GetTransform()
	T3 = robot.GetLinks()[24].GetTransform()

T1InObj= dot(inv(ObjFrame),T1)
T2InObj= dot(inv(ObjFrame),T2)
T3InObj= dot(inv(ObjFrame),T3)

hf1,hf2,hf3= PlotFrame(env, T1, 0.015)
hf4,hf5,hf6= PlotFrame(env, T2, 0.015)
hf7,hf8,hf9= PlotFrame(env, T3, 0.015)

mani_1 = robot.SetActiveManipulator("indexfin")
ikmod1 = databases.inversekinematics.InverseKinematicsModel(robot=robot,freeindices=[3], iktype=IkParameterization.Type.Translation3D)
if not ikmod1.load():
    ikmod1.autogenerate()

# pdb.set_trace();

mani_2 = robot.SetActiveManipulator("middlefin")
ikmod2 = databases.inversekinematics.InverseKinematicsModel(robot=robot,freeindices=[7], iktype=IkParameterization.Type.Translation3D)
if not ikmod2.load():
    ikmod2.autogenerate()

mani_3 = robot.SetActiveManipulator("thumbfin")
ikmod3 = databases.inversekinematics.InverseKinematicsModel(robot=robot,freeindices=[15], iktype=IkParameterization.Type.Translation3D)
if not ikmod3.load():
    ikmod3.autogenerate()
pdb.set_trace();

raw_input('press Enter to start the simulation')


"""
# The working version without any disturbance
ind=0
with env:

	for ind in range(shape(pCurve)[0]):
		ObjFrame = k3.GetTransform()
		ObjFrame[0:3,3] = pCurve[ind,:]
		T1 = dot(ObjFrame, T1InObj)
		T2 = dot(ObjFrame, T2InObj)
		T3 = dot(ObjFrame, T3InObj)
		hf1,hf2,hf3= PlotFrame(env, T1, 0.015)
		hf4,hf5,hf6= PlotFrame(env, T2, 0.015)
		hf7,hf8,hf9= PlotFrame(env, T3, 0.015)
		ikparam1 = IkParameterization(T1[0:3,3],ikmod1.iktype)
		sol1 = mani_1.FindIKSolution(ikparam1,IkFilterOptions.IgnoreJointLimits)
		
		ikparam2 = IkParameterization(T2[0:3,3],ikmod2.iktype)
		sol2 = mani_2.FindIKSolution(ikparam2,IkFilterOptions.IgnoreJointLimits)
		
		ikparam3 = IkParameterization(T3[0:3,3],ikmod3.iktype)
		sol3 = mani_3.FindIKSolution(ikparam3,IkFilterOptions.IgnoreJointLimits)
		print sol1, sol2,sol3
		k3.SetTransform(ObjFrame)
		robot.SetDOFValues(sol1, mani_1.GetArmIndices())
		robot.SetDOFValues(sol2, mani_2.GetArmIndices())
		robot.SetDOFValues(sol3, mani_3.GetArmIndices())
		handles.append(env.plot3(points=ObjFrame[0:3,3],pointsize=0.0016,colors=array(((1,0,0))),drawstyle = 1))
		env.UpdatePublishedBodies()
		ind= ind+1
		time.sleep(0.1)
raw_input('press Enter to continue')
"""
# add disturbance and start to adapt

ppcurve=deepcopy(pCurve)
indpert =randint(20,40)
print indpert
v = raw_input('with projection or not(y/n):\n')
print v
bPert = True
indpert=25;
ind=0

with env:
	while ind in range(shape(pCurve)[0]):

		if ind == indpert and bPert:
			timepert=0
			bPert = False
			dirpert = random.random((3))
			dirpert[0]=0.5;
			dirpert[2]=2;
			dirpert = dirpert/LA.norm(dirpert)
			ObjFrame[0:3,3] = ppcurve[ind,:]
			while timepert<15:
				ObjFrame[0:3,3] = ObjFrame[0:3,3]+0.001*dirpert
				handles.append(env.plot3(points=ObjFrame[0:3,3],pointsize=0.0008,colors=array(((1,0,0))),drawstyle = 1))
				k3.SetTransform(ObjFrame)
				ObjFrame=k3.GetTransform()
				T1 = dot(ObjFrame, T1InObj)
				T2 = dot(ObjFrame, T2InObj)
				T3 = dot(ObjFrame, T3InObj)
				hf1,hf2,hf3= PlotFrame(env, T1, 0.015)
				hf4,hf5,hf6= PlotFrame(env, T2, 0.015)
				hf7,hf8,hf9= PlotFrame(env, T3, 0.015)
				ikparam1 = IkParameterization(T1[0:3,3],ikmod1.iktype)
				sol1 = mani_1.FindIKSolution(ikparam1,IkFilterOptions.IgnoreJointLimits)
				
				ikparam2 = IkParameterization(T2[0:3,3],ikmod2.iktype)
				sol2 = mani_2.FindIKSolution(ikparam2,IkFilterOptions.IgnoreJointLimits)
				
				ikparam3 = IkParameterization(T3[0:3,3],ikmod3.iktype)
				sol3 = mani_3.FindIKSolution(ikparam3,IkFilterOptions.IgnoreJointLimits)
				#print sol1, sol2,sol3
				k3.SetTransform(ObjFrame)
				robot.SetDOFValues(sol1, mani_1.GetArmIndices())
				robot.SetDOFValues(sol2, mani_2.GetArmIndices())
				robot.SetDOFValues(sol3, mani_3.GetArmIndices())
				handles.append(env.plot3(points=ObjFrame[0:3,3],pointsize=0.0008,colors=array(((1,0,0))),drawstyle = 1))
				env.UpdatePublishedBodies()

				timepert=timepert+1
				#print timepert
				env.UpdatePublishedBodies()
				time.sleep(0.01)
			#	After perturbation, the system should either use an adaptation strategy or 
			#	just simply follow previous approach
			if v == 'n':
				ind=ind+15
				while LA.norm(ObjFrame[0:3,3]-ppcurve[ind,:])>0.001:
					ObjFrame[0:3,3] = ObjFrame[0:3,3]+(ppcurve[ind,:]-ObjFrame[0:3,3])*0.01
					handles.append(env.plot3(points=ObjFrame[0:3,3],pointsize=0.0008,colors=array(((1,0,0))),drawstyle = 1))
					k3.SetTransform(ObjFrame)
					ObjFrame=k3.GetTransform()
					T1 = dot(ObjFrame, T1InObj)
					T2 = dot(ObjFrame, T2InObj)
					T3 = dot(ObjFrame, T3InObj)
					hf1,hf2,hf3= PlotFrame(env, T1, 0.015)
					hf4,hf5,hf6= PlotFrame(env, T2, 0.015)
					hf7,hf8,hf9= PlotFrame(env, T3, 0.015)
					ikparam1 = IkParameterization(T1[0:3,3],ikmod1.iktype)
					sol1 = mani_1.FindIKSolution(ikparam1,IkFilterOptions.IgnoreJointLimits)					
					ikparam2 = IkParameterization(T2[0:3,3],ikmod2.iktype)
					sol2 = mani_2.FindIKSolution(ikparam2,IkFilterOptions.IgnoreJointLimits)					
					ikparam3 = IkParameterization(T3[0:3,3],ikmod3.iktype)
					sol3 = mani_3.FindIKSolution(ikparam3,IkFilterOptions.IgnoreJointLimits)
					#print sol1, sol2,sol3
					k3.SetTransform(ObjFrame)
					robot.SetDOFValues(sol1, mani_1.GetArmIndices())
					robot.SetDOFValues(sol2, mani_2.GetArmIndices())
					robot.SetDOFValues(sol3, mani_3.GetArmIndices())
					handles.append(env.plot3(points=ObjFrame[0:3,3],pointsize=0.0008,colors=array(((1,0,0))),drawstyle = 1))

					env.UpdatePublishedBodies()
					time.sleep(0.02)
			else:
				print 'start the manifold projection'
				pt = GetProjCurveHand(ObjFrame[0:3,3],eps=0.001)
				handles.append(env.plot3(points=pt,pointsize=0.002,colors=array(((0,0,1))),drawstyle = 1))
				while LA.norm(ObjFrame[0:3,3]-pt)>0.0001:
					ObjFrame[0:3,3] = ObjFrame[0:3,3]+(pt-ObjFrame[0:3,3])*0.02
					handles.append(env.plot3(points=ObjFrame[0:3,3],pointsize=0.0008,colors=array(((1,0,0))),drawstyle = 1))
					k3.SetTransform(ObjFrame)
					ObjFrame=k3.GetTransform()

					T1 = dot(ObjFrame, T1InObj)
					T2 = dot(ObjFrame, T2InObj)
					T3 = dot(ObjFrame, T3InObj)
					hf1,hf2,hf3= PlotFrame(env, T1, 0.015)
					hf4,hf5,hf6= PlotFrame(env, T2, 0.015)
					hf7,hf8,hf9= PlotFrame(env, T3, 0.015)
					
					ikparam1 = IkParameterization(T1[0:3,3],ikmod1.iktype)
					sol1 = mani_1.FindIKSolution(ikparam1,IkFilterOptions.IgnoreJointLimits)					
					ikparam2 = IkParameterization(T2[0:3,3],ikmod2.iktype)
					sol2 = mani_2.FindIKSolution(ikparam2,IkFilterOptions.IgnoreJointLimits)					
					ikparam3 = IkParameterization(T3[0:3,3],ikmod3.iktype)
					sol3 = mani_3.FindIKSolution(ikparam3,IkFilterOptions.IgnoreJointLimits)
					#print sol1, sol2,sol3

					k3.SetTransform(ObjFrame)
					robot.SetDOFValues(sol1, mani_1.GetArmIndices())
					robot.SetDOFValues(sol2, mani_2.GetArmIndices())
					robot.SetDOFValues(sol3, mani_3.GetArmIndices())
					handles.append(env.plot3(points=ObjFrame[0:3,3],pointsize=0.0008,colors=array(((1,0,0))),drawstyle = 1))

					env.UpdatePublishedBodies()
					time.sleep(0.01)
				# Here it should use replanning on the manifold;
				# Here we just reuse the previous planning since we know this is only 1D; 
				dist2=np.sum((ppcurve-pt)**2,axis=1)
				ind = np.argmin(dist2)
				


		else:
			ObjFrame = k3.GetTransform()
			ObjFrame[0:3,3] = pCurve[ind,:]
			T1 = dot(ObjFrame, T1InObj)
			T2 = dot(ObjFrame, T2InObj)
			T3 = dot(ObjFrame, T3InObj)

			hf1,hf2,hf3= PlotFrame(env, T1, 0.015)
			hf4,hf5,hf6= PlotFrame(env, T2, 0.015)
			hf7,hf8,hf9= PlotFrame(env, T3, 0.015)

			ikparam1 = IkParameterization(T1[0:3,3],ikmod1.iktype)
			sol1 = mani_1.FindIKSolution(ikparam1,IkFilterOptions.IgnoreJointLimits)
			
			ikparam2 = IkParameterization(T2[0:3,3],ikmod2.iktype)
			sol2 = mani_2.FindIKSolution(ikparam2,IkFilterOptions.IgnoreJointLimits)
			
			ikparam3 = IkParameterization(T3[0:3,3],ikmod3.iktype)
			sol3 = mani_3.FindIKSolution(ikparam3,IkFilterOptions.IgnoreJointLimits)
			#print sol1, sol2,sol3
			k3.SetTransform(ObjFrame)
			h4,h5,h6= PlotFrame(env, ObjFrame, framescale)
			robot.SetDOFValues(sol1, mani_1.GetArmIndices())
			robot.SetDOFValues(sol2, mani_2.GetArmIndices())
			robot.SetDOFValues(sol3, mani_3.GetArmIndices())
			handles.append(env.plot3(points=ObjFrame[0:3,3],pointsize=0.0008,colors=array(((1,0,0))),drawstyle = 1))
			env.UpdatePublishedBodies()
			ind= ind+1
			time.sleep(0.1)
raw_input('press Enter to continue')



raw_input('press Enter to continue')
