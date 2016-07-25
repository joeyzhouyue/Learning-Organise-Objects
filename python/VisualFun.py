#!/usr/bin/python
"""
This function is to simulate the manifold learning approach
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

from sensor_msgs.msg import JointState


def monitor(msg):
	with env:
		#robot.SetDOFValues(msg.position, range(7))
		Joints=numpy.zeros(7)
		Joints[0] = radians(-30)
		Joints[1] = radians(30)
		Joints[2] = radians(2.0)
		Joints[3] = radians(-90)
		Joints[4] = radians(-20)
		Joints[5] = radians(-30)
		robot.SetDOFValues(Joints,range(7))
		env.UpdatePublishedBodies()

robotFile = '../robots/lwr_hand.robot.xml'

pdb.set_trace()

env = Environment() # create openrave environment
env.SetDebugLevel(DebugLevel.Fatal)
env.SetViewer('qtcoin') # attach viewer (optional)
env.Load(robotFile) # load the robot
robot = env.GetRobots()[0] # get the first robot
framescale=0.3
handles=[]
with env:
    infocylinder = KinBody.Link.GeometryInfo()
    infocylinder._type = KinBody.Link.GeomType.Box
    #infocylinder._t[0,3] = 0.1
    infocylinder._vGeomData = [0.04,0.08,0.05]
    infocylinder._bVisible = True
    infocylinder._fTransparency = 0.1
    infocylinder._vDiffuseColor = [0.133333,0.545098,0.133333]
    k3 = RaveCreateKinBody(env,'')
    k3.InitFromGeometries([infocylinder])
    k3.SetName('tmpcylinder')
    manipulator = robot.SetActiveManipulator("lwr")
    Tend= manipulator.GetEndEffectorTransform()
    #ObjInHand = array([[1, 0,0 , -0.11],[0, 1 ,0 , 0],[0, 0, 1 , 0.1],[0, 0, 0 , 1]])
    ObjInHand = array([[1, 0,0 , 2],[0, 1 ,0 , 0],[0, 0, 1 , 0.1],[0, 0, 0 , 1]])

    ObjInBase =dot(Tend,ObjInHand)
    k3.SetTransform(ObjInBase)
    env.Add(k3,True)
pdb.set_trace()

T=array([[1, 0,0 , 0],[0, 1 ,0 , 0],[0, 0, 1 , 0],[0.015, 0.015, 0 , 1]])
h1=env.drawlinestrip(points=array((T[0:3,3],T[0:3,3]+T[0:3,0]*framescale)),linewidth=5.0, colors=array(((1,0,0,0.5))))
h2=env.drawlinestrip(points=array((T[0:3,3],T[0:3,3]+T[0:3,1]*framescale)),linewidth=5.0, colors=array(((0,1,0,0.5))))
h3=env.drawlinestrip(points=array((T[0:3,3],T[0:3,3]+T[0:3,2]*framescale)),linewidth=5.0, colors=array(((0,0,1,0.5))))

handles.append(h1)
handles.append(h2)
handles.append(h3)

# change the color
ind =0
indlist=[1,2,4,6,8,10,12]
for link in robot.GetLinks():
	if ind in indlist:
		for geom in link.GetGeometries():
			geom.SetDiffuseColor([1,0.4, 0, 0.1])
			geom.SetTransparency(0.5)
	else:
		if ind<14:
			for geom in link.GetGeometries():
				geom.SetDiffuseColor([0.862745,0.862745,0.862745,0.2])
				geom.SetTransparency(0.2)
	ind = ind+1
print "Number of joints:------", repr(robot.GetActiveDOF())
pdb.set_trace()

h=1
phi=2.5
xxrange=array([-0.8,-0.4])
yyrange=array([-0.6,0.6])
nb=20
noise = 0.0
Psurf,xx,yy,zz = GenerateSurface(h, phi, xxrange,yyrange,nb)
Psurfnoise,xxnoise,yynoise,zznoise = GenerateSurface(h, phi, xxrange,0.5*yyrange,nb,noise)
id = random.random_integers(0, shape(Psurfnoise)[0]-1,100)
numpy.savetxt('KUKASurf.txt',Psurfnoise[id,:]) #save the data for learning

handles.append(env.plot3(points=Psurfnoise[id,:],pointsize=0.015,colors=array(((1,0.5,0))),drawstyle=1))
for i in range(nb):
	pxmesh = np.vstack([xx[i,:],yy[i,:],zz[i,:]]).T
	handles.append(env.drawlinestrip(points=pxmesh,linewidth=2.5,colors=array(((0,1,0,0.5)))))
for i in range(nb):
	pxmesh = np.vstack([xx[:,i],yy[:,i],zz[:,i]]).T
	handles.append(env.drawlinestrip(points=pxmesh,linewidth=2.5,colors=array(((0,1,0,0.5)))))

env.UpdatePublishedBodies()
#raw_input('press enter to continue')

# Joints=numpy.zeros(7)
# Joints[0] = radians(-30)
# Joints[1] = radians(30)
# Joints[2] = radians(2.0)
# Joints[3] = radians(-90)
# Joints[4] = radians(-20)
# Joints[5] = radians(-30)
# robot.SetDOFValues(Joints,range(7))

robot.SetDOFValues([radians(10)],[7])
robot.SetDOFValues([radians(30)],[8])
robot.SetDOFValues([radians(45)],[9])
robot.SetDOFValues([radians(45)],[10])
robot.SetDOFValues([radians(10)],[11])
robot.SetDOFValues([radians(30)],[12])
robot.SetDOFValues([radians(45)],[13])
robot.SetDOFValues([radians(45)],[14])
robot.SetDOFValues([radians(-100)],[19])
robot.SetDOFValues([radians(0)],[20])
robot.SetDOFValues([radians(-10)],[21])
robot.SetDOFValues([radians(50)],[22])
env.UpdatePublishedBodies()
raw_input('press enter to continue to use IK')
manipulator = robot.SetActiveManipulator("lwr")
ikmodel = databases.inversekinematics.InverseKinematicsModel(robot=robot,freeindices=[4], iktype=IkParameterization.Type.Transform6D)
if not ikmodel.load():
    ikmodel.autogenerate()
with env:
	idx=1;
	idy=16;
	p1=array([[xx[idx,idy],yy[idx,idy],zz[idx,idy]]])
	p2=array([xx[idx+1,idy],yy[idx+1,idy],zz[idx+1,idy]])
	p3=array([xx[idx,idy+1],yy[idx,idy+1],zz[idx,idy+1]])
	#handles.append(env.plot3(points=p1,pointsize=10.0,colors=array(((1.0, 0.0, 0.)))))
	#handles.append(env.plot3(points=p2,pointsize=10.0,colors=array(((0., 1., 0.)))))
	#handles.append(env.plot3(points=p3,pointsize=10.0,colors=array(((1., 1., 0)))))
	T= eye(4,4)
	T[0:3,0] = (p2-p1)/LA.norm(p2-p1)
	T[0:3,1] = (p3-p1)/LA.norm(p3-p1)
	T[0:3,2] = numpy.cross(T[0:3,0],T[0:3,1]) 
	T[0:3,3] = p1
	framescale=0.1
	h1=env.drawlinestrip(points=array((T[0:3,3],T[0:3,3]+T[0:3,0]*framescale)),linewidth=5.0, colors=array(((1,0,0,0.5))))
	h2=env.drawlinestrip(points=array((T[0:3,3],T[0:3,3]+T[0:3,1]*framescale)),linewidth=5.0, colors=array(((0,1,0.5,0.5))))
	h3=env.drawlinestrip(points=array((T[0:3,3],T[0:3,3]+T[0:3,2]*framescale)),linewidth=5.0, colors=array(((0,0,1,0.5))))

	handles.append(h1)
	handles.append(h2)
	handles.append(h3)
	ObjInManifold = array([[0,0,-1,0],[1,0,0,0],[0,-1,0,0],[0,0,-0.03,1]]).T
	ObjInBase = dot(T,ObjInManifold)
	#k3.SetTransform(ObjInBase)
	Tgoal = dot(ObjInBase,inv(ObjInHand))
	sol = manipulator.FindIKSolution(Tgoal,IkFilterOptions.IgnoreJointLimits)
	AngleRot =0
	while sol is None:
		print "no solution"
		AngleRot = AngleRot+5
		tmpMat = matrixFromAxisAngle([1,0,0],radians(AngleRot))
		print tmpMat
		ObjInManifold = dot(ObjInManifold,tmpMat)
		ObjInBase = dot(T,ObjInManifold)
		k3.SetTransform(ObjInBase)
		Tgoal = dot(ObjInBase,inv(ObjInHand))
		#print Tgoal
		h2=env.drawlinestrip(points=array((Tgoal[0:3,3],Tgoal[0:3,3]+Tgoal[0:3,1]*framescale)),linewidth=5.0, colors=array(((0,0,1,0.5))))
		h3=env.drawlinestrip(points=array((Tgoal[0:3,3],Tgoal[0:3,3]+Tgoal[0:3,2]*framescale)),linewidth=5.0, colors=array(((0,0,1,0.5))))
		handles.append(h2)
		handles.append(h3)
		sol = manipulator.FindIKSolution(Tgoal,IkFilterOptions.IgnoreJointLimits)
		if AngleRot >190:
			break
	CurrJnt = robot.GetActiveDOFValues()[0:7]
	GoalJnt = sol
	tmp = GoalJnt-CurrJnt
	while LA.norm(tmp)>0.02:
		sol =CurrJnt+tmp*0.1	
		robot.SetDOFValues(sol,manipulator.GetArmIndices())
		env.UpdatePublishedBodies()
		Tend= manipulator.GetEndEffectorTransform()
		ObjInBase =dot(Tend,ObjInHand)
		k3.SetTransform(ObjInBase)
		env.UpdatePublishedBodies()
		time.sleep(0.1)
		CurrJnt = robot.GetActiveDOFValues()[0:7]
		tmp = GoalJnt-CurrJnt
raw_input('press enter to move to target')

# set target
with env:
		idx=8;
		idy=8;
		p1e=array([[xx[idx,idy],yy[idx,idy],zz[idx,idy]]])
		p2e=array([xx[idx+1,idy],yy[idx+1,idy],zz[idx+1,idy]])
		p3e=array([xx[idx,idy+1],yy[idx,idy+1],zz[idx,idy+1]])
		T= eye(4,4)
		T[0:3,0] = (p2e-p1e)/LA.norm(p2e-p1e)
		T[0:3,1] = (p3e-p1e)/LA.norm(p3e-p1e)
		T[0:3,2] = numpy.cross(T[0:3,0],T[0:3,1]) 
		T[0:3,3] = p1e
		
		H = GetTangentVec(p1e)
		framescale=0.1
		h1=env.drawlinestrip(points=array((T[0:3,3],T[0:3,3]+T[0:3,0]*framescale)),linewidth=5.0, colors=array(((1,0,0,0.5))))
		h2=env.drawlinestrip(points=array((T[0:3,3],T[0:3,3]+T[0:3,1]*framescale)),linewidth=5.0, colors=array(((0,1,0.5,0.5))))
		h3=env.drawlinestrip(points=array((T[0:3,3],T[0:3,3]+T[0:3,2]*framescale)),linewidth=5.0, colors=array(((0,0,1,0.5))))
		handles.append(h1)
		handles.append(h2)
		handles.append(h3)

		#draw target frame
		h4=env.drawarrow(p1=T[0:3,3],p2=T[0:3,3]+H[0:3,0]*framescale,linewidth=0.005, color=[1,0,0])
		h5=env.drawarrow(p1=T[0:3,3],p2=T[0:3,3]+H[0:3,1]*framescale,linewidth=0.005, color=[0,1,0])
		handles.append(h4)
		handles.append(h5)



		ObjInBase = dot(T,ObjInManifold)
		Tgoale = dot(ObjInBase,inv(ObjInHand))
		Tend= manipulator.GetEndEffectorTransform()
		# add interpolation
		tmp= (Tgoale[0:3,3]-Tend[0:3,3])
		TgoalTmp = deepcopy(Tgoale)
		TgoalTmp[0:3,3] = Tend[0:3,3] + tmp*0.01
		ind=0

		indpert =randint(30,80)
		indpert=50
		print indpert

		v = raw_input('with projection or not(y/n):\n')
		print v

		while LA.norm(tmp)>0.01:
				ind = ind+1
				# raw_input('press enter to continue')
				if ind == indpert:
					#start to perturb
					timepert=0
					dirpert = 2*random.random((3))-1
					dirpert[0] = abs(dirpert[0])
					dirpert[2] = abs(dirpert[2])+1
					while timepert<10:
						timepert=timepert+1					
						hDisturb = env.drawarrow(p1= Tend[0:3,3],p2= Tend[0:3,3]+0.2*dirpert,linewidth=0.01, color=[1,0,0])
						TgoalTmp[0:3,3] = Tend[0:3,3] + dirpert/LA.norm(dirpert)*0.02
						sol = manipulator.FindIKSolution(TgoalTmp,IkFilterOptions.IgnoreJointLimits)
						robot.SetDOFValues(sol,manipulator.GetArmIndices())				
						Tend= manipulator.GetEndEffectorTransform()
						ObjInBase =dot(Tend,ObjInHand)
						k3.SetTransform(ObjInBase)
						handles.append(env.plot3(points=ObjInBase[0:3,3],pointsize=0.01,colors=array(((1.0, 0.0, 0.))),drawstyle = 1))					
						env.UpdatePublishedBodies()
						time.sleep(0.05)
						del hDisturb
					if v=='y':
						# plot the vector field;
						Psurf,xx,yy,zz = GenerateSurface(h, phi, xxrange,yyrange,10)
						for i in range(shape(Psurf)[0]):
							
							H = GetTangentVec(Psurf[i,:])
							h4=env.drawarrow(p1=Psurf[i,:],p2=Psurf[i,:]+H[0:3,0]*framescale,linewidth=0.005, color=[1,0,0])
							h5=env.drawarrow(p1=Psurf[i,:],p2=Psurf[i,:]+H[0:3,1]*framescale,linewidth=0.005, color=[0,1,0])
							handles.append(h4)
							handles.append(h5)
							"""
							Ptmp1=array([Psurf[i,0],Psurf[i,1],Psurf[i,2]+0.1])
							H = GetTangentVec(Ptmp1)
							h4=env.drawarrow(p1=Ptmp1,p2=Ptmp1+H[0:3,0]*framescale,linewidth=0.006, color=[1,0,0])
							h5=env.drawarrow(p1=Ptmp1,p2=Ptmp1+H[0:3,1]*framescale,linewidth=0.006, color=[0,1,0])
							handles.append(h4)
							handles.append(h5)
							"""


						#here we do the projection
						pt = GetProjection((ObjInBase[0:3,3]).T)
						handles.append(env.plot3(points=pt,pointsize=0.02,colors=array(((0.541176, 0.168627, 0.886275))),drawstyle = 1))
						Tgoale0 = deepcopy(Tgoale) 
						Tgoale0[0:3,3]= pt
						Tgoale0 = dot(Tgoale0,inv(ObjInHand))
						tmp= (Tgoale0[0:3,3]-Tend[0:3,3])
						while LA.norm(tmp)>0.01:
							TgoalTmp[0:3,3] = Tend[0:3,3] + tmp*0.02
							sol = manipulator.FindIKSolution(TgoalTmp,IkFilterOptions.IgnoreJointLimits)
							robot.SetDOFValues(sol,manipulator.GetArmIndices())				
							Tend= manipulator.GetEndEffectorTransform()
							ObjInBase =dot(Tend,ObjInHand)
							k3.SetTransform(ObjInBase)
							handles.append(env.plot3(points=ObjInBase[0:3,3],pointsize=0.01,colors=array(((1.0, 0.0, 0.))),drawstyle = 1))
							env.UpdatePublishedBodies()
							time.sleep(0.02)
							tmp = (Tgoale0[0:3,3]-Tend[0:3,3])
					tmp=1
				else:
					#print 'here.............'
					
					sol = manipulator.FindIKSolution(TgoalTmp,IkFilterOptions.IgnoreJointLimits)
					robot.SetDOFValues(sol,manipulator.GetArmIndices())				
					Tend= manipulator.GetEndEffectorTransform()
					ObjInBase =dot(Tend,ObjInHand)
					k3.SetTransform(ObjInBase)

					# pt = GetProjection((ObjInBase[0:3,3]).T)
					# ObjInBase[0:3,3]=pt
					
					handles.append(env.plot3(points=ObjInBase[0:3,3],pointsize=0.01,colors=array(((1.0, 0.0, 0.))),drawstyle = 1))
					env.UpdatePublishedBodies()
					time.sleep(0.02)
					tmp= (Tgoale[0:3,3]-Tend[0:3,3])
					#handles.append(env.plot3(points=Tgoale[0:3,3],pointsize=0.02,colors=array(((1.0, 0.0, 0.))),drawstyle = 1))
					TgoalTmp[0:3,3] = Tend[0:3,3] + tmp*0.01


raw_input('press enter to stop')
raw_input('press enter to exit')


#h4= env.drawtrimesh(points=array(ptTrj),indices=ptInd,colors=array([0.,1.,0.,0.1]))
#handles.append(h4)
#env.UpdatePublishedBodies()
#############
#framescale=0.5
#h4=env.drawlinestrip(points=array((T[0:3,3],T[0:3,3]+T[0:3,0]*framescale)),linewidth=5.0, colors=array(((1,0,0,0.5))))
#h5=env.drawlinestrip(points=array((T[0:3,3],T[0:3,3]+T[0:3,1]*framescale)),linewidth=5.0, colors=array(((0,1,0,0.5))))
#h6=env.drawlinestrip(points=array((T[0:3,3],T[0:3,3]+T[0:3,2]*framescale)),linewidth=5.0, colors=array(((0,0,1,0.5))))
#handles.append(h1)
#handles.append(h2)
#handles.append(h3)
#env.UpdatePublishedBodies()
#############

#rospy.init_node('listener', anonymous=True)
#rospy.Subscriber("/kukaAllegroHand/robot_cmd", JointState, monitor, queue_size = 1)
#rospy.spin()




