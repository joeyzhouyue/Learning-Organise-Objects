#!/usr/bin/python
"""
This function is to simulate the partial fingertip grasps and test the learned model
"""
# drill
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

pdb.set_trace()
objname='pliers'
env = Environment() # create openrave environment
env.SetViewer('qtcoin') # attach viewer (optional)
obj = env.ReadKinBodyXMLFile('../robots/' + objname + '.kinbody.xml')
env.Add(obj)

env.Load('../robots/allegroGrasp.robot.xml') # load 16 dof allegro hand
robot = env.GetRobots()[0] # get the first robot

pdb.set_trace()

handles = []
dataGrasp = np.genfromtxt(objname+'.txt')
print dataGrasp


T=array([[1, 0,0 , 0],[0, 1 ,0 , 0],[0, 0, 1 , 0],[0.0, 0.0, 0 , 1]]).T
if objname=='spatula':
	"""
	TKP=array([[-1, 0,0 , 0],[0, 0 ,-1 , 0],[0, -1, 0 , 0],[0.45, 0, 0.03,1]]).T
	tmp = matrixFromAxisAngle([0,1,0],radians(-90))
	TKP=dot(TKP,tmp)
	IndexInKP  = array([[1,0,0 , 0],[0, 1,0, 0],[0, 0, 1 , 0],[0.0, 0.0, 0.0 , 1]]).T
	"""
	TKP=array([[-1, 0,0 , 0],[0, 0 ,-1 , 0],[0, -1, 0 , 0],[0.30, 0, 0.02,1]]).T
	tmp = matrixFromAxisAngle([0,0,1],radians(10))
	TKP=dot(TKP,tmp)
	tmp = matrixFromAxisAngle([0,1,0],radians(-20))
	TKP=dot(TKP,tmp)
	print inv(TKP)
	IndexInKP  = array([[1,0,0 , 0],[0, 1,0, 0],[0, 0, 1 , 0],[0.0, 0.0, 0.0 , 1]]).T
	dataGrasp[2] =5
	dataGrasp[3] =5

if objname=='drill':
	TKP=array([[0, 0,-1 , 0],[1, 0 ,0 , 0],[0, -1, 0 , 0],[0.15,-0.075, 0 , 1]]).T
	IndexInKP  = array([[1,0,0 , 0],[0, 1,0, 0],[0, 0, 1 , 0],[0.0, 0.0, 0.0 , 1]]).T

if objname=='pliers':
	TKP=array([[1, 0,0 , 0],[0, 1 ,0 , 0],[0, 0, 1 , 0],[0.0,0.045, 0.05 , 1]]).T
	tmp = matrixFromAxisAngle([1,0,0],radians(15))
	TKP=dot(TKP,tmp)
	IndexInKP  = array([[1,0,0 , 0],[0, 1,0, 0],[0, 0, 1 , 0],[0.0, 0.0, 0.0 , 1]]).T

if objname=='drinkbottle':
	TKP=array([[-1, 0,0 , 0],[0, 0 ,-1 , 0],[0, -1, 0 , 0],[0.0,0.06, 0.07 , 1]]).T
	tmp = matrixFromAxisAngle([0,1,0],radians(-45))
	TKP=dot(TKP,tmp)
	IndexInKP  = array([[1,0,0 , 0],[0, 1,0, 0],[0, 0, 1 , 0],[0.0, 0.0, 0.0 , 1]]).T	

if objname == 'spray':
	TKP=array([[1, 0,0 , 0],[0, 1 ,0 , 0],[0, 0, 1 , 0],[-0.04, 0.115, 0.035 , 1]]).T
	tmp = matrixFromAxisAngle([0,1,0],radians(-45))
	TKP=dot(TKP,tmp)
	IndexInKP  = array([[-1, 0,0 , 0],[0, 0 ,-1, 0],[0, -1, 0 , 0],[0.0, 0.0, 0.0 , 1]]).T

if objname == 'camera':
	TKP=array([[1, 0,0 , 0],[0, 0 ,-1 , 0],[0, 1, 0 , 0],[0.01, 0.1, 0.00 , 1]]).T
	tmp = matrixFromAxisAngle([1,0,0],radians(-60))
	TKP=dot(TKP,tmp)
	IndexInKP  = array([[1,0,0 , 0],[0, 1,0, 0],[0, 0, 1 , 0],[0.0, 0.0, 0.0 , 1]]).T

if objname == 'spray-top':
	TKP=array([[-1, 0,0 , 0],[0, -1 ,0 , 0],[0, 0, 1 , 0],[-0.01, 0.23, 0.0, 1]]).T
	tmp = matrixFromAxisAngle([0,1,0],radians(180))
	TKP=dot(TKP,tmp)
	# tmp = matrixFromAxisAngle([0,0,1],radians(-30))
	# TKP=dot(TKP,tmp)
	tmp = matrixFromAxisAngle([1,0,0],radians(-35))
	TKP=dot(TKP,tmp)
	IndexInKP  = array([[1,0,0 , 0],[0, 1 ,0, 0],[0, 0, 1 , 0],[0.0, 0.0, 0.0 , 1]]).T

if objname=='bbqlighter':
	TKP=array([[-1, 0,0 , 0],[0, 0 ,1 , 0],[0, 1, 0 , 0],[0.02,0.05, 0.20 , 1]]).T
	tmp = matrixFromAxisAngle([1,0,0],radians(-30))
	TKP=dot(TKP,tmp)
	IndexInKP  = array([[1,0,0 , 0],[0, 1,0, 0],[0, 0, 1 , 0],[0.0, 0.0, 0.0 , 1]]).T
with env:
	# print dataGrasp 
	for i in range(16):
		robot.SetDOFValues([radians(dataGrasp[i])],[i])
env.UpdatePublishedBodies()

pdb.set_trace()

framescale=0.03
h1,h2,h3= PlotFrame(env, T, 0.05)
#hk1,hk2,hk3= PlotFrame(env, TKP, framescale)
TF1 = robot.GetLinks()[6].GetTransform()
#hf1,hf2,hf3= PlotFrame(env, TF1, 0.03)

IndexInObj = dot(TKP,IndexInKP)
HandInObj = dot(IndexInObj,inv(TF1))
hf1,hf2,hf3= PlotFrame(env, IndexInObj, 0.03)


robot.SetTransform(HandInObj)

env.UpdatePublishedBodies()
TF1 = robot.GetLinks()[6].GetTransform()

# hf1,hf2,hf3= PlotFrame(env, TF1, 0.03)
# startsim: This is only for simulation purpose 
with env:
	currentJoint = robot.GetActiveDOFValues()
	if env.CheckCollision(robot.GetLinks()[22],obj):
		#robot.SetDOFValues([radians(-79)],[12])
		#robot.SetDOFValues([radians(-5)],[13])
		robot.SetDOFValues([-radians(45)+currentJoint[14]],[14])
		robot.SetDOFValues([radians(0)],[15])
		env.UpdatePublishedBodies()
###endsim
pdb.set_trace()

raw_input('press enter to close the finger')
with env:
	pdb.set_trace()
	while not (env.CheckCollision(robot.GetLinks()[5],obj) and env.CheckCollision(robot.GetLinks()[11],obj) and
	 env.CheckCollision(robot.GetLinks()[17],obj) and env.CheckCollision(robot.GetLinks()[22],obj)):
		currentJoint = robot.GetActiveDOFValues()
		if not env.CheckCollision(robot.GetLinks()[5],obj):
			print env.CheckCollision(robot.GetLinks()[5],obj)
			
			robot.SetDOFValues([radians(1)+currentJoint[1]],[1])
			#robot.SetDOFValues([radians(41)],[2])
			#robot.SetDOFValues([radians(45)],[3])
		if not env.CheckCollision(robot.GetLinks()[11],obj):
			#robot.SetDOFValues([radians(-5)],[4])
			robot.SetDOFValues([radians(1)+currentJoint[5]],[5])
			#robot.SetDOFValues([radians(45)],[6])
			#robot.SetDOFValues([radians(47)],[7])
		if not env.CheckCollision(robot.GetLinks()[17],obj):
			#robot.SetDOFValues([radians(-10)],[8])
			robot.SetDOFValues([radians(1)+currentJoint[9]],[9])
			#robot.SetDOFValues([radians(45)],[10])
			#robot.SetDOFValues([radians(47)],[11])
		if not env.CheckCollision(robot.GetLinks()[22],obj):
			#robot.SetDOFValues([radians(-79)],[12])
			#robot.SetDOFValues([radians(-5)],[13])
			robot.SetDOFValues([radians(1)+currentJoint[14]],[14])
			#robot.SetDOFValues([radians(42)],[15])
			# pdb.set_trace()
			env.UpdatePublishedBodies()
			time.sleep(0.01)
			# pdb.set_trace()

raw_input('press enter to continue')