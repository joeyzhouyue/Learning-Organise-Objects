#!/usr/bin/python
"""
This function is to simulate the partial fingertip grasps. (conceptual idea)
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

env = Environment() # create openrave environment
env.SetViewer('qtcoin') # attach viewer (optional)
env.Load('../robots/allegroGrasp.robot.xml') # load 16 dof allegro hand
robot = env.GetRobots()[0] # get the first robot
handles = []
with env:
	robot.SetDOFValues([radians(30)],[0])
	robot.SetDOFValues([radians(20)],[1])
	robot.SetDOFValues([radians(10)],[2])
	robot.SetDOFValues([radians(5)],[3])

	robot.SetDOFValues([radians(30)],[4])
	robot.SetDOFValues([radians(90)],[5])
	robot.SetDOFValues([radians(50)],[6])
	robot.SetDOFValues([radians(10)],[7])
	
	robot.SetDOFValues([radians(20)],[8])
	robot.SetDOFValues([radians(90)],[9])
	robot.SetDOFValues([radians(40)],[10])
	robot.SetDOFValues([radians(0)],[11])

	robot.SetDOFValues([radians(-90)],[12])
	robot.SetDOFValues([radians(-30)],[13])
	robot.SetDOFValues([radians(0)],[14])
	robot.SetDOFValues([radians(120)],[15])
T=array([[1, 0,0 , 0],[0, 1 ,0 , 0],[0, 0, 1 , 0],[0.0, 0.0, 0 , 1]]).T
TKP=array([[-1, 0,0 , 0],[0, 0 ,-1 , 0],[0, -1, 0 , 0],[0.35, 0, 0.02,1]]).T
tmp = matrixFromAxisAngle([0,0,1],radians(30))
TKP=dot(TKP,tmp)
tmp = matrixFromAxisAngle([0,1,0],radians(-30))
TKP=dot(TKP,tmp)
IndexInKP  = array([[1,0,0 , 0],[0, 1,0, 0],[0, 0, 1 , 0],[0.0, 0.0, 0.0 , 1]]).T
framescale=0.03
h1,h2,h3= PlotFrame(env, T, 0.05)
# hk1,hk2,hk3= PlotFrame(env, TKP, framescale)
obj = env.ReadKinBodyXMLFile('../robots/' + 'pliers' + '.kinbody.xml')
env.Add(obj)

TF1 = robot.GetLinks()[6].GetTransform()
#hf1,hf2,hf3= PlotFrame(env, TF1, 0.03)
IndexInKP  = array([[-1, 0,0 , 0],[0, 0 ,-1, 0],[0, -1, 0 , 0],[0.0, 0.0, 0.0 , 1]]).T
IndexInObj = dot(TKP,IndexInKP)
HandInObj = dot(IndexInObj,inv(TF1))


robot.SetTransform(HandInObj)
env.UpdatePublishedBodies()

TF1 = robot.GetLinks()[6].GetTransform()
hf1,hf2,hf3= PlotFrame(env, TF1, 0.03)
raw_input('press enter to continue')
with env:
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
			env.UpdatePublishedBodies()
			time.sleep(0.05)
raw_input('press enter to continue')