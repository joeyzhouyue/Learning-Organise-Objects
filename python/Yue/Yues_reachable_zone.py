from openravepy import *
import random
import pickle
import math, time
import numpy as np
import sys
sys.path.insert(0, '../')
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
	from Yues_functions import * 
	import pdb


env = Environment() # create the environment
env.SetViewer('qtcoin') # start the viewer
env.SetDebugLevel(DebugLevel.Fatal)
env.Load('../../robots/Kitchen_LASA/Kitchen_Robot.robot.xml') # load a scene
robot = env.GetRobots()[0]
manip = robot.SetActiveManipulator("lwr")
manipTool_Transform = robot.GetLinks()[15].GetTransform()

h1,h2,h3 = PlotFrame(env, np.identity(4), 0.2) # axis origin
h4,h5,h6 = PlotFrame(env, manipTool_Transform, 0.2)

# granularity_index = 3
# the joint number 0
jointNumber = 0
# pdb.set_trace() #breakpoint

numberOfSamplesPerJoint = 5
sample_values_of_all_joint = []

for i_joint in range(7): # 7 joints on the arm
  # i_joint: this joint in this loop
  # find random joint value
  sample_values_of_this_joint = giveSamplesOfJointValue(robot, numberOfSamplesPerJoint, i_joint)
  # add into the total joint array
  sample_values_of_all_joint.append(sample_values_of_this_joint)
  # pdb.set_trace() #
sample_values_of_all_joint = array(sample_values_of_all_joint) # conver to array data type with rows being joints
# generade the all possible joint value combinations
sample_values_of_all_joint_array = giveSamplesOfJointValueInArray(sample_values_of_all_joint)

raw_input('press enter to save joint value sampling')

# save the joint value data
with open('sample_values_of_all_joint_array.pickle','w') as f: 
	pickle.dump(sample_values_of_all_joint_array, f)

raw_input('press enter to begin end effector position and orientation sampling')

numberOfSamplesTot = shape(sample_values_of_all_joint_array)[0]
Pos_Ori_Tool = []
validConfigWithToolPosOri = [] # index: 0-6: joint values, 7-9: tool position, 10-16: tool orientation 
Transform_for_plot = [] # only for plot the orientation of end-effector
with env:
	for i_sample_Joint_Combi in range(numberOfSamplesTot): # for every group of sample data
		currentJointValue = sample_values_of_all_joint_array[i_sample_Joint_Combi,:]
		#pdb.set_trace() #
		robot.SetDOFValues(currentJointValue,range(7)) 
		if not robot.CheckSelfCollision():
			# pdb.set_trace()
			# env.UpdatePublishedBodies()
			Pos_Tool = np.asmatrix(robot.GetLinks()[15].GetTransform()[0:3,3])
			Ori_Tool = np.asmatrix(robot.GetLinks()[15].GetTransform()[0:3,0:3]).T
			Pos_Ori_Tool = np.asarray(np.vstack((Pos_Tool, Ori_Tool)).T)
			Pos_Ori_Tool_vec = Pos_Ori_Tool.flatten('F')
			validConfigWithToolPosOri.append(np.append(currentJointValue, Pos_Ori_Tool_vec))
			Transform_for_plot.append(robot.GetLinks()[15].GetTransform().flatten("F"))
			# handles.append(env.plot3(points=np.asarray(Pos_Tool), pointsize=0.02, colors=array(((1,0,0))),drawstyle = 1))
			# handles.append(PlotFrame(env, robot.GetLinks()[15].GetTransform(), 0.03)[2])
			# Tool_transform.append(robot.GetLinks()[15].GetTransform())

	# print env.CheckCollision(robot.GetLinks()[8],obj)
validConfigWithToolPosOri = array(validConfigWithToolPosOri)
Transform_for_plot = array(Transform_for_plot) # for plotting the orientation of end effector

raw_input('press enter to save end effector position and orientation sampling')

# save the configuration data
with open('validConfigWithToolPosOri.pickle','w') as ff: 
	pickle.dump(validConfigWithToolPosOri, ff)
pdb.set_trace() #
with open('Transform_for_plot.pickle','w') as fff: 
	pickle.dump(Transform_for_plot, fff)
raw_input('press enter to finish')