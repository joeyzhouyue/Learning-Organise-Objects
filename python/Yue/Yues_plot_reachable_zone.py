from openravepy import *
from types import *
import random
import pickle
import math, time
import numpy as np
from openravepy import misc
import sys
sys.path.insert(0, '../')
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

# load the joint value data
#with open('sample_values_of_all_joint_array.pickle') as f: 
#	sample_values_of_all_joint_array = pickle.load(f)

# load the configuration data
with open('validConfigWithToolPosOri.pickle') as ff: 
	validConfigWithToolPosOri = pickle.load(ff)
with open('Transform_for_plot.pickle') as fff: 
	Transform_for_plot = pickle.load(fff)

env = Environment() # create the environment
env.SetViewer('qtcoin') # start the viewer
env.SetDebugLevel(DebugLevel.Fatal)
env.Load('../../robots/Kitchen_LASA/Kitchen_Robot.robot.xml') # load a scene
robot = env.GetRobots()[0]
robotBaseInitiX= 3
robotBaseInitiY = -1
robotBasePos = robot.GetTransform()
robotBasePos[0:2,3] = array([robotBaseInitiX, robotBaseInitiY])
robot.SetTransform(robotBasePos)


manip = robot.SetActiveManipulator("lwr")
manipTool_Transform = robot.GetLinks()[15].GetTransform()
handles = []

h1,h2,h3 = PlotFrame(env, np.identity(4), 0.2) # axis origin
h4,h5,h6 = PlotFrame(env, manipTool_Transform, 0.2)
# handles.append(h1), handles.append(h2), handles.append(h3)
# handles.append(h4), handles.append(h5), handles.append(h6)

pdb.set_trace() #

#	if i_sample_Joint_Combi < 10:
#		time.sleep(0.01)
#	elif i_sample_Joint_Combi < 20:
#		time.sleep(0.05)
#	elif i_sample_Joint_Combi < 50:
#		time.sleep(0.001)


# manipprob.MoveManipulator(goal)



"""
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
"""
plot_or_not = raw_input('plot the position and orientation of the end effector? (y/n): ')
if plot_or_not is 'y':
	numberOfValidConfig = shape(validConfigWithToolPosOri)[0]
	for i_numberOfValidConfig in range(numberOfValidConfig):
		if i_numberOfValidConfig % 2 is 0: # Pos_Tool = np.asmatrix(robot.GetLinks()[15].GetTransform()[0:3,3])
			Pos_this_config = validConfigWithToolPosOri[i_numberOfValidConfig][7:10]
			Orientatino_this_config = Transform_for_plot[i_numberOfValidConfig,:].reshape((4,4), order = 'F')
			handles.append(env.plot3(points=Pos_this_config, pointsize=0.02, colors=array(((1,0,0))),drawstyle = 1))
			# handles.append(PlotFrame(env, Orientatino_this_config, 0.05)[2])
		if i_numberOfValidConfig % 10000 is 0:
			raw_input('press enter to continue:')

	raw_input('press enter to finish the game')
else:
	raw_input('press enter to finish the game')




pdb.set_trace() #breakpoint
