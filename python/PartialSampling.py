#!/usr/bin/python
"""
This function is to sample the partial fingertip grasp
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

class PlotSpinner(threading.Thread):
    def __init__(self,handle):
        threading.Thread.__init__(self)
        self.starttime = time.time()
        self.handle=handle
        self.ok = True
    def run(self):
        while self.ok:
            self.handle.SetShow(bool(mod(time.time()-self.starttime,2.0) < 2.0))
            #time.sleep(0.0001)


env = Environment() # create openrave environment
env.SetViewer('qtcoin') # attach viewer (optional)
env.Load('../robots/allegroGrasp.robot.xml') # load 16 dof allegro hand
robot = env.GetRobots()[0] # get the first robot

time.sleep(0.5)
nbSample = 10
handles = []
AllegroHandData = []
nbValid = 0
vis = True
with env: # lock the environment since robot will be used
		lower,upper = robot.GetActiveDOFLimits()
		for i in range(nbSample):
			print i 
			#raw_input('press Enter to continue')

			JntTarget = numpy.random.rand(len(lower))*(upper-lower)+lower
			#print JntTarget
			robot.SetDOFValues([JntTarget[0]],[0])
			robot.SetDOFValues([JntTarget[1]],[1])
			robot.SetDOFValues([JntTarget[2]],[2])
			robot.SetDOFValues([JntTarget[3]],[3])

			robot.SetDOFValues([radians(-5)],[4])
			robot.SetDOFValues([JntTarget[5]],[5])
			robot.SetDOFValues([JntTarget[6]],[6])
			robot.SetDOFValues([JntTarget[7]],[7])

			robot.SetDOFValues([radians(-10)],[8])
			robot.SetDOFValues([JntTarget[5]],[9])
			robot.SetDOFValues([JntTarget[6]],[10])
			robot.SetDOFValues([JntTarget[7]],[11])
			
			robot.SetDOFValues([radians(-90)],[12])
			robot.SetDOFValues([radians(-5)],[13])
			robot.SetDOFValues([JntTarget[14]],[14])
			robot.SetDOFValues([JntTarget[15]],[15])


			if not robot.CheckSelfCollision():

				T0 = robot.GetLinks()[6].GetTransform()  #index
				T1 = robot.GetLinks()[12].GetTransform() #middle
				T2 = robot.GetLinks()[24].GetTransform() #thumb

				idList=[7,8,9,10,11,12,13,14,15,16,17,18,20,21,22,23,24]
				idx= 0
				datanew=[]
				for links in robot.GetLinks():
					if idx in idList:
						TF = links.GetTransform()
						TFInIndex = dot(inv(T0),TF)
						# hf1,hf2,hf3= PlotFrame(env, TF, 0.03)
						# handles.append(hf1)
						# handles.append(hf2)
						# handles.append(hf3)
						datanew.append(TFInIndex[0:3,3])
					idx = idx+1

				datanew= np.array(datanew)
				datanew= np.hstack(datanew)
				currentJoint= robot.GetActiveDOFValues()
				datanew= np.hstack([datanew,currentJoint])
				AllegroHandData.append(datanew)
				nbValid  = nbValid +1
				if(vis):
					try:

						handles = []
						env.UpdatePublishedBodies()
						h1,h2,h3= PlotFrame(env, T0, 0.05)
						h4,h5,h6= PlotFrame(env, T1, 0.05)
						h7,h8,h9= PlotFrame(env, T2, 0.05)
						handles.append([h1,h2,h3,h4,h5,h6,h7,h8,h9])						
						spinner = PlotSpinner(handles[-1])
						spinner.start()
						#raw_input('press Enter to continue')
						time.sleep(0.1)
					finally:
						if spinner is not None:
						  spinner.ok = False

numpy.savetxt('AllegroHandData.txt',AllegroHandData)
print nbValid
print('The number of valid sample is : '  + str(size(AllegroHandData,0)))
raw_input('press Enter to exit')