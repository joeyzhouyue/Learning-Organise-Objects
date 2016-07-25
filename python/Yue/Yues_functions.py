from openravepy import *
import sys
sys.path.insert(0, '../')
sys.path.insert(0, 'imported_library')
from Dijkstra_shortest_path import *
from types import *
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
# import rospy
from copy import copy, deepcopy
from GraspFun import *
import pdb


#----------------------------------------------------------------------------------------------------
# randomly generate joint values
def giveSamplesOfJointValue (robot, numberOfSamples, jointNumber):
 	limit_values = robot.GetActiveDOFLimits()
	limit_values = array(limit_values)
	limit_values_of_this_joint = [limit_values[0][jointNumber], limit_values[1][jointNumber]]
	mu = np.mean(limit_values_of_this_joint)
	sigma = (mu - limit_values_of_this_joint[0])/2 # choose sigma such that 2*sigma = half
	samples = np.random.normal(mu, sigma, numberOfSamples)
	for sample_index in range(len(samples)):
		if samples[sample_index] > limit_values_of_this_joint[1]:
			samples[sample_index] = limit_values_of_this_joint[1]
		elif samples[sample_index] < limit_values_of_this_joint[0]:
			samples[sample_index] = limit_values_of_this_joint[0]

	# pdb.set_trace() #breakpoint
	return samples

#---------------------------------------------------------------------------------------------------
# give all possible joint value combinations (assume 7 joints)
# columns: joints
# rows: possible joint value combinations
def giveSamplesOfJointValueInArray(sample_values_of_all_joint):
	# sample_values_of_all_joint = np.round(sample_values_of_all_joint,decimals=3)
	numberOfJoints = shape(sample_values_of_all_joint)[0]
	numberOfSamplesEveryJoint = shape(sample_values_of_all_joint)[1]
	sampleInJoint1 = sample_values_of_all_joint[0,:]
	sampleInJoint2 = sample_values_of_all_joint[1,:]
	sampleInJoint3 = sample_values_of_all_joint[2,:]
	sampleInJoint4 = sample_values_of_all_joint[3,:]
	sampleInJoint5 = sample_values_of_all_joint[4,:]
	sampleInJoint6 = sample_values_of_all_joint[5,:]
	sampleInJoint7 = sample_values_of_all_joint[6,:]

	# pdb.set_trace() #breakpoint

	sample_values_of_all_joint_array = zeros(7)
	for i_sampleInJoint1 in range(numberOfSamplesEveryJoint):
		jointValueJoint1 = round(sampleInJoint1[i_sampleInJoint1],3)
		one_possible_joint_value_group = np.zeros(7)
		one_possible_joint_value_group[0] = jointValueJoint1
		for i_sampleInJoint2 in range(numberOfSamplesEveryJoint):
			jointValueJoint2 = round(sampleInJoint2[i_sampleInJoint2],3)
			one_possible_joint_value_group[1] = jointValueJoint2
			for i_sampleInJoint3 in range(numberOfSamplesEveryJoint):
				jointValueJoint3 = round(sampleInJoint3[i_sampleInJoint3],3)
				one_possible_joint_value_group[2] = jointValueJoint3
				for i_sampleInJoint4 in range(numberOfSamplesEveryJoint):
					jointValueJoint4 = round(sampleInJoint4[i_sampleInJoint4],3)
					one_possible_joint_value_group[3] = jointValueJoint4
					for i_sampleInJoint5 in range(numberOfSamplesEveryJoint):
						jointValueJoint5 = round(sampleInJoint5[i_sampleInJoint5],3)
						one_possible_joint_value_group[4] = jointValueJoint5
						for i_sampleInJoint6 in range(numberOfSamplesEveryJoint):
							jointValueJoint6 = round(sampleInJoint6[i_sampleInJoint6],3)
							one_possible_joint_value_group[5] = jointValueJoint6
							for i_sampleInJoint7 in range(numberOfSamplesEveryJoint):
								jointValueJoint7 = round(sampleInJoint7[i_sampleInJoint7],3)
								one_possible_joint_value_group[6] = jointValueJoint7
								sample_values_of_all_joint_array = np.vstack((sample_values_of_all_joint_array, one_possible_joint_value_group))
								# pdb.set_trace() #breakpointn
	sample_values_of_all_joint_array = np.delete(sample_values_of_all_joint_array, 0, 0)
	return sample_values_of_all_joint_array

#--------------------------------------------------------------------------------------------------------------
# randomly generate object position on the tables
def gerateRandomPositionObjects(table1, table2, Obj):
	tableObj1 = table1[0]
	length_table1 = table1[1]
	width_table1 = table1[2]
	height_table1 = table1[3]
	tableObj2 = table2[0]
	length_table2 = table2[1]
	width_table2 = table2[2]
	height_table2 = table2[3]
	# place the object near to the edge of the tables
	if np.random.uniform(0,1,1) > 0.5: # the table 1
		pos_on_length = tableObj1.GetTransform()[0,3] + np.random.uniform(-length_table1/2, -length_table1/2 + length_table1 / 8, 1)
		pos_on_width = tableObj1.GetTransform()[1,3] + np.random.uniform(-width_table1/2, width_table1/2, 1)
		pos_on_height = height_table1 + 0.005
	else: # the table 2
		pos_on_length = tableObj2.GetTransform()[0,3] + np.random.uniform(length_table2/4, length_table2/2, 1)
		pos_on_width = tableObj2.GetTransform()[1,3] + np.random.uniform(-width_table2/2, width_table2/2, 1)
		pos_on_height = height_table2 + 0.005
	position = [pos_on_length, pos_on_width, pos_on_height]
	posArray = Obj.GetTransform()
	posArray[0:3,3] = position
	return posArray

def genrateRandomPositionObjectsInARectAreaWithZRot(area,objectList,env, randomType):
	centerTableX = area[0]
	centerTableY = area[1]
	lengthX = area[2] # table 1
	lengthY = area[3]
	posInXMin = centerTableX - lengthX/2
	posInXMax = centerTableX + lengthX/2
	posInYMin = centerTableY - lengthY/2
	posInYMax = centerTableY + lengthY/2
	posInZ = area[4]	
	cfgObjList = []
	distanceThreshold = 0.15
	distanceThresholdForkKnife = 0.2
	for i_item in range(shape(objectList)[0]):
		thereIsCollision = True
		while(thereIsCollision):
			# pdb.set_trace() # breakpoint
			if randomType == 'reachable':
				if objectList[i_item].GetName() == 'kitchen_knife' or objectList[i_item].GetName() == 'fork' or objectList[i_item].GetName() == 'kitchen_knife0' or objectList[i_item].GetName() == 'fork0':
					# pdb.set_trace() # breakpoint
					posInX = centerTableX + np.random.uniform(-lengthX/10+lengthX/20, lengthX/10-lengthX/20, 1)
					posInY = centerTableY + np.random.uniform(-lengthY/5+lengthY/20, lengthX/10-lengthY/20, 1)
				else:
					posInX = centerTableX + np.random.uniform(-lengthX/2+lengthX/20, lengthX/2-lengthX/20, 1)
					posInY = centerTableY + np.random.uniform(-lengthY/2+lengthY/20, lengthY/2-lengthY/20, 1)
			else:
				posInX = centerTableX + np.random.uniform(-lengthX/2+lengthX/20, lengthX/2-lengthX/20, 1)
				posInY = centerTableY + np.random.uniform(-lengthY/2+lengthY/20, lengthY/2-lengthY/20, 1)
			tooCloseToOtherObjs = False
			for i_objPlaced in range(i_item):
				xThisObj = objectList[i_objPlaced].GetTransform()[0,3]
				yThisObj = objectList[i_objPlaced].GetTransform()[1,3]
				distanceToThisObj = sqrt((posInX - xThisObj)**2 + (posInY - yThisObj)**2)
				if objectList[i_item].GetName() == 'kitchen_knife' or objectList[i_item].GetName() == 'fork' or objectList[i_item].GetName() == 'kitchen_knife0' or objectList[i_item].GetName() == 'fork0':
					if distanceToThisObj < distanceThresholdForkKnife:
						tooCloseToOtherObjs = True
				else: 
					if distanceToThisObj < distanceThreshold:
						tooCloseToOtherObjs = True
			if tooCloseToOtherObjs == True:
				continue
			zRotDegree = np.random.uniform(0, 360, 1)
			position = numpy.zeros([4,4])
			# pdb.set_trace() #eakpoint
			# newTransform = objectList[i_item][0].GetTransform()
			position[0:3,3] = [posInX, posInY, posInZ]
			zRotRadian = zRotDegree/180*pi
			objectTransformNew = giveRotationMatrix3D_4X4('z',zRotRadian) + position
			if size(shape(objectList)) > 1:
				objectList[i_item][0].SetTransform(objectTransformNew)
			else:
				objectList[i_item].SetTransform(objectTransformNew)
			# pdb.set_trace() #eakpoint

#			newTransform[0:3,3] = [posInX, posInY, posInZ]

#			objectList[i_item][0].SetTransform(newTransform)
			if size(shape(objectList)) > 1:
				thereIsCollision = checkCollision(objectList[i_item][0], env)
			else:
				thereIsCollision = checkCollision(objectList[i_item], env)
		cfgObjList.append([posInX, posInY, zRotDegree])
	# pdb.set_trace() #breakpoint
	return cfgObjList
#--------------------------------------------------------------------------------------------------------------
# check if the chosen object has collision with other objects on the table
def checkCollision(obj, env):
	isThereCollision = False
	for i_item in range(shape(env.GetBodies())[0]):
		#someCollision = env.CheckCollision(env.GetBodies()[i_item],obj)
		isThereCollision = isThereCollision or env.CheckCollision(env.GetBodies()[i_item],obj)
		#report1 = CollisionReport()
		#if env.CheckCollision(env.GetBodies()[i_item],obj):
		#	print str(env.GetBodies()[i_item].GetName()) + 'and' + str(obj.GetName())

	return isThereCollision
#-------------------------------------------------------------------------------------------------------------

def giveGraspPosOriFromObj(obj, handle, env):
	#pdb.set_trace() #breakpoint
	if obj[1] == 'oliveOilBottle':
		PosOriOld = obj[0].GetTransform() 
		offset_grasp = 0.085
		sideOffset_grasp = 0.06
		PosOriGrasp1 = dot(giveRotationMatrix3D_4X4('x', pi/2), giveRotationMatrix3D_4X4('y', pi/2)) + array([[0,0,0,-offset_grasp],[0,0,0,sideOffset_grasp],[0,0,0,0.15],[0,0,0,0]]) + (PosOriOld - np.eye(4))
		PosOriGrasp1a = dot(giveRotationMatrix3D_4X4('x', pi/2), giveRotationMatrix3D_4X4('y', pi/2)) + array([[0,0,0,-offset_grasp],[0,0,0, -sideOffset_grasp],[0,0,0,0.15],[0,0,0,0]]) + (PosOriOld - np.eye(4))
		"""
		PosOriGrasp1a = dot(giveRotationMatrix3D_4X4('y', pi/2),giveRotationMatrix3D_4X4('x', -pi/2)) + array([[0,0,0,0],[0,0,0,-offset_grasp],[0,0,0,0.15],[0,0,0,0]]) + (PosOriOld - np.eye(4))		
		PosOriGrasp1b = dot(giveRotationMatrix3D_4X4('y', pi),giveRotationMatrix3D_4X4('x', -pi/2)) + array([[0,0,0,0],[0,0,0,-offset_grasp],[0,0,0,0.15],[0,0,0,0]]) + (PosOriOld - np.eye(4))
		PosOriGrasp1c = dot(giveRotationMatrix3D_4X4('y', pi*3/2),giveRotationMatrix3D_4X4('x', -pi/2)) + array([[0,0,0,0],[0,0,0,-offset_grasp],[0,0,0,0.15],[0,0,0,0]]) + (PosOriOld - np.eye(4))
		PosOriGrasp2 = giveRotationMatrix3D_4X4('x', pi/2) + array([[0,0,0,0],[0,0,0,offset_grasp],[0,0,0,0.15],[0,0,0,0]]) + (PosOriOld - np.eye(4))
		PosOriGrasp2a = dot(giveRotationMatrix3D_4X4('y', pi/2), giveRotationMatrix3D_4X4('x', pi/2))+ array([[0,0,0,0],[0,0,0,offset_grasp],[0,0,0,0.15],[0,0,0,0]]) + (PosOriOld - np.eye(4))
		PosOriGrasp2b =  dot(giveRotationMatrix3D_4X4('y', pi), giveRotationMatrix3D_4X4('x', pi/2))+ array([[0,0,0,0],[0,0,0,offset_grasp],[0,0,0,0.15],[0,0,0,0]]) + (PosOriOld - np.eye(4))
		PosOriGrasp2c = dot(giveRotationMatrix3D_4X4('y', pi*3/2), giveRotationMatrix3D_4X4('x', pi/2))+ array([[0,0,0,0],[0,0,0,offset_grasp],[0,0,0,0.15],[0,0,0,0]]) + (PosOriOld - np.eye(4))
		PosOriGrasp3 = giveRotationMatrix3D_4X4('y', -pi/2) + array([[0,0,0,offset_grasp],[0,0,0,0],[0,0,0,0.15],[0,0,0,0]]) + (PosOriOld - np.eye(4))
		PosOriGrasp3a = dot(giveRotationMatrix3D_4X4('x',pi/2),giveRotationMatrix3D_4X4('y', -pi/2))+ array([[0,0,0,offset_grasp],[0,0,0,0],[0,0,0,0.15],[0,0,0,0]]) + (PosOriOld - np.eye(4))
		PosOriGrasp3b = dot(giveRotationMatrix3D_4X4('x',pi),giveRotationMatrix3D_4X4('y', -pi/2))+ array([[0,0,0,offset_grasp],[0,0,0,0],[0,0,0,0.15],[0,0,0,0]]) + (PosOriOld - np.eye(4))
		PosOriGrasp3c = dot(giveRotationMatrix3D_4X4('x',pi*3/2),giveRotationMatrix3D_4X4('y', -pi/2))+ array([[0,0,0,offset_grasp],[0,0,0,0],[0,0,0,0.15],[0,0,0,0]]) + (PosOriOld - np.eye(4))
		PosOriGrasp4 = giveRotationMatrix3D_4X4('y', pi/2) + array([[0,0,0,-offset_grasp],[0,0,0,0],[0,0,0,0.15],[0,0,0,0]]) + (PosOriOld - np.eye(4))
		PosOriGrasp4a = dot(giveRotationMatrix3D_4X4('x',pi/2), giveRotationMatrix3D_4X4('y', pi/2))+ array([[0,0,0,-offset_grasp],[0,0,0,0],[0,0,0,0.15],[0,0,0,0]]) + (PosOriOld - np.eye(4))
		PosOriGrasp4b = dot(giveRotationMatrix3D_4X4('x',pi), giveRotationMatrix3D_4X4('y', pi/2))+ array([[0,0,0,-offset_grasp],[0,0,0,0],[0,0,0,0.15],[0,0,0,0]]) + (PosOriOld - np.eye(4))
		PosOriGrasp4c = dot(giveRotationMatrix3D_4X4('x',pi*3/2), giveRotationMatrix3D_4X4('y', pi/2))+ array([[0,0,0,-offset_grasp],[0,0,0,0],[0,0,0,0.15],[0,0,0,0]]) + (PosOriOld - np.eye(4))
"""
		
		handle.append(PlotFrame(env, PosOriGrasp1, 0.01))
		handle.append(PlotFrame(env, PosOriGrasp1a, 0.01))

		pdb.set_trace() #breakpoint
		"""
		handle.append(PlotFrame(env, PosOriGrasp1b, 0.01))
		handle.append(PlotFrame(env, PosOriGrasp1c, 0.01))
		handle.append(PlotFrame(env, PosOriGrasp2, 0.01))
		handle.append(PlotFrame(env, PosOriGrasp2a, 0.01))
		handle.append(PlotFrame(env, PosOriGrasp2b, 0.01))
		handle.append(PlotFrame(env, PosOriGrasp2c, 0.01))
		handle.append(PlotFrame(env, PosOriGrasp3, 0.01))
		handle.append(PlotFrame(env, PosOriGrasp3a, 0.01))
		handle.append(PlotFrame(env, PosOriGrasp3b, 0.01))
		handle.append(PlotFrame(env, PosOriGrasp3c, 0.01))
		handle.append(PlotFrame(env, PosOriGrasp4, 0.01))
		handle.append(PlotFrame(env, PosOriGrasp4a, 0.01))
		handle.append(PlotFrame(env, PosOriGrasp4b, 0.01))
		handle.append(PlotFrame(env, PosOriGrasp4c, 0.01))
		"""
		return [PosOriGrasp1]
		"""
		, PosOriGrasp1a, PosOriGrasp1b, PosOriGrasp1c, \
		PosOriGrasp2, PosOriGrasp2a, PosOriGrasp2b, PosOriGrasp2c, \
		PosOriGrasp3, PosOriGrasp3a, PosOriGrasp3b, PosOriGrasp3c, \
		PosOriGrasp4, PosOriGrasp4a, PosOriGrasp4b, PosOriGrasp4c]
		"""
		
	elif obj[1] == 'glass':
		PosOriOld = obj[0].GetTransform() 
		offset_grasp = 0.2
		PosOriGrasp1 = giveRotationMatrix3D_4X4('x', -pi/2) + array([[0,0,0,0],[0,0,0,-offset_grasp],[0,0,0,0.1],[0,0,0,0]]) + (PosOriOld - np.eye(4))
		PosOriGrasp1a = dot(giveRotationMatrix3D_4X4('y', pi/2),giveRotationMatrix3D_4X4('x', -pi/2)) + array([[0,0,0,0],[0,0,0,-offset_grasp],[0,0,0,0.1],[0,0,0,0]]) + (PosOriOld - np.eye(4))
		PosOriGrasp1b = dot(giveRotationMatrix3D_4X4('y', pi),giveRotationMatrix3D_4X4('x', -pi/2)) + array([[0,0,0,0],[0,0,0,-offset_grasp],[0,0,0,0.1],[0,0,0,0]]) + (PosOriOld - np.eye(4))
		PosOriGrasp1c = dot(giveRotationMatrix3D_4X4('y', pi*3/2),giveRotationMatrix3D_4X4('x', -pi/2)) + array([[0,0,0,0],[0,0,0,-offset_grasp],[0,0,0,0.1],[0,0,0,0]]) + (PosOriOld - np.eye(4))
		PosOriGrasp2 = giveRotationMatrix3D_4X4('x', pi/2) + array([[0,0,0,0],[0,0,0,offset_grasp],[0,0,0,0.1],[0,0,0,0]]) + (PosOriOld - np.eye(4))
		PosOriGrasp2a = dot(giveRotationMatrix3D_4X4('y', pi/2), giveRotationMatrix3D_4X4('x', pi/2))+ array([[0,0,0,0],[0,0,0,offset_grasp],[0,0,0,0.1],[0,0,0,0]]) + (PosOriOld - np.eye(4))
		PosOriGrasp2b =  dot(giveRotationMatrix3D_4X4('y', pi), giveRotationMatrix3D_4X4('x', pi/2))+ array([[0,0,0,0],[0,0,0,offset_grasp],[0,0,0,0.1],[0,0,0,0]]) + (PosOriOld - np.eye(4))
		PosOriGrasp2c = dot(giveRotationMatrix3D_4X4('y', pi*3/2), giveRotationMatrix3D_4X4('x', pi/2))+ array([[0,0,0,0],[0,0,0,offset_grasp],[0,0,0,0.1],[0,0,0,0]])+ (PosOriOld - np.eye(4))
		PosOriGrasp3 = giveRotationMatrix3D_4X4('y', -pi/2) + array([[0,0,0,offset_grasp],[0,0,0,0],[0,0,0,0.1],[0,0,0,0]])+ (PosOriOld - np.eye(4))
		PosOriGrasp3a = dot(giveRotationMatrix3D_4X4('x',pi/2),giveRotationMatrix3D_4X4('y', -pi/2))+ array([[0,0,0,offset_grasp],[0,0,0,0],[0,0,0,0.1],[0,0,0,0]]) + (PosOriOld - np.eye(4))
		PosOriGrasp3b = dot(giveRotationMatrix3D_4X4('x',pi),giveRotationMatrix3D_4X4('y', -pi/2))+array([[0,0,0,offset_grasp],[0,0,0,0],[0,0,0,0.1],[0,0,0,0]]) + (PosOriOld - np.eye(4))
		PosOriGrasp3c = dot(giveRotationMatrix3D_4X4('x',pi*3/2),giveRotationMatrix3D_4X4('y', -pi/2))+ array([[0,0,0,offset_grasp],[0,0,0,0],[0,0,0,0.1],[0,0,0,0]]) + (PosOriOld - np.eye(4))
		PosOriGrasp4 = giveRotationMatrix3D_4X4('y', pi/2) + array([[0,0,0,-offset_grasp],[0,0,0,0],[0,0,0,0.1],[0,0,0,0]])+ (PosOriOld - np.eye(4))
		PosOriGrasp4a = dot(giveRotationMatrix3D_4X4('x',pi/2), giveRotationMatrix3D_4X4('y', pi/2))+ array([[0,0,0,-offset_grasp],[0,0,0,0],[0,0,0,0.1],[0,0,0,0]]) + (PosOriOld - np.eye(4))
		PosOriGrasp4b = dot(giveRotationMatrix3D_4X4('x',pi), giveRotationMatrix3D_4X4('y', pi/2))+ array([[0,0,0,-offset_grasp],[0,0,0,0],[0,0,0,0.1],[0,0,0,0]]) + (PosOriOld - np.eye(4))
		PosOriGrasp4c = dot(giveRotationMatrix3D_4X4('x',pi*3/2), giveRotationMatrix3D_4X4('y', pi/2))+ array([[0,0,0,-offset_grasp],[0,0,0,0],[0,0,0,0.1],[0,0,0,0]]) + (PosOriOld - np.eye(4))

		handle.append(PlotFrame(env, PosOriGrasp1, 0.01))
		handle.append(PlotFrame(env, PosOriGrasp1a, 0.01))
		handle.append(PlotFrame(env, PosOriGrasp1b, 0.01))
		handle.append(PlotFrame(env, PosOriGrasp1c, 0.01))
		handle.append(PlotFrame(env, PosOriGrasp2, 0.01))
		handle.append(PlotFrame(env, PosOriGrasp2a, 0.01))
		handle.append(PlotFrame(env, PosOriGrasp2b, 0.01))
		handle.append(PlotFrame(env, PosOriGrasp2c, 0.01))
		handle.append(PlotFrame(env, PosOriGrasp3, 0.01))
		handle.append(PlotFrame(env, PosOriGrasp3a, 0.01))
		handle.append(PlotFrame(env, PosOriGrasp3b, 0.01))
		handle.append(PlotFrame(env, PosOriGrasp3c, 0.01))
		handle.append(PlotFrame(env, PosOriGrasp4, 0.01))
		handle.append(PlotFrame(env, PosOriGrasp4a, 0.01))
		handle.append(PlotFrame(env, PosOriGrasp4b, 0.01))
		handle.append(PlotFrame(env, PosOriGrasp4c, 0.01))
		return [PosOriGrasp1, PosOriGrasp1a, PosOriGrasp1b, PosOriGrasp1c, \
		PosOriGrasp2, PosOriGrasp2a, PosOriGrasp2b, PosOriGrasp2c, \
		PosOriGrasp3, PosOriGrasp3a, PosOriGrasp3b, PosOriGrasp3c, \
		PosOriGrasp4, PosOriGrasp4a, PosOriGrasp4b, PosOriGrasp4c]
	elif obj[1] == 'cup':
		PosOriOld = obj[0].GetTransform() 
		offset_grasp = 0.2
		PosOriGrasp1 = giveRotationMatrix3D_4X4('x', -pi/2) + array([[0,0,0,0],[0,0,0,-offset_grasp],[0,0,0,0.06],[0,0,0,0]]) + (PosOriOld - np.eye(4))
		PosOriGrasp1a = dot(giveRotationMatrix3D_4X4('y', pi/2),giveRotationMatrix3D_4X4('x', -pi/2)) + array([[0,0,0,0],[0,0,0,-offset_grasp],[0,0,0,0.06],[0,0,0,0]])  + (PosOriOld - np.eye(4))
		PosOriGrasp1b = dot(giveRotationMatrix3D_4X4('y', pi),giveRotationMatrix3D_4X4('x', -pi/2)) + array([[0,0,0,0],[0,0,0,-offset_grasp],[0,0,0,0.06],[0,0,0,0]])  + (PosOriOld - np.eye(4))
		PosOriGrasp1c = dot(giveRotationMatrix3D_4X4('y', pi*3/2),giveRotationMatrix3D_4X4('x', -pi/2)) + array([[0,0,0,0],[0,0,0,-offset_grasp],[0,0,0,0.06],[0,0,0,0]])  + (PosOriOld - np.eye(4))
		PosOriGrasp2 = giveRotationMatrix3D_4X4('x', pi/2) + array([[0,0,0,0],[0,0,0,offset_grasp*1.2],[0,0,0,0.06],[0,0,0,0]])+ (PosOriOld - np.eye(4))
		PosOriGrasp2a = dot(giveRotationMatrix3D_4X4('y', pi/2), giveRotationMatrix3D_4X4('x', pi/2))+array([[0,0,0,0],[0,0,0,offset_grasp*1.2],[0,0,0,0.06],[0,0,0,0]]) + (PosOriOld - np.eye(4))
		PosOriGrasp2b =  dot(giveRotationMatrix3D_4X4('y', pi), giveRotationMatrix3D_4X4('x', pi/2))+ array([[0,0,0,0],[0,0,0,offset_grasp*1.2],[0,0,0,0.06],[0,0,0,0]]) + (PosOriOld - np.eye(4))
		PosOriGrasp2c = dot(giveRotationMatrix3D_4X4('y', pi*3/2), giveRotationMatrix3D_4X4('x', pi/2))+ array([[0,0,0,0],[0,0,0,offset_grasp*1.2],[0,0,0,0.06],[0,0,0,0]])+ (PosOriOld - np.eye(4))
		PosOriGrasp3 = giveRotationMatrix3D_4X4('y', -pi/2) + array([[0,0,0,offset_grasp],[0,0,0,0],[0,0,0,0.06],[0,0,0,0]]) + (PosOriOld - np.eye(4))
		PosOriGrasp3a = dot(giveRotationMatrix3D_4X4('x',pi/2),giveRotationMatrix3D_4X4('y', -pi/2))+ array([[0,0,0,offset_grasp],[0,0,0,0],[0,0,0,0.06],[0,0,0,0]]) + (PosOriOld - np.eye(4))
		PosOriGrasp3b = dot(giveRotationMatrix3D_4X4('x',pi),giveRotationMatrix3D_4X4('y', -pi/2))+array([[0,0,0,offset_grasp],[0,0,0,0],[0,0,0,0.06],[0,0,0,0]]) + (PosOriOld - np.eye(4))
		PosOriGrasp3c = dot(giveRotationMatrix3D_4X4('x',pi*3/2),giveRotationMatrix3D_4X4('y', -pi/2))+ array([[0,0,0,offset_grasp],[0,0,0,0],[0,0,0,0.06],[0,0,0,0]]) + (PosOriOld - np.eye(4))
		PosOriGrasp4 = giveRotationMatrix3D_4X4('y', pi/2) + array([[0,0,0,-offset_grasp],[0,0,0,0],[0,0,0,0.06],[0,0,0,0]])+ (PosOriOld - np.eye(4))
		PosOriGrasp4a = dot(giveRotationMatrix3D_4X4('x',pi/2), giveRotationMatrix3D_4X4('y', pi/2))+ array([[0,0,0,-offset_grasp],[0,0,0,0],[0,0,0,0.06],[0,0,0,0]]) + (PosOriOld - np.eye(4))
		PosOriGrasp4b = dot(giveRotationMatrix3D_4X4('x',pi), giveRotationMatrix3D_4X4('y', pi/2))+ array([[0,0,0,-offset_grasp],[0,0,0,0],[0,0,0,0.06],[0,0,0,0]]) + (PosOriOld - np.eye(4))
		PosOriGrasp4c = dot(giveRotationMatrix3D_4X4('x',pi*3/2), giveRotationMatrix3D_4X4('y', pi/2))+ array([[0,0,0,-offset_grasp],[0,0,0,0],[0,0,0,0.06],[0,0,0,0]]) + (PosOriOld - np.eye(4))

		handle.append(PlotFrame(env, PosOriGrasp1, 0.01))
		handle.append(PlotFrame(env, PosOriGrasp1a, 0.01))
		handle.append(PlotFrame(env, PosOriGrasp1b, 0.01))
		handle.append(PlotFrame(env, PosOriGrasp1c, 0.01))
		handle.append(PlotFrame(env, PosOriGrasp2, 0.01))
		handle.append(PlotFrame(env, PosOriGrasp2a, 0.01))
		handle.append(PlotFrame(env, PosOriGrasp2b, 0.01))
		handle.append(PlotFrame(env, PosOriGrasp2c, 0.01))
		handle.append(PlotFrame(env, PosOriGrasp3, 0.01))
		handle.append(PlotFrame(env, PosOriGrasp3a, 0.01))
		handle.append(PlotFrame(env, PosOriGrasp3b, 0.01))
		handle.append(PlotFrame(env, PosOriGrasp3c, 0.01))
		handle.append(PlotFrame(env, PosOriGrasp4, 0.01))
		handle.append(PlotFrame(env, PosOriGrasp4a, 0.01))
		handle.append(PlotFrame(env, PosOriGrasp4b, 0.01))
		handle.append(PlotFrame(env, PosOriGrasp4c, 0.01))

		return [PosOriGrasp1, PosOriGrasp1a, PosOriGrasp1b, PosOriGrasp1c, \
		PosOriGrasp2, PosOriGrasp2a, PosOriGrasp2b, PosOriGrasp2c, \
		PosOriGrasp3, PosOriGrasp3a, PosOriGrasp3b, PosOriGrasp3c, \
		PosOriGrasp4, PosOriGrasp4a, PosOriGrasp4b, PosOriGrasp4c]
	elif obj[1] == 'knife':
		PosOriOld = obj[0].GetTransform() 
		offset_grasp = 0.2
		PosOriGrasp1 = giveRotationMatrix3D_4X4('y', -pi/2) + array([[0,0,0,offset_grasp],[0,0,0,0],[0,0,0,0.003],[0,0,0,1]]) + (PosOriOld - np.eye(4))
		PosOriGrasp1a = dot(giveRotationMatrix3D_4X4('x',pi/2), giveRotationMatrix3D_4X4('y', -pi/2))+ array([[0,0,0,offset_grasp],[0,0,0,0],[0,0,0,0.003],[0,0,0,1]]) + (PosOriOld - np.eye(4))
		PosOriGrasp1b = dot(giveRotationMatrix3D_4X4('x',pi), giveRotationMatrix3D_4X4('y', -pi/2))+ array([[0,0,0,offset_grasp],[0,0,0,0],[0,0,0,0.003],[0,0,0,1]]) + (PosOriOld - np.eye(4))
		PosOriGrasp1c = dot(giveRotationMatrix3D_4X4('x',pi*3/2), giveRotationMatrix3D_4X4('y', -pi/2))+ array([[0,0,0,offset_grasp],[0,0,0,0],[0,0,0,0.003],[0,0,0,1]]) + (PosOriOld - np.eye(4))
		PosOriGrasp2 = giveRotationMatrix3D_4X4('y', pi/2) + array([[0,0,0,-offset_grasp],[0,0,0,0],[0,0,0,0.003],[0,0,0,0]]) + (PosOriOld - np.eye(4))
		PosOriGrasp2a = dot(giveRotationMatrix3D_4X4('x',pi/2), giveRotationMatrix3D_4X4('y', pi/2)) +  + array([[0,0,0,-offset_grasp],[0,0,0,0],[0,0,0,0.003],[0,0,0,0]]) + (PosOriOld - np.eye(4))
		PosOriGrasp2b = dot(giveRotationMatrix3D_4X4('x',pi), giveRotationMatrix3D_4X4('y', pi/2)) +  + array([[0,0,0,-offset_grasp],[0,0,0,0],[0,0,0,0.003],[0,0,0,0]]) + (PosOriOld - np.eye(4))
		PosOriGrasp2c = dot(giveRotationMatrix3D_4X4('x',pi*3/2), giveRotationMatrix3D_4X4('y', pi/2)) +  + array([[0,0,0,-offset_grasp],[0,0,0,0],[0,0,0,0.003],[0,0,0,0]]) + (PosOriOld - np.eye(4))
		PosOriGrasp3 = giveRotationMatrix3D_4X4('y', pi) + array([[0,0,0,0],[0,0,0,0],[0,0,0,offset_grasp],[0,0,0,0]]) + (PosOriOld - np.eye(4))
		PosOriGrasp3a = dot(giveRotationMatrix3D_4X4('z',pi/2),giveRotationMatrix3D_4X4('y', pi)) + array([[0,0,0,0],[0,0,0,0],[0,0,0,offset_grasp],[0,0,0,0]]) + (PosOriOld - np.eye(4))
		PosOriGrasp3b = dot(giveRotationMatrix3D_4X4('z',pi),giveRotationMatrix3D_4X4('y', pi)) + array([[0,0,0,0],[0,0,0,0],[0,0,0,offset_grasp],[0,0,0,0]]) + (PosOriOld - np.eye(4))
		PosOriGrasp3c = dot(giveRotationMatrix3D_4X4('z',pi*3/2),giveRotationMatrix3D_4X4('y', pi)) + array([[0,0,0,0],[0,0,0,0],[0,0,0,offset_grasp],[0,0,0,0]]) + (PosOriOld - np.eye(4))
		#PosOriGrasp4 = giveRotationMatrix3D_4X4('y', 0) + array([[0,0,0,0],[0,0,0,0],[0,0,0,-offset_grasp],[0,0,0,0]]) + (PosOriOld - np.eye(4))
		#PosOriGrasp4a = dot(giveRotationMatrix3D_4X4('z',pi/2),giveRotationMatrix3D_4X4('y', 0) ) + array([[0,0,0,0],[0,0,0,0],[0,0,0,-offset_grasp],[0,0,0,0]]) + (PosOriOld - np.eye(4))
		#PosOriGrasp4b = dot(giveRotationMatrix3D_4X4('z',pi),giveRotationMatrix3D_4X4('y', 0) ) + array([[0,0,0,0],[0,0,0,0],[0,0,0,-offset_grasp],[0,0,0,0]]) + (PosOriOld - np.eye(4))
		#PosOriGrasp4c = dot(giveRotationMatrix3D_4X4('z',pi*3/2),giveRotationMatrix3D_4X4('y', 0) ) + array([[0,0,0,0],[0,0,0,0],[0,0,0,-offset_grasp],[0,0,0,0]]) + (PosOriOld - np.eye(4))
		handle.append(PlotFrame(env, PosOriGrasp1, 0.01))
		handle.append(PlotFrame(env, PosOriGrasp1a, 0.01))
		handle.append(PlotFrame(env, PosOriGrasp1b, 0.01))
		handle.append(PlotFrame(env, PosOriGrasp1c, 0.01))
		handle.append(PlotFrame(env, PosOriGrasp2, 0.01))
		handle.append(PlotFrame(env, PosOriGrasp2a, 0.01))
		handle.append(PlotFrame(env, PosOriGrasp2b, 0.01))
		handle.append(PlotFrame(env, PosOriGrasp2c, 0.01))
		handle.append(PlotFrame(env, PosOriGrasp3, 0.01))
		handle.append(PlotFrame(env, PosOriGrasp3a, 0.01))
		handle.append(PlotFrame(env, PosOriGrasp3b, 0.01))
		handle.append(PlotFrame(env, PosOriGrasp3c, 0.01))
		#handle.append(PlotFrame(env, PosOriGrasp4, 0.01))
		#handle.append(PlotFrame(env, PosOriGrasp4a, 0.01))
		#handle.append(PlotFrame(env, PosOriGrasp4b, 0.01))
		#handle.append(PlotFrame(env, PosOriGrasp4c, 0.01))
		return [PosOriGrasp1, PosOriGrasp1a, PosOriGrasp1b, PosOriGrasp1c, \
		PosOriGrasp2, PosOriGrasp2a, PosOriGrasp2b, PosOriGrasp2c, \
		PosOriGrasp3, PosOriGrasp3a, PosOriGrasp3b, PosOriGrasp3c]
		#, \		PosOriGrasp4, PosOriGrasp4a, PosOriGrasp4b, PosOriGrasp4c]


def canReachAnyObjectOnTable(robot, objectList, env):
	manip = robot.SetActiveManipulator("lwr")
	origDOF = robot.GetDOFValues(manip.GetArmIndices())
	ikmodel = databases.inversekinematics.InverseKinematicsModel(robot=robot, freeindices=[4], iktype=IkParameterizationType.Transform6D)
	if not ikmodel.load():
 		ikmodel.autogenerate()
	numObjects = shape(objectList)[0]
	# reachAnyOne = False
	for i_object in range(numObjects):
		objectName = objectList[i_object][1]
		for i_grasp in range(shape(objectList[i_object][2])[0]):
			# pdb.set_trace() #breakpoint
			graspTransform = objectList[i_object][2][i_grasp]
			# sol_DOF = manip.FindIKSolution(graspTransform,IkFilterOptions.IgnoreJointLimits)
			sol_DOF = manip.FindIKSolution(graspTransform,IkFilterOptions.CheckEnvCollisions)
			if type(sol_DOF) is NoneType:
				continue 
			else:
				robot.SetDOFValues(sol_DOF, manip.GetArmIndices())
				# pdb.set_trace() #breakpoint
				if checkCollision(robot, env):
					continue
				else:				
					robot.SetDOFValues(sol_DOF, manip.GetArmIndices())
					raw_input('press enter to next position')
					robot.SetDOFValues(origDOF, manip.GetArmIndices())
					# env.UpdatePublishedBodies()
					return True
	robot.SetDOFValues(origDOF, manip.GetArmIndices())
	return False

def giveRotationMatrix3D_4X4(axis, radians):
	if axis == 'x':
		return array([[1,0,0,0],[0,cos(radians),-sin(radians),0],[0,sin(radians),cos(radians),0],[0,0,0,1]])
	elif axis == 'y':
		return array([[cos(radians),0,sin(radians),0],[0,1,0,0],[-sin(radians),0,cos(radians),0],[0,0,0,1]])
	elif axis == 'z':
		return array([[cos(radians),-sin(radians),0,0],[sin(radians),cos(radians),0,0],[0,0,1,0],[0,0,0,1]])

def drawPathRangeToObj(robot, objPosXY, robotRangeMax, env):
	rangeHandle = []
	robPosX = robot.GetTransform()[0,3]
	robPosY = robot.GetTransform()[1,3]
	radiusAroundObj = robotRangeMax * 1.2
	distObjToRob = sqrt( (objPosXY[0] - robPosX)**2 + (objPosXY[1] - robPosY)**2 )
	lengthPlane = sqrt( (distObjToRob)**2 - (radiusAroundObj)**2 )
	slopeObjToRob = (objPosXY[1] - robPosY) / (objPosXY[0] - robPosX)
	angleDeg = np.rad2deg(np.arctan(slopeObjToRob))
	if (objPosXY[1] > robPosY) and (angleDeg < 0):
		angleDeg = angleDeg + 180
	elif (objPosXY[1] < robPosY) and (angleDeg > 0):
		angleDeg = angleDeg - 180
	halfPathAngleDeg = np.rad2deg( math.acos(( distObjToRob**2 + lengthPlane**2 - radiusAroundObj **2 ) / (2 * distObjToRob * lengthPlane)))
	endPlanePosX1 = 1.2 * lengthPlane * math.cos(np.deg2rad(angleDeg + halfPathAngleDeg)) + robPosX
	endPlanePosY1 = 1.2 * lengthPlane * math.sin(np.deg2rad(angleDeg + halfPathAngleDeg)) + robPosY
	endPlanePosX2 = 1.2 * lengthPlane * math.cos(np.deg2rad(angleDeg - halfPathAngleDeg)) + robPosX
	endPlanePosY2 = 1.2 * lengthPlane * math.sin(np.deg2rad(angleDeg - halfPathAngleDeg)) + robPosY
	# raw_input('Draw the area around the object, where the robot can reach the object theoretically.')
	# raw_input('Draw the path range.')
	rangeHandle.append(env.drawtrimesh(points=array(((robPosX,robPosY,0),(robPosX,robPosY,1.5),(endPlanePosX1,endPlanePosY1,0),(endPlanePosX1,endPlanePosY1,1.5))),indices=array(((0,1,2),(2,1,3)),int64),colors=array((1,0,0,0.5))))
	rangeHandle.append(env.drawtrimesh(points=array(((robPosX,robPosY,0),(robPosX,robPosY,1.5),(endPlanePosX2,endPlanePosY2,0),(endPlanePosX2,endPlanePosY2,1.5))),indices=array(((0,1,2),(2,1,3)),int64),colors=array((1,0,0,0.5))))
	return rangeHandle


def generateMovablePointAreaAroundAnObj(center, radius, granularityIndex, robot, env, typeChosen,  pointsType = 'allAround'):
	# pdb.set_trace() #breakpoint
	granularity = 2 * radius / granularityIndex
	#pdb.set_trace() #breakpoint
	pointNearestToOrig = center - array([radius, radius])
	pointsInGrid = []
	pointsHandle = []
	robotPosOri = robot.GetTransform()
	robotPosX = robot.GetTransform()[0,3]
	robotPosY = robot.GetTransform()[1,3]
	distObjToRob = sqrt((center[0] - robotPosX)**2 + (center[1] - robotPosY)**2)
	with env:
		for i_x in range(granularityIndex):
			for i_y in range(granularityIndex):
				pointInLoopX = pointNearestToOrig[0] + i_x * granularity
				pointInLoopY = pointNearestToOrig[1] + i_y * granularity
				pointInLoop = pointNearestToOrig + array([i_x * granularity, i_y * granularity])
				distPointToObj = sqrt((pointInLoopX - center[0])**2 + (pointInLoopY - center[1])**2)
				if distPointToObj > radius: # the point is outside the circle, not considered
					continue
				distPointToRob = sqrt((pointInLoopX - robotPosX)**2 + (pointInLoopY - robotPosY)**2)
				if distPointToRob > distObjToRob and typeChosen == 'object' and pointsType == 'nearRob': # the point is too far, not considered
					continue
				# pdb.set_trace() #breakpoint
				setRobPos = robot.GetTransform()
				setRobPos[0:2,3] = array([pointInLoopX, pointInLoopY])
				robot.SetTransform(setRobPos)
				if checkCollision(robot, env):
					continue
				if typeChosen == 'object':
					pointsHandle.append(env.plot3(points=array([pointInLoopX, pointInLoopY, 0.05]), pointsize=0.05, colors=array(((0,1,0))),drawstyle = 1))
				elif typeChosen == 'point':
					pointsHandle.append(env.plot3(points=array([pointInLoopX, pointInLoopY, 0.05]), pointsize=0.03, colors=array(((0,1,0))),drawstyle = 1))				
				pointsInGrid.append(array([pointInLoopX, pointInLoopY]))
		robot.SetTransform(robotPosOri)
		# pdb.set_trace() #breakpoint
	return	pointsInGrid, pointsHandle
			
def findPointWithHighManipXY(robot, givenPointsMovable, obj, env, plotheight, plotsSize):
	# pdb.set_trace() #breakpoint
	manip = robot.SetActiveManipulator("lwr")
	manip_tool = manip.GetEndEffector()
	# origDOF = robot.GetDOFValues(manip.GetArmIndices())
	#ikmodel = databases.inversekinematics.InverseKinematicsModel(robot=robot, freeindices=[4], iktype=IkParameterizationType.Transform6D)
	objPosXY = obj.GetTransform()[0:2,3]
	objPos = obj.GetTransform()[0:3,3]
	pointsWithManipXY = np.zeros((shape(givenPointsMovable)[0],3))
	pointsManipulabilityHandle = []
	for i_point in range(shape(givenPointsMovable)[0]):
		this_point = givenPointsMovable[i_point]
		pointsWithManipXY[i_point, 0] = givenPointsMovable[i_point][0]
		pointsWithManipXY[i_point, 1] = givenPointsMovable[i_point][1]
		origPosXYRob = robot.GetTransform()[0:2,3]
		origDOFManip = robot.GetDOFValues(manip.GetArmIndices())
		newPosRob = robot.GetTransform()
		newPosRob[0:2,3] = this_point
		robot.SetTransform(newPosRob)
		# checkCollision(robot, env)
		allGraspPosOri = giveAllPossibleGraspOfAnObj(obj, robot)

		
		"""
		##
		aPlot = []
		for i in range(shape(allGraspPosOri)[0]): 
			pdb.set_trace() #breakpoint
			aPlot.append(PlotFrame(env, allGraspPosOri[i], 0.01))
		pdb.set_trace() #breakpoint
		##
		"""
		nearGraspPosOri = []
		distObjToRob = sqrt((this_point[1] - objPosXY[1])**2 + (this_point[0] - objPosXY[0])**2)
		for i_grasp in range(shape(allGraspPosOri)[0]):
			this_grasp = allGraspPosOri[i_grasp]
			distGraspToRob = sqrt((this_point[1] - this_grasp[1,3])**2 + (this_point[0] - this_grasp[0,3])**2)
			if distGraspToRob < distObjToRob:
				nearGraspPosOri.append(this_grasp)
		if shape(nearGraspPosOri)[0] == 0:
			pointsWithManipXY[i_point, 2] = max(0, pointsWithManipXY[i_point, 2])
			continue
		for i_nearGrasp in range(shape(nearGraspPosOri)[0]):
			# pdb.set_trace() #breakpoint
			sol_DOF = manip.FindIKSolution(nearGraspPosOri[i_nearGrasp],IkFilterOptions.CheckEnvCollisions)
			if type(sol_DOF) is NoneType:
				pointsWithManipXY[i_point, 2] = max(0, pointsWithManipXY[i_point, 2])
				continue
			else:
				robot.SetDOFValues(sol_DOF, manip.GetArmIndices())
				jacobAllDOF = robot.CalculateActiveJacobian(manip_tool.GetIndex(),manip_tool.GetTransform()[0:3,3])[0:3][0:6]
				jacobXY = np.matrix(array([jacobAllDOF[0][0:7], jacobAllDOF[1][0:7]])) # jacobian in X, Y translational direction
				pointsWithManipXY[i_point, 2] = max(sqrt(np.linalg.det(np.dot(jacobXY, jacobXY.T))),pointsWithManipXY[i_point, 2])
		# print "till now, the highest manipulability is " + str(manipulabilityInXYPoint)
		# print "and at point: " + str(pointWithHighManipXY[0]) + " " +  str(pointWithHighManipXY[1])
		# pdb.set_trace() #breakpoint
		newPosRob[0:2,3] = origPosXYRob
		robot.SetTransform(newPosRob)
		robot.SetDOFValues(origDOFManip, manip.GetArmIndices())
	# pdb.set_trace() #breakpoint
	manipulabilityXY = max(pointsWithManipXY[:,2])
	indexPointsWithManipXY = [i for i, j in enumerate(pointsWithManipXY[:,2]) if j == manipulabilityXY]
	pointsManipulabilityHandle = plotMonoContourOfManip(pointswithManip = pointsWithManipXY, env=env, height = plotheight, plotsSize = plotsSize)
	xOfPointsWithHighestManipXY = pointsWithManipXY[indexPointsWithManipXY[0],0]
	yOfPointsWithHighestManipXY = pointsWithManipXY[indexPointsWithManipXY[0],1]
	# pdb.set_trace() #breakpoint
	pointsManipulabilityHandle.append(env.plot3(points=array(((xOfPointsWithHighestManipXY,yOfPointsWithHighestManipXY,plotheight- 0.015))),pointsize=plotsSize*1.2,colors=array(((1,1,1))), drawstyle = 1))
	return pointsWithManipXY[indexPointsWithManipXY[0],0:2], pointsManipulabilityHandle
	
	

def giveAllPossibleGraspOfAnObj(obj, robot):
	allPossibleGrasp = []
	robotPosXY = robot.GetTransform()[0:2,3]
	if obj.GetName() == 'kitchen_olive_oil_bottle':
		backOffset_grasp = 0.085
		sideOffset_grasp = 0.07
		PosOriOld = obj.GetTransform() 
		PosOriGrasp1 = dot(giveRotationMatrix3D_4X4('x', pi/2), giveRotationMatrix3D_4X4('y', pi/2)) + array([[0,0,0,-backOffset_grasp],[0,0,0,sideOffset_grasp],[0,0,0,0.15],[0,0,0,0]]) + (PosOriOld - np.eye(4))
		PosOriGrasp1a = dot(giveRotationMatrix3D_4X4('x', pi), dot(giveRotationMatrix3D_4X4('x', pi/2), giveRotationMatrix3D_4X4('y', pi/2))) + array([[0,0,0,-backOffset_grasp],[0,0,0, -sideOffset_grasp],[0,0,0,0.15],[0,0,0,0]]) + (PosOriOld - np.eye(4))
		PosOriGrasp2 = dot(giveRotationMatrix3D_4X4('y', pi), dot(giveRotationMatrix3D_4X4('x', -pi/2), giveRotationMatrix3D_4X4('z', pi))) + array([[0,0,0,sideOffset_grasp],[0,0,0,-backOffset_grasp],[0,0,0,0.15],[0,0,0,0]]) + (PosOriOld - np.eye(4))
		PosOriGrasp2a = dot(giveRotationMatrix3D_4X4('x', -pi/2), giveRotationMatrix3D_4X4('z', pi)) + array([[0,0,0,-sideOffset_grasp],[0,0,0,-backOffset_grasp],[0,0,0,0.15],[0,0,0,0]]) + (PosOriOld - np.eye(4))
		PosOriGrasp3 = dot(giveRotationMatrix3D_4X4('y', -pi/2), giveRotationMatrix3D_4X4('z', 3*pi/2)) + array([[0,0,0,backOffset_grasp],[0,0,0,-sideOffset_grasp],[0,0,0,0.15],[0,0,0,0]]) + (PosOriOld - np.eye(4))
		PosOriGrasp3a = dot(giveRotationMatrix3D_4X4('x', pi), dot(giveRotationMatrix3D_4X4('y', -pi/2), giveRotationMatrix3D_4X4('z', 3*pi/2))) + array([[0,0,0,backOffset_grasp],[0,0,0,sideOffset_grasp],[0,0,0,0.15],[0,0,0,0]]) + (PosOriOld - np.eye(4))
		PosOriGrasp4 = giveRotationMatrix3D_4X4('x', pi/2) + array([[0,0,0,sideOffset_grasp],[0,0,0,backOffset_grasp],[0,0,0,0.15],[0,0,0,0]]) + (PosOriOld - np.eye(4))
		PosOriGrasp4a = dot(giveRotationMatrix3D_4X4('y', pi), giveRotationMatrix3D_4X4('x', pi/2)) + array([[0,0,0,-sideOffset_grasp],[0,0,0,backOffset_grasp],[0,0,0,0.15],[0,0,0,0]]) + (PosOriOld - np.eye(4))
	allPossibleGrasp = [PosOriGrasp1, PosOriGrasp1a, PosOriGrasp2, PosOriGrasp2a, PosOriGrasp3, PosOriGrasp3a, PosOriGrasp4, PosOriGrasp4a]
	return allPossibleGrasp

def slowlyMoveToAPoint(obj, destination, segmentOfRoute, stepNumber, env, drawOrNotDraw, withOrWithoutTimeSleep):
	objTransf = obj.GetTransform()
	objPosOri = obj.GetTransform()[0:2,3]
	diff = destination - objPosOri
	diffToGo = segmentOfRoute * diff
	incrementStep = diffToGo/stepNumber
	if drawOrNotDraw is 'draw':	
		pathPointHandle = []
		pathPointHandle.append(env.drawlinestrip(points=array(((objPosOri[0],objPosOri[1],0.01),(destination[0],destination[1],0))), linewidth=1.0, colors=array(((0.81,0.12,0.56),(0.81,0.12,0.56)))))
	for i in range(stepNumber):
		# pdb.set_trace() #breakpoint
		if drawOrNotDraw is 'draw':
			pathPointHandle.append(env.plot3(points=array([obj.GetTransform()[0,3],obj.GetTransform()[1,3],0.05]), pointsize=0.05, colors=array(((0.58, 0, 0.83))),drawstyle = 1))
		objPosStep = obj.GetTransform()[0:2,3] + incrementStep
		objTransf[0:2,3] = objPosStep
		obj.SetTransform(objTransf)
		if withOrWithoutTimeSleep is 'with':
			time.sleep(0.2)
	if drawOrNotDraw is 'draw':	
		return pathPointHandle

def findGraspWithHighManipZ(robot, givenRobPos, obj, env):
	robTransf = robot.GetTransform()
	robTransf[0:2,3] = givenRobPos
	robot.SetTransform(robTransf)
	manip = robot.SetActiveManipulator("lwr")
	manip_tool = manip.GetEndEffector()
	origDOFManip = robot.GetDOFValues(manip.GetArmIndices())
	objPosXY = obj.GetTransform()[0:2,3]
	# objPos = obj.GetTransform()[0:3,3]
	grapsWithHighManipZ = []
	DOFManipWithHighManipZ = []
	manipulabilityZGrasp = 0
	graspManipulabilityHandle = []
	allGraspPosOri = giveAllPossibleGraspOfAnObj(obj, robot)
	nearGraspPosOri = []
	distObjToRob = sqrt((givenRobPos[1] - objPosXY[1])**2 + (givenRobPos[0] - objPosXY[0])**2)
	for i_grasp in range(shape(allGraspPosOri)[0]):
		this_grasp = allGraspPosOri[i_grasp]
		distGraspToRob = sqrt((givenRobPos[1] - this_grasp[1,3])**2 + (givenRobPos[0] - this_grasp[0,3])**2)
		if distGraspToRob < distObjToRob:
			nearGraspPosOri.append(this_grasp)
	manipulabilityZGraspMax = 0
	if shape(nearGraspPosOri)[0] == 0:
		print "Nowhere to grasp? Seriously?"
		pdb.set_trace() #breakpoint
	for i_nearGrasp in range(shape(nearGraspPosOri)[0]):
		# graspManipulabilityHandle.append(PlotFrame(env, nearGraspPosOri[i_nearGrasp], 0.03))
		# pdb.set_trace() #breakpoint
		sol_DOF = manip.FindIKSolution(nearGraspPosOri[i_nearGrasp],IkFilterOptions.CheckEnvCollisions)
		if type(sol_DOF) is NoneType:
			continue
		else:
			# pdb.set_trace() #breakpoint
			robot.SetDOFValues(sol_DOF, manip.GetArmIndices())
			jacobAllDOF = robot.CalculateActiveJacobian(manip_tool.GetIndex(),manip_tool.GetTransform()[0:3,3])[0:3][0:6]
			jacobZ = np.matrix(jacobAllDOF[2][0:7]) # jacobian in Z translational direction
			manipulabilityZGrasp = sqrt(np.linalg.det(np.dot(jacobZ, jacobZ.T)))
			if  manipulabilityZGrasp > manipulabilityZGraspMax:
				manipulabilityZGraspMax = manipulabilityZGrasp
				grapsWithHighManipZ = nearGraspPosOri[i_nearGrasp]
				DOFManipWithHighManipZ = sol_DOF
	robot.SetDOFValues(origDOFManip, manip.GetArmIndices())
	if manipulabilityZGraspMax == 0: 
		print "Cannot grasp this object here without collision."		
	return grapsWithHighManipZ, DOFManipWithHighManipZ, graspManipulabilityHandle

def slowlyMoveManipToAnObj(robot, solDOF, env):
	manip = robot.SetActiveManipulator("lwr")
	DOFManipCurrent = robot.GetActiveDOFValues()[0:7]
	diffDOF = solDOF-DOFManipCurrent
	while LA.norm(diffDOF) > 0.01:
		DOFStep = DOFManipCurrent + diffDOF * 0.1
  		robot.SetDOFValues(DOFStep, manip.GetArmIndices())
  		time.sleep(0.1)
  		DOFManipCurrent = robot.GetActiveDOFValues()[0:7]
  		diffDOF = solDOF - DOFManipCurrent
  		if LA.norm(diffDOF) < 0.01:
  			robot.SetDOFValues(solDOF, manip.GetArmIndices())
  		if checkCollision(robot, env):
  			print "Stop! There is a collision!"
  			pdb.set_trace() #breakpoint

def slowlyMoveManipToAnObjWithManipulability(robot, solDOF, env, drawTimes):
	manip = robot.SetActiveManipulator("lwr")
	manip_tool = manip.GetEndEffector()
	DOFManipCurrent = robot.GetActiveDOFValues()[0:7]
	increaseDOF = (solDOF-DOFManipCurrent)/drawTimes
	drawtype = 'ellipsoid'
	# pdb.set_trace() #breakpoint
	# creates some virtual robots for plotting
	newrobots = []
	for ind in range(drawTimes-1):
		newrobot = RaveCreateRobot(env,robot.GetXMLId())
		newrobot.Clone(robot,0)
		# env.Add(newrobot,True)
		for link in newrobot.GetLinks():
			for geom in link.GetGeometries():
				geom.SetTransparency(0.7)
		newrobots.append(newrobot)
	# pdb.set_trace() #breakpoint
	manipEllipsoidHandle = []
	for i_draw in range(drawTimes):
		if i_draw == (drawTimes-1):
			pdb.set_trace() #breakpoint
		DOFStep = increaseDOF + DOFManipCurrent
		if i_draw != (drawTimes-1):
			env.Add(newrobots[i_draw],True)
			manipDraw = newrobots[i_draw].SetActiveManipulator("lwr")
			manipDraw_tool = manipDraw.GetEndEffector()
			newrobots[i_draw].SetDOFValues(DOFStep, manipDraw.GetArmIndices())
			manipEllipsoidHandle.append(drawManipXYZ(newrobots[i_draw], manipDraw_tool, env, drawtype))
			DOFManipCurrent = newrobots[i_draw].GetActiveDOFValues()[0:7]
		else:
			robot.SetDOFValues(DOFStep, manip.GetArmIndices())
			for link in robot.GetLinks():
				for geom in link.GetGeometries():
					geom.SetTransparency(0)
			manipEllipsoidHandle.append(drawManipXYZ(robot,manip_tool,env, drawtype))
			DOFManipCurrent = robot.GetActiveDOFValues()[0:7]
		time.sleep(0.1)
  		if checkCollision(robot, env):
  			print "Stop! There is a collision!"
  			pdb.set_trace() #breakpoint

def slowlyGraspAnObjForDemo(robot, env):
	diff19 = -100
	diff8_12_16 = 32
	diff9_13_17 = 45
	diff10_14_18 = 50
	diff22 = 28
	for i_increm in range(11):
		robot.SetDOFValues([radians(i_increm * diff19/10.0)],[19])
		robot.SetDOFValues([radians(i_increm * diff8_12_16/10.0)],[8])
		robot.SetDOFValues([radians(i_increm * diff9_13_17/10.0)],[9])
		robot.SetDOFValues([radians(i_increm * diff10_14_18/10.0)],[10])
		robot.SetDOFValues([radians(i_increm * diff8_12_16/10.0)],[12])
		robot.SetDOFValues([radians(i_increm * diff9_13_17/10.0)],[13])
		robot.SetDOFValues([radians(i_increm * diff10_14_18/10.0)],[14])
		robot.SetDOFValues([radians(i_increm * diff8_12_16/10.0)],[16])
		robot.SetDOFValues([radians(i_increm * diff9_13_17/10.0)],[17])
		robot.SetDOFValues([radians(i_increm * diff10_14_18/10.0)],[18])
		robot.SetDOFValues([radians(i_increm * diff22/10.0)],[22])
		time.sleep(0.2)


"""
	robot.SetDOFValues([radians(-100)],[19])

	robot.SetDOFValues([radians(32)],[8])
	robot.SetDOFValues([radians(45)],[9])
	robot.SetDOFValues([radians(50)],[10])

	robot.SetDOFValues([radians(32)],[12])
	robot.SetDOFValues([radians(45)],[13])
	robot.SetDOFValues([radians(50)],[14])

	robot.SetDOFValues([radians(32)],[16])
	robot.SetDOFValues([radians(45)],[17])
	robot.SetDOFValues([radians(50)],[18])

	robot.SetDOFValues([radians(0)],[21])
	robot.SetDOFValues([radians(28)],[22])
"""

def closeFingerExecuteGrasp(robot, objToGraspIndex, objIndexRemainingExceptObjInFocus, objIndexNotRemaining, objectList, env):
	initDOF = robot.GetDOFValues()
	objToGrasp = objectList[objToGraspIndex]
	maxJointValue = robot.GetDOFLimits()[1][:]
	maxJointValue[19] = robot.GetDOFLimits()[0][19]
	robot.SetDOFValues([radians(-100)],[20])
	controlJointsLevel1 = [8, 12, 16, 19]
	controlJointsLevel2 = [9, 13, 17, 21]
	if objToGrasp.GetName() == 'fork' or objToGrasp.GetName() == 'kitchen_knife':

		controlJointsLevel3 = [10, 14, 18]
		# robot.SetDOFValues([radians(30)],[21])
		robot.SetDOFValues([radians(80)],[22])
	else:
		controlJointsLevel3 = [10, 14, 18, 22]
	controlJointsLevelTotal = [controlJointsLevel1, controlJointsLevel2, controlJointsLevel3]
	movableVecLevel1 = [True, True, True, True]
	movableVecLevel2 = [True, True, True, True]
	if objToGrasp.GetName() == 'fork' or objToGrasp.GetName() == 'kitchen_knife':
		
		movableVecLevel3 = [True, True, True, False]
	else:
		movableVecLevel3 = [True, True, True, True]
	
	movableVecLevelTotal = [movableVecLevel1, movableVecLevel2, movableVecLevel3]
	while(True):
		# time.sleep(0.5)
		if np.any(movableVecLevel1):
			currentLevelIndex = 0
		elif np.any(movableVecLevel2):
			currentLevelIndex = 1
		elif np.any(movableVecLevel3):
			currentLevelIndex = 2
		# else:
		# 	pdb.set_trace() # breakpoint
		for i_joint in range(shape(controlJointsLevelTotal[currentLevelIndex])[0]):
			for i_increm in range(10):
				robot.SetDOFValues([i_increm * maxJointValue[controlJointsLevelTotal[currentLevelIndex][i_joint]] /10.0],[controlJointsLevelTotal[currentLevelIndex][i_joint]])
				# pdb.set_trace() #breakpoint
				objIndexCllsnVecNotRemaining = checkCollisionWithIndexList(robot, objectList, objIndexNotRemaining, env)
				if objIndexCllsnVecNotRemaining != []: # collision with other objects
					robot.SetDOFValues(initDOF)
					return ['cllsnNotRemaining', objIndexCllsnVecNotRemaining]
				if (objToGrasp.GetName() != 'kitchen_knife') and (objToGrasp.GetName() != 'fork'): # not consider the collision with desk if grasping the knife or fork
					if env.CheckCollision(robot, env.GetBodies()[2]):
						robot.SetDOFValues(initDOF)
						return ['cllsnTable', []]
				objIndexCllsnVecRemainingExceptObjInFocus = checkCollisionWithIndexList(robot, objectList, objIndexRemainingExceptObjInFocus, env)
				if objIndexCllsnVecRemainingExceptObjInFocus != []: # collision with other objects
					robot.SetDOFValues(initDOF)
					return ['cllsnRemainingExceptObjInFocus', objIndexCllsnVecRemainingExceptObjInFocus]
				if env.CheckCollision(robot, objToGrasp): # collision with object to be grasped
					robot.SetDOFValues([(i_increm-1) * maxJointValue[controlJointsLevelTotal[currentLevelIndex][i_joint]] /10.0],[controlJointsLevelTotal[currentLevelIndex][i_joint]])
					movableVecLevelTotal[currentLevelIndex][i_joint] = False
					break
				if i_increm == 9:
					movableVecLevelTotal[currentLevelIndex][i_joint] = False
		# pdb.set_trace() # breakpoint
		if not np.any(movableVecLevelTotal):
			break
	return ['grasped',[]]

def executeCloseFingerGrasp(robot, objToGraspIndex, objectList, env):
	initDOF = robot.GetDOFValues()
	objToGrasp = objectList[objToGraspIndex]
	maxJointValue = robot.GetDOFLimits()[1][:]
	maxJointValue[19] = robot.GetDOFLimits()[0][19]
	robot.SetDOFValues([radians(-100)],[20])
	controlJointsLevel1 = [8, 12, 16, 19]
	controlJointsLevel2 = [9, 13, 17, 21]
	if objToGrasp.GetName() == 'fork' or objToGrasp.GetName() == 'kitchen_knife':

		controlJointsLevel3 = [10, 14, 18]
		# robot.SetDOFValues([radians(30)],[21])
		robot.SetDOFValues([radians(80)],[22])
	else:
		controlJointsLevel3 = [10, 14, 18, 22]
	controlJointsLevelTotal = [controlJointsLevel1, controlJointsLevel2, controlJointsLevel3]
	movableVecLevel1 = [True, True, True, True]
	movableVecLevel2 = [True, True, True, True]
	if objToGrasp.GetName() == 'fork' or objToGrasp.GetName() == 'kitchen_knife':
		
		movableVecLevel3 = [True, True, True, False]
	else:
		movableVecLevel3 = [True, True, True, True]
	
	movableVecLevelTotal = [movableVecLevel1, movableVecLevel2, movableVecLevel3]
	while(True):
		time.sleep(0.5)
		if np.any(movableVecLevel1):
			currentLevelIndex = 0
		elif np.any(movableVecLevel2):
			currentLevelIndex = 1
		elif np.any(movableVecLevel3):
			currentLevelIndex = 2
		# else:
		# pdb.set_trace() # breakpoint
		for i_joint in range(shape(controlJointsLevelTotal[currentLevelIndex])[0]):
			for i_increm in range(10):
				robot.SetDOFValues([i_increm * maxJointValue[controlJointsLevelTotal[currentLevelIndex][i_joint]] /10.0],[controlJointsLevelTotal[currentLevelIndex][i_joint]])
				# pdb.set_trace() #breakpoint
				if env.CheckCollision(robot, objToGrasp): # collision with object to be grasped
					robot.SetDOFValues([(i_increm-1) * maxJointValue[controlJointsLevelTotal[currentLevelIndex][i_joint]] /10.0],[controlJointsLevelTotal[currentLevelIndex][i_joint]])
					movableVecLevelTotal[currentLevelIndex][i_joint] = False
					break
				if i_increm == 9:
					movableVecLevelTotal[currentLevelIndex][i_joint] = False
		# pdb.set_trace() # breakpoint
		if not np.any(movableVecLevelTotal):
			break


def openFingerAfterGrasp(robot, objInFocusIndex, objIndexRemainingExceptObjInFocus, objIndexNotRemaining, objectList, initDOF, env):
	objToGrasp = objectList[objInFocusIndex]
	crrtDOFTotal = robot.GetDOFValues()
	crrtDOF = robot.GetDOFValues()[8:23]
	diffDOF = initDOF[8:23] - crrtDOF
	incrementDOF = diffDOF/10.0
	for i_increm in range(11):
		newDOF = crrtDOF + i_increm * incrementDOF
		robot.SetDOFValues(newDOF, range(8,23))
		objRobotIndexCllsnVecNotRemaining = checkCollisionWithIndexList(robot, objectList, objIndexNotRemaining, env)
		objObjIndexCllsnVecNotRemaining = checkCollisionWithIndexList(objToGrasp, objectList, objIndexNotRemaining, env)
		objIndexCllsnVecNotRemaining = objRobotIndexCllsnVecNotRemaining + list(set(objObjIndexCllsnVecNotRemaining) - set(objRobotIndexCllsnVecNotRemaining))
		if objIndexCllsnVecNotRemaining != []: # collision with other objects
			robot.SetDOFValues(initDOF)
			# pdb.set_trace() #breakpoint
			return ['cllsnNotRemaining', objIndexCllsnVecNotRemaining]
		if (objectList[objInFocusIndex].GetName() != 'kitchen_knife') and (objectList[objInFocusIndex].GetName() != 'fork'): # not consider the collision with desk if grasping the knife or fork
			if env.CheckCollision(robot, env.GetBodies()[2]):
				robot.SetDOFValues(initDOF)
				pdb.set_trace() #breakpoint
				return ['cllsnTable', []]
		objRobotIndexCllsnVecRemainingExceptObjInFocus = checkCollisionWithIndexList(robot, objectList, objIndexRemainingExceptObjInFocus, env)
		objObjIndexCllsnVecRemainingExceptObjInFocus = checkCollisionWithIndexList(objToGrasp, objectList, objIndexRemainingExceptObjInFocus, env)
		objIndexCllsnVecRemainingExceptObjInFocus = objRobotIndexCllsnVecRemainingExceptObjInFocus + list(set(objObjIndexCllsnVecRemainingExceptObjInFocus) - set(objRobotIndexCllsnVecRemainingExceptObjInFocus))
		if objIndexCllsnVecRemainingExceptObjInFocus != []: # collision with remaining objects
			robot.SetDOFValues(initDOF)
			# pdb.set_trace() #breakpoint
			return ['cllsnRemainingExceptObjInFocus', objIndexCllsnVecRemainingExceptObjInFocus]
		# if env.CheckCollision(robot, objToGrasp): # collision with object grasped
			# pdb.set_trace() #breakpoint
	robot.SetDOFValues(crrtDOFTotal)
	return ['released',[]]


def executeOpenFingerAfterGrasp(robot, initDOF, env):
	crrtDOFTotal = robot.GetDOFValues()
	crrtDOF = robot.GetDOFValues()[8:23]
	diffDOF = initDOF - crrtDOF
	incrementDOF = diffDOF/10.0
	for i_increm in range(11):
		newDOF = crrtDOF + i_increm * incrementDOF
		robot.SetDOFValues(newDOF, range(8,23))
		time.sleep(0.1)


def	plotMonoContourOfManip(pointswithManip, env, height, plotsSize):
	pointsManipulabilityHandle = []
	manipulability = pointswithManip[:,2]
	manipulabilityMax = max(manipulability)
	manipulabilityMin = min(manipulability)
	manipulabilityDef = manipulabilityMax - manipulabilityMin
	manipulabilityRelativ = (manipulability - manipulabilityMin)/manipulabilityDef
	for i_point in range(shape(pointswithManip)[0]):
		x = round(pointswithManip[i_point,0],3)
		y = round(pointswithManip[i_point,1],3)
		# pdb.set_trace() #breakpoint
		manipulabilityRelativPoint = round(manipulabilityRelativ[i_point],3)
		if manipulabilityRelativPoint< 0.5:
			R = 0
			G = manipulabilityRelativPoint/0.5
			B = (0.5-manipulabilityRelativPoint)/0.5
		else:
			R = (manipulabilityRelativPoint-0.5)/0.5
			G = (1-manipulabilityRelativPoint)/0.5
			B = 0
		# pdb.set_trace() #breakpoint
		pointsManipulabilityHandle.append(env.plot3(points=array(((x,y,height))),pointsize=plotsSize,colors=array(((R,G,B))), drawstyle = 1))
	return pointsManipulabilityHandle

def drawEVEVManipEllipTool(robot,manip_tool,env):
	jacobAllDOF = robot.CalculateActiveJacobian(manip_tool.GetIndex(),manip_tool.GetTransform()[0:3,3])[0:3][0:6]
	jacobXYZ = np.matrix(array([jacobAllDOF[0][0:7], jacobAllDOF[1][0:7], jacobAllDOF[1][0:7]]))
	U_XYZ, sigma_XYZ, Vh_XYZ = linalg.svd(jacobXYZ)
	EV1_XYZ = array([U_XYZ[0,0],U_XYZ[1,0],U_XYZ[2,0]])
	EV2_XYZ = array([U_XYZ[0,1],U_XYZ[1,1],U_XYZ[2,1]])
	EV3_XYZ = array([U_XYZ[0,2],U_XYZ[1,2],U_XYZ[2,2]])
	manipTool_Pos = manip_tool.GetTransform()[0:3,3]
	EV1_EndPos = manipTool_Pos + EV1_XYZ * sigma_XYZ[0]
	EV2_EndPos = manipTool_Pos + EV2_XYZ * sigma_XYZ[1]
	EV3_EndPos = manipTool_Pos + EV3_XYZ * sigma_XYZ[2]
	EVHandle = []
	if sigma_XYZ[0] > 1e-3:
		EVHandle.append(env.drawarrow(p1=manipTool_Pos,p2=EV1_EndPos,linewidth=0.03,color=[1,0,0]))
	if sigma_XYZ[1] > 1e-3:
		EVHandle.append(env.drawarrow(p1=manipTool_Pos,p2=EV2_EndPos,linewidth=0.03,color=[0,1,0]))
	if sigma_XYZ[2] > 1e-3:
		EVHandle.append(env.drawarrow(p1=manipTool_Pos,p2=EV3_EndPos,linewidth=0.03,color=[0,0,1]))
	return EVHandle

def drawManipXYZ(robot,manip_tool,env,drawtype):
	manipTool_Pos = manip_tool.GetTransform()[0:3,3]
	jacobAllDOF = robot.CalculateActiveJacobian(manip_tool.GetIndex(),manip_tool.GetTransform()[0:3,3])[0:3][0:6]
	jacobX = np.matrix(jacobAllDOF[0][0:7]) # jacobian in X translational direction
	jacobY = np.matrix(jacobAllDOF[1][0:7]) # jacobian in Y translational direction
	jacobZ = np.matrix(jacobAllDOF[2][0:7]) # jacobian in Z translational direction
	manipulabilityX = sqrt(np.linalg.det(np.dot(jacobX, jacobX.T)))
	manipulabilityY = sqrt(np.linalg.det(np.dot(jacobY, jacobY.T)))
	manipulabilityZ = sqrt(np.linalg.det(np.dot(jacobZ, jacobZ.T)))
	ManipVecXEndPos = manipTool_Pos + [manipulabilityX, 0, 0]
	ManipVecYEndPos = manipTool_Pos + [0, manipulabilityY, 0]
	ManipVecZEndPos = manipTool_Pos + [0, 0, manipulabilityZ]
	ManipVecXYZHandle = []
	# pdb.set_trace() #breakpoint
	if drawtype == 'semiaxes':
		if manipulabilityX > 1e-3 :
			ManipVecXYZHandle.append(env.drawarrow(p1=manipTool_Pos,p2=ManipVecXEndPos,linewidth=0.03,color=[1,0,0]))
		if manipulabilityY > 1e-3 :
			ManipVecXYZHandle.append(env.drawarrow(p1=manipTool_Pos,p2=ManipVecYEndPos,linewidth=0.03,color=[0,1,0]))
		if manipulabilityZ > 1e-3 :
			ManipVecXYZHandle.append(env.drawarrow(p1=manipTool_Pos,p2=ManipVecZEndPos,linewidth=0.03,color=[0,0,1]))
	elif drawtype == 'ellipsoid':
		aEllipsoid = manipulabilityX/5
		bEllipsoid = manipulabilityY/5
		cEllipsoid = manipulabilityZ/5
		iX = 35
		iY = 35
		xEllipsoidArray = np.linspace(manipTool_Pos[0]-aEllipsoid,manipTool_Pos[0]+aEllipsoid,num=iX)
		yEllipsoidArray = np.linspace(manipTool_Pos[1]-bEllipsoid,manipTool_Pos[1]+bEllipsoid,num=iY)
		# zEllipsoidArray = np.linspace(manipTool_Pos[2]-manipulabilityZ,manipTool_Pos[2]+manipulabilityZ,num=10)
		for i_X in range(iX):
			xInLoop = xEllipsoidArray[i_X]
			for i_Y in range(iY):
				# pdb.set_trace() #breakpoint
				yInLoop = yEllipsoidArray[i_Y]
				if ((xInLoop-manipTool_Pos[0])**2/aEllipsoid**2 + (yInLoop-manipTool_Pos[1])**2/bEllipsoid**2>1):
					continue
				else:
					zInLoop =  sqrt(round(cEllipsoid**2 * (1-(xInLoop-manipTool_Pos[0])**2/aEllipsoid**2 - (yInLoop-manipTool_Pos[1])**2/bEllipsoid**2),5)) + manipTool_Pos[2]
					zInLoop2 = 2*manipTool_Pos[2] - zInLoop
					ManipVecXYZHandle.append(env.plot3(points=array([xInLoop,yInLoop,zInLoop]), pointsize=0.02, colors=array(((1, 1, 0, 0.1))),drawstyle = 1))
					ManipVecXYZHandle.append(env.plot3(points=array([xInLoop,yInLoop,zInLoop2]), pointsize=0.02, colors=array(((1, 1, 0, 0.1))),drawstyle = 1))
	return ManipVecXYZHandle



def possiblePlacement(samplePosition, covMatrix, pointNumber, env):

	meanDistr = [samplePosition[0], samplePosition[1]]
	plotHeight = samplePosition[2]
	xPoints, yPoints = np.random.multivariate_normal(meanDistr, covMatrix, pointNumber).T
	xPoints = xPoints.T
	yPoints = yPoints.T
	pointsWithPossibility = np.zeros((shape(xPoints)[0],3))
	possiblePlacementHandle = []
	for i_point in range(shape(pointsWithPossibility)[0]):
		pointsWithPossibility[i_point][0] = xPoints[i_point]
		pointsWithPossibility[i_point][1] = yPoints[i_point]
		xyPosition = [xPoints[i_point], yPoints[i_point]]
		pointsWithPossibility[i_point][2] = computeBivariateGaussionPossib(xyPosition, meanDistr, covMatrix)
		if pointsWithPossibility[i_point][2]< 0.5:
			R = 0
			G = pointsWithPossibility[i_point][2]/0.5
			B = (0.5-pointsWithPossibility[i_point][2])/0.5
		else:
			R = (pointsWithPossibility[i_point][2]-0.5)/0.5
			G = (1-pointsWithPossibility[i_point][2])/0.5
			B = 0
		possiblePlacementHandle.append(env.plot3(points=array(((xPoints[i_point],yPoints[i_point],plotHeight))),pointsize=0.01,colors=array(((R,G,B))), drawstyle = 1))
	return possiblePlacementHandle

def computeBivariateGaussionPossib(point, mean, covMatrix):
	stepA = 1/ sqrt((2*pi)**2*np.linalg.det(covMatrix))
	stepB = np.dot(np.subtract(point,mean).T, inv(covMatrix))
	stepC = (-1)/2 * np.dot(stepB, np.subtract(point,mean))
	return stepA * exp(stepC)

def rotateAroundAnAxisNormalToGround(obj, posAxis, radian):
	posObj = obj.GetTransform()
	posObj[0:3,3] = posObj[0:3,3] - posAxis
	rotationMatrix = giveRotationMatrix3D_4X4('z', radian) 
	newPosObj = np.dot(rotationMatrix, posObj)
	newPosObj[0:3,3] = newPosObj[0:3,3] + posAxis
	obj.SetTransform(newPosObj)

def slowlyRotateTwoObjAroundAnAxisNormalToGround(obj1, obj2, posAxis, radian, env):
	stepNum = 10
	with env:
		for i_rot in range(stepNum):
			rotateAroundAnAxisNormalToGround(obj1, posAxis, radian/stepNum)
			rotateAroundAnAxisNormalToGround(obj2, posAxis, radian/stepNum)
			env.UpdatePublishedBodies()
			time.sleep(0.2)

def slowlyMoveTwoObjToAPoint(obj1, obj2, destination1_2D, destination2_2D, stepNum, env, drawOrNotDraw1, drawOrNotDraw2):
	# stepNum = 10
	objTransf1 = obj1.GetTransform()
	objPos1 = obj1.GetTransform()[0:2,3]
	diff1 = destination1_2D - objPos1
	diffStep1 = diff1/stepNum
	objTransf2 = obj2.GetTransform()
	objPos2 = obj2.GetTransform()[0:2,3]
	diff2 = destination2_2D - objPos2
	diffStep2 = diff2/stepNum
	with env:
		for i_move in range(stepNum):
			# pdb.set_trace() #breakpoint
			destination1_2D = objPos1 + (i_move + 1) * diffStep1
			destination2_2D = objPos2 + (i_move + 1) * diffStep2
			slowlyMoveToAPoint(obj1, destination1_2D, segmentOfRoute = 1, stepNumber = 1, env = env, drawOrNotDraw = drawOrNotDraw1, withOrWithoutTimeSleep = 'without')
			slowlyMoveToAPoint(obj2, destination2_2D, segmentOfRoute = 1, stepNumber = 1, env = env, drawOrNotDraw = drawOrNotDraw2, withOrWithoutTimeSleep = 'without')
			env.UpdatePublishedBodies()
			time.sleep(0.1)


def slowlyPlaceAnObjToAPosition(robot, obj, objNewTransform, stepNumber, env):
	objTransform = obj.GetTransform()
	manip = robot.SetActiveManipulator("lwr")
	manip_tool = manip.GetEndEffector()
	manipToolTransform = manip_tool.GetTransform()
	transformFromBottToTool = np.dot(inv(objTransform), manipToolTransform)
	manipToolNewTransform = np.dot(objNewTransform, transformFromBottToTool)
	origDOFManip = robot.GetDOFValues(manip.GetArmIndices())
	sol_DOF = manip.FindIKSolution(manipToolNewTransform,IkFilterOptions.CheckEnvCollisions)
	if type(sol_DOF) is NoneType:
		pdb.set_trace() #breakpoint
		print "Robot: I can not reach there!"
	diffDOF = sol_DOF - origDOFManip
	incremDOFStep = diffDOF/stepNumber
	with env:
		for i_place in range(stepNumber):
			newDOFInStep = origDOFManip + (i_place + 1) * incremDOFStep
			robot.SetDOFValues(newDOFInStep, manip.GetArmIndices())
			manipToolTransformInStep = manip_tool.GetTransform()
			objTransformInStep = np.dot(manipToolTransformInStep,inv(transformFromBottToTool))
			obj.SetTransform(objTransformInStep)
			env.UpdatePublishedBodies()
			time.sleep(0.2)

def slowlyReleaseGraspAnObjForDemo(robot, env):
	diff19 = -100
	diff8_12_16 = 32
	diff9_13_17 = 45
	diff10_14_18 = 50
	diff22 = 28
	for i_increm in range(11):
		robot.SetDOFValues([radians((10-i_increm) * diff19/10.0)],[19])
		robot.SetDOFValues([radians((10-i_increm)* diff8_12_16/10.0)],[8])
		robot.SetDOFValues([radians((10-i_increm)* diff9_13_17/10.0)],[9])
		robot.SetDOFValues([radians((10-i_increm)* diff10_14_18/10.0)],[10])
		robot.SetDOFValues([radians((10-i_increm)* diff8_12_16/10.0)],[12])
		robot.SetDOFValues([radians((10-i_increm)* diff9_13_17/10.0)],[13])
		robot.SetDOFValues([radians((10-i_increm)* diff10_14_18/10.0)],[14])
		robot.SetDOFValues([radians((10-i_increm)* diff8_12_16/10.0)],[16])
		robot.SetDOFValues([radians((10-i_increm)* diff9_13_17/10.0)],[17])
		robot.SetDOFValues([radians((10-i_increm)* diff10_14_18/10.0)],[18])
		robot.SetDOFValues([radians((10-i_increm)* diff22/10.0)],[22])
		time.sleep(0.1)
	
def slowlyReleaseMoveManipToAnObj(robot, env):
	manip = robot.SetActiveManipulator("lwr")
	DOFManipCurrent = robot.GetActiveDOFValues()[0:7]
	solDOF = np.zeros(shape(DOFManipCurrent))
	diffDOF = solDOF-DOFManipCurrent
	while LA.norm(diffDOF) > 0.01:
		DOFStep = DOFManipCurrent + diffDOF * 0.1
  		robot.SetDOFValues(DOFStep, manip.GetArmIndices())
  		time.sleep(0.1)
  		DOFManipCurrent = robot.GetActiveDOFValues()[0:7]
  		diffDOF = solDOF - DOFManipCurrent
  		if LA.norm(diffDOF) < 0.01:
  			robot.SetDOFValues(solDOF, manip.GetArmIndices())
  		if checkCollision(robot, env):
  			print "Stop! There is a collision!"
  			pdb.set_trace() #breakpoint
###############################################################################################################
# Yues algorithm
def initSeqAct(objectList, refObjectList, env):
	# pdb.set_trace() #breakpoint

	seqActions = np.zeros([shape(objectList)[0], 4])
	objCllsnStatus = np.zeros([shape(objectList)[0], 2]) # [object, Enumeration, number of collisions]
	for i_objOrig in range(shape(objectList)[0]):
		objOrig = objectList[i_objOrig]
		numCllsn = 0
		for i_objRef in range(shape(refObjectList)[0]):
			if env.CheckCollision(objectList[i_objOrig], refObjectList[i_objRef]):
				numCllsn = numCllsn + 1
		objCllsnStatus[i_objOrig] = [i_objOrig, numCllsn]
		# pdb.set_trace() #breakpoint
	objCllsnStatus = sorted(objCllsnStatus, key=lambda a_entry: a_entry[1], reverse=True) 
	for i_objOrig in range(shape(objectList)[0]):
		seqActions[i_objOrig][0] = objCllsnStatus[i_objOrig][0]
	seqActions = seqActions.astype(int)
	return seqActions
	# pdb.set_trace() #breakpoint

def checkSeqActionCollision(objectList, refObjectList, refCfgObjList, initCfgObjList, seqActions, onlyPushObjEnum, area, env):

	onlyPushObjEnumInObjList = onlyPushObjEnum[0]-1 # assume only one object must be pushed instead of grasped, e.g. the monitor
	initTransformObjList = giveTransformObjList(objectList)
	goalTransformObjList = giveTransformObjList(refObjectList)
	seqActSuccs = True
	stepNumberInSeq = 0
	for i_action in range(shape(seqActions)[0]):
		# action_Loop = seqActions[i_action]
		objToMove_Loop = objectList[seqActions[i_action][0]]
		moveTarget_Loop = seqActions[i_action][1]
		objCrrtForCllsn = []
		# objFinalForCllsn = []
		# pdb.set_trace() #breakpoint

		# move to where?
		if moveTarget_Loop == 0: # 0 = move object to goal configuration
			targetTransform = goalTransformObjList[seqActions[i_action][0]]
		else: # move object to random position
			targetTransform = objToMove_Loop.GetTransform()
			targetTransform[0,3] = seqActions[i_action][2] # randX
			targetTransform[1,3] = seqActions[i_action][3] # randY

			"""
		# only push or not?
		if seqActions[i_action][0] == onlyPushObjEnumInObjList: # if the object can only be pushed instead of grasped
			targetXY = targetTransform[0:2,3]
			crrtXY = objToMove_Loop.GetTransform()[0:2,3]
			diffXY = targetXY-crrtXY
			incrementXY = diffXY/10.0
			objCrrtCfgForCllsnArray = []
			objFinalCfgForCllsnArray = []
			for i_increm in range(10):
				objCrrtCfgForCllsnArray.append(checkCollisionWithList(objToMove_Loop, objectList))
				objFinalCfgForCllsnArray.append(checkCollisionWithList(objToMove_Loop, refObjectList))
				if i_increm < 9:
					transformAfterIncrem = objToMove_Loop.GetTransform()
					transformAfterIncrem[0:2,3] = transformAfterIncrem[0:2,3] + incrementXY
					objToMove_Loop.SetTransform(transformAfterIncrem)
			objCrrtForCllsn = findVecCllsnObjIndex(objCrrtCfgForCllsnArray)
			objFinalForCllsn = findVecCllsnObjIndex(objFinalCfgForCllsnArray)
			# ifObjCrrtCfgForCllsnEmpty = not objCrrtCfgForCllsn
			# ifObjFinalCfgForCllsnEmpty = not objFinalCfgForCllsn
			if objCrrtForCllsn or objFinalForCllsn: # collision along the pushing path
				seqActSuccs = False
				stepNumberInSeq = i_action
				return [seqActSuccs, stepNumberInSeq, objCrrtForCllsn, objFinalForCllsn]
		else: # the object can be grasped or pushed
			"""

		objToMove_Loop.SetTransform(targetTransform)
		objCrrtForCllsn = checkCollisionWithList(objToMove_Loop, objectList, env)
		if objCrrtForCllsn:
			seqActSuccs = False
			stepNumberInSeq = i_action
			break
	# pdb.set_trace() #breakpoint
	setTransformObjList(objectList, initTransformObjList)
	return [seqActSuccs, stepNumberInSeq, objCrrtForCllsn]

def	tryReorderSeqAct(seqActions, stepNumberInSeq, objCrrtForCllsn):
	for i_objCllsn in range(shape(objCrrtForCllsn)[0]):
		objCllsnIndex = objCrrtForCllsn[i_objCllsn]
		seqActions = reorderSeqActSingle(seqActions, objCllsnIndex, stepNumberInSeq)
	return seqActions

def checkIfInHistory(seqActions,seqActHist):
	# pdb.set_trace() #breakpoint
	if len(seqActHist) == len(seqActions):
		return False
	for i_hist in range(shape(seqActHist)[0]):
		seqActInHist = seqActHist[i_hist]
		shapeSeqActInHist = shape(seqActInHist)
		shapeSeqActions = shape(seqActions)
		corrThisSeqAct = False
		# pdb.set_trace() #breakpoint
		if shapeSeqActInHist == shapeSeqActions:
			corrMatrix = np.zeros(shape(seqActions))
			for i in range(shape(seqActions)[0]):
				for j in range(shape(seqActions)[1]):
					if seqActions[i][j] == seqActInHist[i][j]:
						corrMatrix[i][j] = 1
			if all(corrMatrix):
				corrThisSeqAct = True
				return corrThisSeqAct
	return corrThisSeqAct



def reorderSeqActSingle(seqActions, objCllsnIndex, stepNumberInSeq):
	for i_action in range(shape(seqActions)[0]):
		objToMoveInAction = seqActions[i_action][0]
		if objToMoveInAction == objCllsnIndex:
			objToMoveIndex = i_action
			# pdb.set_trace() #breakpoint

			vecInRoll = seqActions[stepNumberInSeq:objToMoveIndex+1]
			vecInRoll = np.roll(vecInRoll,1, axis=0)
			seqActions[stepNumberInSeq:objToMoveIndex+1] = vecInRoll
	"""
	oldAction = seqActions[stepNumberInSeq]
	seqActions[stepNumberInSeq] = seqActions[objToMoveIndex]
	for i_action in range(shape(seqActions)[0]):
		if i_action > stepNumberInSeq and i_action <= objToMoveIndex:
			newAction = seqActions[i_action]
			seqActions[i_action] = oldAction
	seqActions[stepNumberInSeq] = newAction
	"""
	#pdb.set_trace() #breakpoint
	return seqActions

def addActRmvCycCllsn(seqActions, stepNumberInSeq, objectList, refCfgObjList, area, env):
#	actionCycCllsnIndex = findActionCycCllsnIndex(seqActions, seqActHist)
	thereIsCollision = True
	newRowInSeqActions = np.zeros([1,4])
	objToMoveRand = objectList[seqActions[stepNumberInSeq+1][0]]
	objToMoveRandTransform = objToMoveRand.GetTransform()
	while(thereIsCollision):
		xRand = np.random.uniform(area[0] - area[2]/2, area[0] + area[2]/2, 1)
		yRand = np.random.uniform(area[1] - area[3]/2, area[1] + area[3]/2, 1)
		objToMoveRandTransform[0,3] = xRand
		objToMoveRandTransform[1,3] = yRand
		objToMoveRand.SetTransform(objToMoveRandTransform)
		thereIsCollisionCrrtCfgGrp = checkAnyCllsn(objToMoveRand, objectList, env)
		thereIsCollisionFinalCfgGrp = checkAnyCllsn(objToMoveRand, refCfgObjList, env)
		thereIsCollision = thereIsCollisionCrrtCfgGrp or thereIsCollisionFinalCfgGrp
	newRowInSeqActions = [seqActions[stepNumberInSeq+1][0], 1, xRand, yRand]
	seqActions.append(newRowInSeqActions)
	rollPartVec = seqActions[stepNumberInSeq:]
	rollPartVec = np.roll(rollPartVec, 1)
	seqActions[stepNumberInSeq:] = rollPartVec
	return seqActions


def moveAnimFromSeqActions(objectList, refObjectList, seqActions, initCfgObjList, refCfgObjList, env):
	crrtCfgObjList = initCfgObjList
	for i_seqAction in range(shape(seqActions)[0]):
		target = seqActions[i_seqAction][1]
		objectToMoveIndex = seqActions[i_seqAction][0]
		objectToMove = objectList[objectToMoveIndex]
		if target == 0: # to goal
			goalTransform = refObjectList[objectToMoveIndex].GetTransform()[0:2,3]
			# goalTransform = refObjectList[objectToMoveIndex].GetTransform()
		else: # to random
			goalTransform = refObjectList[objectToMoveIndex].GetTransform()
			goalTransform[0,3] = seqActions[i_seqAction][2]
			goalTransform[1,3] = seqActions[i_seqAction][3]
			# crrtCfgObjList
			# movement: lift
		# pdb.set_trace() #breakpoint
		for i_lift in range(10):
			zIncrem = 0.08
			transformObjList = objectToMove.GetTransform()
			transformObjList[2,3] = transformObjList[2,3] + zIncrem
			objectToMove.SetTransform(transformObjList)
		  	time.sleep(0.1)

		# pdb.set_trace() #breakpoint
		# movement: move in the air
		deltaX = (goalTransform[0] - objectToMove.GetTransform()[0,3])/10.0
		deltaY = (goalTransform[1] - objectToMove.GetTransform()[1,3])/10.0
		if target == 0: # to goal 
			deltaRotZ = (refCfgObjList[objectToMoveIndex][2] - crrtCfgObjList[objectToMoveIndex][2])/10.0

		for i_move in range(10):
			xTo = objectToMove.GetTransform()[0,3] + deltaX
			yTo = objectToMove.GetTransform()[1,3] + deltaY
			if target == 0: # to goal
				rotZTo =crrtCfgObjList[objectToMoveIndex][2] + deltaRotZ
			crrtCfgObjList[objectToMoveIndex][0] = xTo
			crrtCfgObjList[objectToMoveIndex][1] = yTo
			transformObjList = objectToMove.GetTransform()
			transformObjList[0,3] = xTo
			transformObjList[1,3] = yTo
			objectToMove.SetTransform(transformObjList)
			if target == 0: # to goal
				crrtCfgObjList[objectToMoveIndex][2] = rotZTo
				rotateObjectInAir(objectToMove, rotZTo)
		  	time.sleep(0.1)


		for i_lift in range(10):
			zIncrem = -0.08
			transformObjList = objectToMove.GetTransform()
			transformObjList[2,3] = transformObjList[2,3] + zIncrem
			objectToMove.SetTransform(transformObjList)
	  		time.sleep(0.1)

		
def	rotateObjectInAir(objectToMove, rotZTo):
	# objectTransformOld = np.eye(4)
	rotZToRadian = rotZTo/180 * pi
	position = numpy.zeros([4,4])
	position[0:3,3] = objectToMove.GetTransform()[0:3,3]
	# objectTransformOld[0:3,0:3] = self.objectInitRotMatrix[self.objectSelectedIndex]
	objectTransformNew = giveRotationMatrix3D_4X4('z', rotZToRadian) + position
	objectToMove.SetTransform(objectTransformNew)


# def findActionCycCllsnIndex(seqActions, seqActHist):
#	for i_actionHist in range(shape(seqActHist)[0]):
# 	return actionCycCllsnIndex
def seqActHistAppend(seqActions, seqActHist):
	interArray = np.zeros(shape(seqActions))
	for i in range(shape(seqActions)[0]):
		for j in range(shape(seqActions)[1]):
			if j == 0 or j == 1:
				interArray[i][j] = seqActions[i][j].astype(int)
			else:
				interArray[i][j] = seqActions[i][j]

	seqActHist.append(interArray)
	return seqActHist

def giveTransformObjList(objectList):
	transformObjList = []
	for i_object in range(shape(objectList)[0]):
		transformObjList.append(objectList[i_object].GetTransform())
# 	pdb.set_trace() #breakpoint
	return transformObjList

def setTransformObjList(objectList, transformObjList):
	for i_object in range(shape(objectList)[0]):
		objectList[i_object].SetTransform(transformObjList[i_object])

def checkCollisionWithList(obj, objectList, env):
	objIndexCllsnVec = []
	for i_object in range(shape(objectList)[0]):
		if env.CheckCollision(obj, objectList[i_object]):
			objIndexCllsnVec.append(i_object)
	return objIndexCllsnVec

def checkCollisionWithIndexList(obj, objectList, objectIndexList, env):
	objIndexCllsnVec = []
	for i_object in range(shape(objectIndexList)[0]):
		if env.CheckCollision(obj, objectList[objectIndexList[i_object]]):
			objIndexCllsnVec.append(objectIndexList[i_object])
	return objIndexCllsnVec

def findVecCllsnObjIndex(objForCllsnArray):
	objForCllsnVec = []
	for i_row in range(shape(objForCllsnArray)[0]):
		objForCllsnRow = objForCllsnArray[i_row]
		for i_value in range(shape(objForCllsnRow)[0]):
			value = objForCllsnRow[i_value]
			if value not in objForCllsnVec:
				objForCllsnVec.apppend(value)
	return objForCllsnVec

def checkAnyCllsn(singleObject, objectGroupe, env):
	thereIsCollisionGroupe = False
	for i_object in range(shape(objectGroupe)[0]):
		thereIsCollisionSingle = env.CheckCollision(singleObject, objectGroupe[i_object])
		thereIsCollisionGroupe = thereIsCollisionGroupe or thereIsCollisionSingle
	return thereIsCollisionGroupe

def findPathReArrangementPRM(reArrangeGraph, initCfgObjList, refCfgObjList):
	reArrangeGraphDijkstra = Graph()
	addReArrangementGraphToDijkstra(reArrangeGraphDijkstra, reArrangeGraph)
	dijkstra(reArrangeGraphDijkstra, reArrangeGraphDijkstra.get_vertex(0)) # define starting node
	target = reArrangeGraphDijkstra.get_vertex(1) # define target node
	pathUnsorted = [target.get_id()]
	shortest(target, pathUnsorted)
	path = pathUnsorted[::-1]
	# path = [6197, 1002, 6198]
	nodesIncludedInPath = shape(path)[0]
	if nodesIncludedInPath == 1: # there is no path found
		print "No path found"
		# pdb.set_trace() #breakpoint
		return 'null'
	return path

def addReArrangementGraphToDijkstra(reArrangeGraphDijkstra, reArrangeGraph):
	reArrangeNode = reArrangeGraph[0]
	numNode = shape(reArrangeNode)[0]
	reArrangeConnectivity = reArrangeGraph[1]
	reArrangeDistance = reArrangeGraph[2]
	for i_node in range(numNode):
		reArrangeGraphDijkstra.add_vertex(i_node)
	for i_node in range(numNode):
		for j_node in range(numNode):
			if i_node < j_node:
				if reArrangeConnectivity[i_node][j_node]:
					reArrangeGraphDijkstra.add_edge(i_node, j_node, reArrangeDistance[i_node][j_node])
	

	#if not reArrangeGraph[3]:
	return 'null'

def sampleArrangementPRM(initCfgObjList, refCfgObjList, objectList, area, env):
	sampleInitOrGoal = np.random.uniform(0,1,1)
	if sampleInitOrGoal > 0.95: # sample the initial configuration or goal configuration
		sampleInit = np.random.uniform(0,1,1)
		if sampleInit > 0.5: # sample the initial configuration
			return [initCfgObjList, 'initial']
		else: # sample the goal configuration
			return [refCfgObjList, 'goal']
	else: # randomly sample the configuration
		initTransformObjList = giveTransformObjList(objectList) # save initial transformation
		resetAllObjToOrigin(objectList)
		newCfgObjList = genrateRandomPositionObjectsInARectAreaWithZRot(area, objectList, env, 'random')
		setTransformObjList(objectList, initTransformObjList) # restore previous transformation
		return [newCfgObjList, 'random']

def addNodeToReArrangementPRMGraph(reArrangeGraph, newNode, nodeStatus, area):
	# pdb.set_trace() # breakpoint, make real object untransparent
	if nodeStatus == 'initial':
		return [reArrangeGraph, 0]
	elif nodeStatus == 'goal': # the newly sampled node is either initial configuration or goal configuration
		return [reArrangeGraph, 1]
	else: 
		reArrangeGraph[0].append(newNode) # update node list in graph
		numberNodes = shape(reArrangeGraph[0])[0]
		newNodeIndex = numberNodes -1
		distanceMatrix = np.zeros([numberNodes, numberNodes])
		connectivityMatrix = np.zeros([numberNodes, numberNodes])
		for i_row in range(numberNodes):
			for i_column in range(numberNodes):
				if (i_row < numberNodes -1) and (i_column < numberNodes -1):
					distanceMatrix[i_row][i_column] = reArrangeGraph[2][i_row][i_column]
				else:
					if i_row < i_column:
						# pdb.set_trace() # breakpoint, make real object untransparent
						distanceMatrix[i_row][i_column] = giveDistanceBetweenCfgs(reArrangeGraph[0][i_row], reArrangeGraph[0][i_column], area)
					elif i_row == i_column:
						distanceMatrix[i_row][i_column] = 0.0
					else:
						distanceMatrix[i_row][i_column] = distanceMatrix[i_column][i_row]
		connectivityMatrix[0:numberNodes -1, 0:numberNodes -1] = reArrangeGraph[1][:,:]
		reArrangeGraph[1] = connectivityMatrix
		reArrangeGraph[2] = distanceMatrix
	return [reArrangeGraph, newNodeIndex]

def giveClosestNodeIndexReArrangementPRMGraph(reArrangeGraph, newNodeIndex, numOfNeighbors):
	numberNodes = shape(reArrangeGraph[0])[0]
	distanceVec = reArrangeGraph[2][newNodeIndex]
	distanceVecSortedIndex = sorted(range(len(distanceVec)), key=lambda i:distanceVec[i]) # from small to large
	closestNodeIndex = distanceVecSortedIndex[1:1 + numOfNeighbors]
	return closestNodeIndex

def giveDistanceReArrangementPRMGraph(reArrangeNode, area):
	numberNodes = shape(reArrangeNode)[0]
	reArrangeDistance = np.zeros([numberNodes, numberNodes])
	for i_row in range(numberNodes):
		for i_column in range(numberNodes):
			if i_row < i_column:
				reArrangeDistance[i_row][i_column] = giveDistanceBetweenCfgs(reArrangeNode[i_row], reArrangeNode[i_column], area)
			elif i_row == i_column:
				reArrangeDistance[i_row][i_column] = 0.0
			else:
				reArrangeDistance[i_row][i_column] = reArrangeDistance[i_column][i_row]
	return reArrangeDistance

def giveDistanceBetweenCfgs(cfg1, cfg2, area):
	cfg1Norm = np.zeros(shape(cfg1))
	cfg2Norm = np.zeros(shape(cfg2))
	xMin = area[0] - area[2]/2
	xMax = area[0] + area[2]/2
	xMean = area[0]
	xStdDv = math.sqrt((xMax - xMin)**2/12)
	yMin = area[1] - area[3]/2
	yMax = area[1] + area[3]/2
	yMean = area[1]
	yStdDv = math.sqrt((yMax - yMin)**2/12)
	rotZMin = 0.0
	rotZMax = 360.0
	rotZMean = 180.0
	rotZStdDv = math.sqrt((rotZMax - rotZMin)**2/12)
	distance = 0
	for i_obj in range(shape(cfg1)[0]):
		#for i_dimension in range(shape(cfg1)[1]):
		cfg1Norm[i_obj][0] = (cfg1[i_obj][0] - xMean)/xStdDv
		cfg2Norm[i_obj][0] = (cfg2[i_obj][0] - xMean)/xStdDv
		distanceX = (cfg1[i_obj][0] - cfg2[i_obj][0])**2
		cfg1Norm[i_obj][1] = (cfg1[i_obj][1] - yMean)/yStdDv
		cfg2Norm[i_obj][1] = (cfg2[i_obj][1] - yMean)/yStdDv
		distanceY = (cfg1[i_obj][1] - cfg2[i_obj][1])**2
		cfg1Norm[i_obj][2] = (cfg1[i_obj][2] - rotZMean)/rotZStdDv
		cfg2Norm[i_obj][2] = (cfg2[i_obj][2] - rotZMean)/rotZStdDv
		distanceRotZ = (cfg1[i_obj][2] - cfg2[i_obj][2])**2
		distance = distance + distanceX + distanceY + distanceRotZ
	distance = math.sqrt(distance)
	return distance

def	resetAllObjToOrigin(objectList):
	for i_obj in range(shape(objectList)[0]):
		objectList[i_obj].SetTransform(np.zeros([4,4]))



def	PLNonMonotoneReArrangementSearchPrimitive(initCfgObjList, refCfgObjList, objectList, area, robot, transitTransferRRTGraphParameter, env):
	for i_object in range(shape(objectList)[0]):
		objIndexRemaining = range(shape(objectList)[0])
		PLNMRPath = PLNonMonotoneReArrangementSearch(i_object, objIndexRemaining, initCfgObjList, refCfgObjList, objectList, area, robot, transitTransferRRTGraphParameter, env)
		if PLNMRPath is not 'null':
			break
	return PLNMRPath

def PLNonMonotoneReArrangementSearch(objInFocusIndex, objIndexRemaining, crrtCfgObjList, refCfgObjList, objectList, area, robot, transitTransferRRTGraphParameter, env):
#	initTransformObjList = giveTransformObjList(objectList)
#	setTransformObjList(objectList, initTransformObjList)
	setObjListTransformFromCfgList(crrtCfgObjList, objectList, area)
	# transitPath = transitMotion()
	objIndexNotRemaining = indexListOfRemainingOrNotRemaining(objIndexRemaining, objectList)
	# pdb.set_trace() # breakpoint	
	transferPath = transferMotion(objInFocusIndex, refCfgObjList[objInFocusIndex], objIndexNotRemaining, objectList, area, robot, transitTransferRRTGraphParameter, env)
	# pdb.set_trace() # breakpoint
	totalPath = transferPath
	if totalPath is 'null': # collision with objects in final cfg, no feasible path found
		# pdb.set_trace() #breakpoint	
		return 'null'
	else: # feasible, might have collision with remaining objects
		if totalPath[2] is False: # feasible, no collision
			if objInFocusIndex in objIndexRemaining:
				objIndexRemaining.remove(objInFocusIndex)
			if not objIndexRemaining: # no objects remaining
				# pdb.set_trace() #breakpoint
				return totalPath
			else: # still objects remaining
				for index_ObjectRemaining in range(shape(objIndexRemaining)[0]):
					try: 
						objInFocusIndex_Loop = objIndexRemaining[index_ObjectRemaining]
					except IndexError:
						pdb.set_trace() #breakpoint	
					crrtCfgObjList = totalPath[1]
					objIndexRemainingCopy = objIndexRemaining[:]
					# crrtCfgObjList = fromObjListToCfgListXYRotZ(objectList)
					# pdb.set_trace() #breakpoint	
					PLRSPath = PLNonMonotoneReArrangementSearch(objInFocusIndex_Loop, objIndexRemainingCopy, crrtCfgObjList, refCfgObjList, objectList, area, robot, transitTransferRRTGraphParameter, env)
					if PLRSPath is not 'null': # feasible path
						# pdb.set_trace() #breakpoint	
						totalPath = appendTwoPaths(totalPath, PLRSPath) # feasible path, might contain removed collisions
						# pdb.set_trace() # breakpoint
						return totalPath
		else: # feasible, with collision
			# for index_ObjectCollision in range(shape(totalPath[3])[0]):
			objectCollisionIndex_Loop = totalPath[3][0] # the first object for collision
			crrtCfgObjList = fromObjListToCfgListXYRotZ(objectList)
			# pdb.set_trace() #breakpoint	

			objIndexRemaining_Loop = objIndexRemaining[:]
			if objectCollisionIndex_Loop in objIndexRemaining_Loop:
				objIndexRemaining_Loop.remove(objectCollisionIndex_Loop)
			[crrtCfgObjList, clearedPath] = clearingPath(robot, objectCollisionIndex_Loop, objIndexRemaining_Loop, objectList, crrtCfgObjList, totalPath, area, transitTransferRRTGraphParameter, env)
			if clearedPath is not 'null':
				# pdb.set_trace() #breakpoint	
				totalPath2 = PLNonMonotoneReArrangementSearch(objInFocusIndex, objIndexRemaining, crrtCfgObjList, refCfgObjList, objectList, area, robot, transitTransferRRTGraphParameter, env)
				if totalPath2 is not 'null':
					# pdb.set_trace() #breakpoint
					return appendTwoPaths(clearedPath, totalPath2)
	# pdb.set_trace()
	return 'null'

def transferMotion(objInFocusIndex, finalCfgObjFocus, objIndexNotRemaining, objectList, area, robot, transitTransferRRTGraphParameter, env):
	manip = robot.SetActiveManipulator("lwr")
	initDOF = robot.GetDOFValues()
	objIndexRemaining = indexListOfRemainingOrNotRemaining(objIndexNotRemaining, objectList)
	# if objInFocusIndex in objIndexNotRemaining:
	# 	objIndexNotRemaining.remove(objInFocusIndex)
	objIndexRemainingExceptObjInFocus = objIndexRemaining[:]
	if objInFocusIndex in objIndexRemainingExceptObjInFocus:
		objIndexRemainingExceptObjInFocus.remove(objInFocusIndex)
	# crrtObjTransform = fromCfgToTransformXYRotZ(crrtCfgObjInFocus, area[4])
	crrtObjTransform = objectList[objInFocusIndex].GetTransform() # get the current transform of focused object
	crrtCfgObjList1 = fromObjListToCfgListXYRotZ(objectList) # get the current XYZ of object list
	targetObjTransform = fromCfgToTransformXYRotZ(finalCfgObjFocus, area[4])
	graspTransform = giveGraspTransformsFromObject(objectList[objInFocusIndex], env)
	numGrasp = shape(graspTransform)[0]
	allGraspsTried = False
	cllsnVecNotRemaining = False
	cllsnVecRemainingExceptObjInFocus = False
	objIndexCllsnVecRemaining = []
	# ifgrasped = False
	noFeasibleGrasp = False
	# pdb.set_trace()
	for i_grasp in range(numGrasp):
		# pdb.set_trace()
		if i_grasp == (numGrasp -1):
			allGraspsTried = True
		# check collision before transfer
		sol_DOF = manip.FindIKSolution(graspTransform[i_grasp], IkFilterOptions.CheckEnvCollisions)
		if sol_DOF is None:
			if i_grasp == (numGrasp -1):
				noFeasibleGrasp = True
			continue
		robot.SetDOFValues(sol_DOF, manip.GetArmIndices())
		startingNodeDOF = robot.GetDOFValues(manip.GetArmIndices())
		initDOFPreGrasp = robot.GetDOFValues()
		closingResult = closeFingerExecuteGrasp(robot, objInFocusIndex, objIndexRemainingExceptObjInFocus, objIndexNotRemaining, objectList, env)
		# if objectList[objInFocusIndex].GetName() == 'kitchen_knife' or objectList[objInFocusIndex].GetName() == 'fork':
			# pdb.set_trace()

		if closingResult[0] == 'grasped':
			# pdb.set_trace()
			# check collision after transfer
			# pdb.set_trace() # breakpoint
			objectList[objInFocusIndex].SetTransform(targetObjTransform)
			crrtCfgObjList2 = fromObjListToCfgListXYRotZ(objectList)
			graspTransform2 = giveGraspTransformsFromObject(objectList[objInFocusIndex], env)
			sol_DOF2 = manip.FindIKSolution(graspTransform2[i_grasp], IkFilterOptions.CheckEnvCollisions)
			if sol_DOF2 is None:
				if i_grasp == (numGrasp -1):
					noFeasibleGrasp = True
				objectList[objInFocusIndex].SetTransform(crrtObjTransform)
				robot.SetDOFValues(initDOF)
				continue
			robot.SetDOFValues(sol_DOF2, manip.GetArmIndices())
			targetNodeDOF = robot.GetDOFValues(manip.GetArmIndices())
			# pdb.set_trace()
			openingResult = openFingerAfterGrasp(robot, objInFocusIndex, objIndexRemainingExceptObjInFocus, objIndexNotRemaining, objectList, initDOF, env)
			if openingResult[0] == 'released':
				robot.SetDOFValues(startingNodeDOF, manip.GetArmIndices())
				cllsnCheckObjList = objectList[:]
				cllsnCheckObjList.append(env.GetBodies()[2])
				objectList[objInFocusIndex].SetTransform(crrtObjTransform)
				# pdb.set_trace()
				[path, RRTGraph] = transitTransferRRT(startingNodeDOF, targetNodeDOF, robot, objectList[objInFocusIndex], cllsnCheckObjList, transitTransferRRTGraphParameter, area, env)
				if path is not 'null':
					collisionStatus = False
					transferPath = [crrtCfgObjList1, crrtCfgObjList2, collisionStatus, objIndexCllsnVecRemaining, [startingNodeDOF, targetNodeDOF], path]
					robot.SetDOFValues(initDOF)
					return transferPath
				else:
					if i_grasp == (numGrasp -1):
						cllsnVecNotRemaining = True
					objectList[objInFocusIndex].SetTransform(crrtObjTransform)
					robot.SetDOFValues(initDOF)
					continue
			elif openingResult[0] == 'cllsnRemainingExceptObjInFocus':
				cllsnVecRemainingExceptObjInFocus = True
				objIndexCllsnVecRemaining = openingResult[1]
				objectList[objInFocusIndex].SetTransform(crrtObjTransform)
				robot.SetDOFValues(initDOF)
				continue
			elif (openingResult[0] == 'cllsnNotRemaining') or (openingResult[0] == 'cllsnTable'):
				if i_grasp == (numGrasp -1):
					cllsnVecNotRemaining = True
				objectList[objInFocusIndex].SetTransform(crrtObjTransform)
				robot.SetDOFValues(initDOF)
				continue
		elif closingResult[0] == 'cllsnRemainingExceptObjInFocus':
			cllsnVecRemainingExceptObjInFocus = True
			objIndexCllsnVecRemaining = closingResult[1]
			robot.SetDOFValues(initDOF)
			# i_graspCllsnRemaining = i_grasp
			# objectList[objInFocusIndex].SetTransform(crrtObjTransform)
			continue			
		elif (closingResult[0] == 'cllsnNotRemaining') or (closingResult[0] == 'cllsnTable'):
			if i_grasp == (numGrasp -1):
				cllsnVecNotRemaining = True
			robot.SetDOFValues(initDOF)
			# objectList[objInFocusIndex].SetTransform(crrtObjTransform)
			continue



	# pdb.set_trace()
	if allGraspsTried and cllsnVecRemainingExceptObjInFocus:
		collisionStatus = True
		objectList[objInFocusIndex].SetTransform(targetObjTransform)
		crrtCfgObjList2 = fromObjListToCfgListXYRotZ(objectList)
		objectList[objInFocusIndex].SetTransform(crrtObjTransform)
		transferPath = [crrtCfgObjList1, crrtCfgObjList2, collisionStatus, objIndexCllsnVecRemaining]
		return transferPath
	if (allGraspsTried and noFeasibleGrasp) or (allGraspsTried and cllsnVecNotRemaining):
		# pdb.set_trace() # breakpoint
		return 'null'

def clearingPath(robot, objectCollisionIndex, objIndexRemaining, objectList, crrtCfgObjList, totalPath, area, transitTransferRRTGraphParameter, env):
	intermediatePosePath = intermediatePose(robot, objectCollisionIndex, objIndexRemaining, objectList, area, totalPath, transitTransferRRTGraphParameter, env)
	if intermediatePosePath[2] is False: # there is no collision with remaining objects
		newTransformObjCollsn = fromCfgToTransformXYRotZ(intermediatePosePath[1][objectCollisionIndex], area[4])
		objectList[objectCollisionIndex].SetTransform(newTransformObjCollsn)
		crrtCfgList = fromObjListToCfgListXYRotZ(objectList)
		# pdb.set_trace() # breakpoint
		return [crrtCfgList, intermediatePosePath]
	else: # there is collision with remaining objects
		# for index_ObjectCollision in range(shape(intermediatePosePath[3])[0]): # objects for collision
		objectCollisionIndex_Loop = intermediatePosePath[3][0] # the first object for collision
		objIndexRemaining_Loop = objIndexRemaining[:]
		if objectCollisionIndex_Loop in objIndexRemaining_Loop:
			objIndexRemaining_Loop.remove(objectCollisionIndex_Loop)
		# pdb.set_trace() #breakpoint
		totalPath = appendTwoPaths(totalPath, intermediatePosePath)
		# pdb.set_trace() #breakpoint
		[crrtCfgList, clearedPathMoreCllsn] = clearingPath(robot, objectCollisionIndex_Loop, objIndexRemaining_Loop, objectList, crrtCfgObjList, totalPath, area, transitTransferRRTGraphParameter, env)
		if clearedPathMoreCllsn is not 'null':
			[crrtCfgList, clearedPath] = clearingPath(robot, objectCollisionIndex, objIndexRemaining, objectList, crrtCfgObjList, totalPath, area, transitTransferRRTGraphParameter, env)
			if clearedPath is not 'null':
				# pdb.set_trace() #breakpoint
				return [crrtCfgList, appendTwoPaths(clearedPathMoreCllsn, clearedPath)]
		pdb.set_trace() #breakpoint
		return [crrtCfgList, 'null']

def intermediatePose(robot, objIndex, objIndexRemaining, objectList, area, totalPath, transitTransferRRTGraphParameter, env):
	# pdb.set_trace() #breakpoint
	manip = robot.SetActiveManipulator("lwr")
	initDOF = robot.GetDOFValues()
	initTransformObjList = giveTransformObjList(objectList)
	# objIndexExceptObjInFocus = range(shape(objectList)[0])
	# del objIndexExceptObjInFocus(objIndexExceptObjInFocus == objIndex)
	crrtCfgObjList1 = fromObjListToCfgListXYRotZ(objectList)
	objIndexNotRemaining = indexListOfRemainingOrNotRemaining(objIndexRemaining, objectList)
	if objIndex in objIndexNotRemaining:
		objIndexNotRemaining.remove(objIndex)
	objIndexRemainingExceptObjInFocus = objIndexRemaining[:]
	if objIndex in objIndexRemainingExceptObjInFocus:
		objIndexRemainingExceptObjInFocus.remove(objIndex)
	centerTableX = area[0]
	centerTableY = area[1]
	lengthX = area[2] # table 1
	lengthY = area[3]
	posInXMin = centerTableX - lengthX/2
	posInXMax = centerTableX + lengthX/2
	posInYMin = centerTableY - lengthY/2
	posInYMax = centerTableY + lengthY/2
	posInZ = area[4]
	# cfgObj = []
	initialTransformObj = objectList[objIndex].GetTransform()
	graspTransform = giveGraspTransformsFromObject(objectList[objIndex], env)
	numGrasp = shape(graspTransform)[0]
	# thereIsCollision = True
	numOfSampling = 0
	maxNumOfSampling = 20
	searchStepLimit = 100
	searchStep = 0
	objIndexCllsnVecRemaining = []
	collisionStatus = True
	# cllsnNotRemaining = True
	startingNodeDOF = []
	targetNodeDOF = []
	path = []
	while(numOfSampling < maxNumOfSampling and collisionStatus):
		# pdb.set_trace() #breakpoint
		numOfSampling = numOfSampling + 1
		posInX = centerTableX + np.random.uniform(-lengthX/2+lengthX/20, lengthX/2-lengthX/20, 1)
		posInY = centerTableY + np.random.uniform(-lengthY/2+lengthY/20, lengthY/2-lengthY/20, 1)
		zRotDegree = np.random.uniform(0, 360, 1)
		zRotRadian = zRotDegree/180 * pi
		position = numpy.zeros([4,4])
		position[0:3,3] = [posInX, posInY, posInZ]
		objectTransformNew = giveRotationMatrix3D_4X4('z',zRotRadian) + position
		objectList[objIndex].SetTransform(objectTransformNew)
		if checkCollisionWithList(objectList[objIndex], objectList, env) != []: # collision with object list
			continue
		objectList[objIndex].SetTransform(initialTransformObj)
		for i_grasp in range(numGrasp):
			# check collision before transfer
			sol_DOF = manip.FindIKSolution(graspTransform[i_grasp], IkFilterOptions.CheckEnvCollisions)
			if sol_DOF is None:
				continue
			# pdb.set_trace() # breakpoint
			robot.SetDOFValues(sol_DOF, manip.GetArmIndices())
			startingNodeDOF = robot.GetDOFValues(manip.GetArmIndices())
			closingResult = closeFingerExecuteGrasp(robot, objIndex, objIndexRemainingExceptObjInFocus, objIndexNotRemaining, objectList, env)
			if closingResult[0] is 'grasped':
				# pdb.set_trace() #breakpoint
				objectList[objIndex].SetTransform(objectTransformNew)
				graspTransform2 = giveGraspTransformsFromObject(objectList[objIndex], env)
				sol_DOF2 = manip.FindIKSolution(graspTransform2[i_grasp], IkFilterOptions.CheckEnvCollisions)
				if sol_DOF2 is None:
					objectList[objIndex].SetTransform(initialTransformObj)
					robot.SetDOFValues(initDOF)
					continue
				robot.SetDOFValues(sol_DOF2, manip.GetArmIndices())
				targetNodeDOF = robot.GetDOFValues(manip.GetArmIndices())
				openingResult = openFingerAfterGrasp(robot, objIndex, objIndexRemainingExceptObjInFocus, objIndexNotRemaining, objectList, initDOF, env)
				if openingResult[0] == 'released':
					# pdb.set_trace() #breakpoint
					robot.SetDOFValues(startingNodeDOF, manip.GetArmIndices()) # arm
					cllsnCheckObjList = objectList[:]
					cllsnCheckObjList.append(env.GetBodies()[2])
					objectList[objIndex].SetTransform(initialTransformObj)
					# pdb.set_trace()
					[path, RRTGraph] = transitTransferRRT(startingNodeDOF, targetNodeDOF, robot, objectList[objIndex], cllsnCheckObjList, transitTransferRRTGraphParameter, area, env)
					if path is not 'null':
						collisionStatus = False
						robot.SetDOFValues(initDOF)
						# cllsnNotRemaining = False
						break
				elif openingResult[0] == 'cllsnRemainingExceptObjInFocus':
					objIndexCllsnVecRemaining = openingResult[1]
					objectList[objIndex].SetTransform(initialTransformObj)
					# cllsnNotRemaining = False
			elif closingResult[0] == 'cllsnRemainingExceptObjInFocus':
				objIndexCllsnVecRemaining = closingResult[1]
				# cllsnNotRemaining = False
			elif closingResult[0] == 'cllsnNotRemaining':
				# pdb.set_trace() #breakpoint
				# print 'wait'
				numOfSampling = numOfSampling -1 
			if numOfSampling == maxNumOfSampling -1:
				pdb.set_trace() #breakpoint
				print 'wait'

	# pdb.set_trace() #breakpoint
	# cfgObj = [posInX, posInY, zRotDegree]
	objectList[objIndex].SetTransform(objectTransformNew)
	crrtCfgObjList2 = fromObjListToCfgListXYRotZ(objectList)
	# pdb.set_trace() #breakpoint
	intermediatePosePath = [crrtCfgObjList1, crrtCfgObjList2, collisionStatus, objIndexCllsnVecRemaining, [startingNodeDOF, targetNodeDOF], path]
	setTransformObjList(objectList, initTransformObjList)
	# pdb.set_trace() #breakpoint	
	robot.SetDOFValues(initDOF)
	return intermediatePosePath

def pathExecution(totalPath, objectList, env, area):
	setObjListTransformFromCfgList(totalPath[0][0], objectList, area)
	pathNumber = shape(totalPath)[0]
	for i_path in range(pathNumber):
		# pdb.set_trace() #breakpoint
		path_Loop = totalPath[i_path]
		cfg1 = path_Loop[0] # start configuration
		cfg2 = path_Loop[1] # target configuration
		objMovedIndex = findMovedObjIndexFromTwoCfgs(cfg1, cfg2)
		objectToMove = objectList[objMovedIndex]

		for i_lift in range(10):
			zIncrem = 0.08
			transformObjList = objectToMove.GetTransform()
			transformObjList[2,3] = transformObjList[2,3] + zIncrem
			objectToMove.SetTransform(transformObjList)
		  	time.sleep(0.1)

		deltaX = (cfg2[objMovedIndex][0] - objectToMove.GetTransform()[0,3])/10.0
		deltaY = (cfg2[objMovedIndex][1] - objectToMove.GetTransform()[1,3])/10.0
		deltaRotZ = (cfg2[objMovedIndex][2] - cfg1[objMovedIndex][2])/10.0 # degree
		for i_move in range(10):
			xTo = objectToMove.GetTransform()[0,3] + deltaX
			yTo = objectToMove.GetTransform()[1,3] + deltaY
			rotZTo = cfg1[objMovedIndex][2] + (i_move+1) * deltaRotZ
			transformObjList = objectToMove.GetTransform()
			transformObjList[0,3] = xTo
			transformObjList[1,3] = yTo
			objectToMove.SetTransform(transformObjList)
			rotateObjectInAir(objectToMove, rotZTo)
		  	time.sleep(0.1)

		for i_lift in range(10):
			zIncrem = -0.08
			transformObjList = objectToMove.GetTransform()
			transformObjList[2,3] = transformObjList[2,3] + zIncrem
			objectToMove.SetTransform(transformObjList)
	  		time.sleep(0.1)

def pathExecutionWithArm(totalPath, objectList, robot, env, transitTransferRRTGraphParameter, area):
	manip = robot.SetActiveManipulator("lwr")
	setObjListTransformFromCfgList(totalPath[0][0], objectList, area)
	pathNumber = shape(totalPath)[0]
	objectListExecute = objectList[:]
	objectListExecute.append(env.GetBodies()[2])
	initDOFArm = robot.GetDOFValues(manip.GetArmIndices())
	for i_path in range(pathNumber):

		# pdb.set_trace() # breakpoint
		path_Loop = totalPath[i_path]
		cfg1 = path_Loop[0] # start configuration 
		cfg2 = path_Loop[1] # target configuration 
		objMovedIndex = findMovedObjIndexFromTwoCfgs(cfg1, cfg2)
		objectToMove = objectListExecute[objMovedIndex]
		grasp1DOF = path_Loop[4][0]
		grasp2DOF = path_Loop[4][1]
		rrtPath = path_Loop[5]
		initDOFHand = robot.GetDOFValues()[8:23]
		startDOFArm = robot.GetDOFValues(manip.GetArmIndices())
		# pdb.set_trace() # breakpoint
		# move arm to start grasping position
		executeTransitTransferRRT(startDOFArm, grasp1DOF, [], robot, [], objectListExecute, transitTransferRRTGraphParameter, area, env)
		# grasp
		executeCloseFingerGrasp(robot, objMovedIndex, objectListExecute, env)
		# move object to target grasping position
		executeTransitTransferRRT(grasp1DOF, grasp2DOF, rrtPath, robot, objectToMove, objectListExecute, transitTransferRRTGraphParameter, area, env)
		# open
		executeOpenFingerAfterGrasp(robot, initDOFHand, env)
	executeTransitTransferRRT(grasp2DOF, initDOFArm, [], robot, [], objectListExecute, transitTransferRRTGraphParameter, area, env)

def totalPathExecutionWithArm(pathFoundInGraph, reArrangeGraph, objectList, robot1, area, transitTransferRRTGraphParameter, env):
	for i_edge in range(shape(pathFoundInGraph)[0]-1):
		print 'Edge: from Node ' + str(pathFoundInGraph[i_edge]) + ' to ' + str(pathFoundInGraph[i_edge+1])
		print 'Execute next edge: '
		pdb.set_trace()
		edgeThisLoop = [pathFoundInGraph[i_edge], pathFoundInGraph[i_edge+1]]
		PLNMRPath = findPathFromTwoNodes(reArrangeGraph[3], edgeThisLoop)
		# PLNMRPath = reArrangeGraph[3][i_edge][2]
		setObjListTransformFromCfgList(PLNMRPath[0][0], objectList, area)
		pathExecutionWithArm(PLNMRPath, objectList, robot1, env, transitTransferRRTGraphParameter, area)



def	findPathFromTwoNodes(edgeSet, edgeThisLoop):
	startingNode = edgeThisLoop[0]
	endNode = edgeThisLoop[1]
	inversePath = False
	for i_edge in range(len(edgeSet)):
		if (edgeSet[i_edge][0] == startingNode) and (edgeSet[i_edge][1] == endNode):
			break
		elif (edgeSet[i_edge][0] == endNode) and (edgeSet[i_edge][1] == startingNode):
			inversePath = True
			break
	if inversePath == True:
		PLNMRPathReversed = edgeSet[i_edge][2]
		sequenceNumber = shape(PLNMRPathReversed)[0]
		PLNMRPathReversed = PLNMRPathReversed[::-1]
		PLNMRPath = []
		for i_path in range(sequenceNumber):
			crrtCfgObjList1 = PLNMRPathReversed[i_path][1]
			crrtCfgObjList2 = PLNMRPathReversed[i_path][0]
			collisionStatus = PLNMRPathReversed[i_path][2]
			objIndexCllsnVecRemaining = PLNMRPathReversed[i_path][3]
			startingNodeDOF = PLNMRPathReversed[i_path][4][1]
			targetNodeDOF = PLNMRPathReversed[i_path][4][0]
			path = PLNMRPathReversed[i_path][5][::-1]
			PLNMRPath.append([crrtCfgObjList1, crrtCfgObjList2, collisionStatus, objIndexCllsnVecRemaining, [startingNodeDOF, targetNodeDOF], path])
	else:
		PLNMRPath = edgeSet[i_edge][2]
	# reversed(array)

	return PLNMRPath

def appendTwoPaths(totalPath, pathToBeAdded):
	# pdb.set_trace() #breakpoint	
	if shape(shape(totalPath))[0] == 1: # totalPath is only one path
		# pdb.set_trace() #breakpoint	
		totalPath = [totalPath]
	if shape(shape(pathToBeAdded))[0] == 1: # pathToBeAdded is only one path
		totalPath.append(pathToBeAdded)
	else: # pathToBeAdded has several paths
		for i_path in range(shape(pathToBeAdded)[0]):
			totalPath.append(pathToBeAdded[i_path])
	# pdb.set_trace() #breakpoint	
	return totalPath

def checkCllsnAlongPath(objIndex, objectList, totalPath, env, area):
	initTransformObjList = giveTransformObjList(objectList)
	# 
	if shape(shape(totalPath))[0] == 1: # there is only one path
		movedObjIndex = findMovedObjIndexFromTwoCfgs(totalPath[0], totalPath[1])
		# configuration before move
		cfgMovedObjIndex1 = totalPath[0][movedObjIndex] 
		transformMovedObjIndex1 = fromCfgToTransformXYRotZ(cfgMovedObjIndex1, area[4])
		objectList[movedObjIndex].SetTransform(transformMovedObjIndex1)
		thereIsCollisionAlongPath1 = env.CheckCollision(objectList[movedObjIndex], objectList[objIndex])
		# configuration after move
		cfgMovedObjIndex2 = totalPath[1][movedObjIndex] 
		transformMovedObjIndex2 = fromCfgToTransformXYRotZ(cfgMovedObjIndex2, area[4])
		objectList[movedObjIndex].SetTransform(transformMovedObjIndex2)
		thereIsCollisionAlongPath2 = env.CheckCollision(objectList[movedObjIndex], objectList[objIndex])
		thereIsCollisionAlongPath = thereIsCollisionAlongPath1 or thereIsCollisionAlongPath2
	else: # there are several paths
		# pdb.set_trace() #breakpoint
		thereIsCollisionAlongPath = False
		for i_path in range(shape(shape(totalPath))[0]):
			thisPath = totalPath[i_path]
			movedObjIndexThisPath = findMovedObjIndexFromTwoCfgs(thisPath[0], thisPath[1])
			# configuration before move
			# pdb.set_trace() #breakpoint
			cfgMovedObjIndexThisPath1 = thisPath[0][movedObjIndexThisPath]
			transformMovedObjIndexThisPath1 = fromCfgToTransformXYRotZ(cfgMovedObjIndexThisPath1, area[4])
			objectList[movedObjIndexThisPath].SetTransform(transformMovedObjIndexThisPath1)
			thereIsCollisionAlongPath_ThisPath1 = env.CheckCollision(objectList[movedObjIndexThisPath], objectList[objIndex])
			# configuration after move
			cfgMovedObjIndexThisPath2 = thisPath[1][movedObjIndexThisPath]
			transformMovedObjIndexThisPath2 = fromCfgToTransformXYRotZ(cfgMovedObjIndexThisPath2, area[4])
			objectList[movedObjIndexThisPath].SetTransform(transformMovedObjIndexThisPath2)
			thereIsCollisionAlongPath_ThisPath2 = env.CheckCollision(objectList[movedObjIndexThisPath], objectList[objIndex])
			thereIsCollisionAlongPath_ThisPath = thereIsCollisionAlongPath_ThisPath1 or thereIsCollisionAlongPath_ThisPath2
			thereIsCollisionAlongPath = thereIsCollisionAlongPath or thereIsCollisionAlongPath_ThisPath
	setTransformObjList(objectList, initTransformObjList)
	return thereIsCollisionAlongPath




def	findMovedObjIndexFromTwoCfgs(cfg1, cfg2):
	# pdb.set_trace() #breakpoint	
	for i_object in range(shape(cfg1)[0]):
		# pdb.set_trace() #breakpoint	
		if np.round(cfg1[i_object][0],5) == np.round(cfg2[i_object][0],5) \
			and np.round(cfg1[i_object][1],5) == np.round(cfg2[i_object][1],5) \
			and np.round(cfg1[i_object][2],5) == np.round(cfg2[i_object][2],5):
			continue
		else:
			return i_object
		


def indexListOfRemainingOrNotRemaining(objIndexList, objectList):
	newObjList = []
	for i_index in range(shape(objectList)[0]):
		if i_index in objIndexList:
			continue
		else:
			newObjList.append(i_index)	
	return newObjList

def fromObjIndexToObjList(objIndexList, objectList):
	newObjList = []
	for i_index in range(shape(objIndexList)[0]):
		objIndex = objIndexList[i_index]
		newObjList.append(objectList[objIndex])
	return newObjList

def setObjListTransformFromCfgList(cfgList, objectList, area):
	transformObjList = []
	for i_cfg in range(shape(cfgList)[0]):
		transformObjList.append(fromCfgToTransformXYRotZ(cfgList[i_cfg], area[4]))
	setTransformObjList(objectList, transformObjList)

def fromObjListToCfgListXYRotZ(objectList):
	cfgList = []
	for i_object in range(shape(objectList)[0]):
		objectTransform = objectList[i_object].GetTransform()
		cfgList.append(fromTransformToCfgXYRotZ(objectTransform))
	return cfgList

def fromTransformToCfgXYRotZ(objectTransform):
	# known that the object is only rotated around z axis.
	xPos = objectTransform[0,3]
	yPos = objectTransform[1,3]
	cosValue = objectTransform[0][0]
	degreeFromCos = np.arccos(cosValue)/pi*180
	sinValue = -objectTransform[0][1]
	degreeFromSin = np.arccos(sinValue)/pi*180
	if cosValue == -1:
		degreeValue = 180
	elif cosValue>-1 and cosValue < 1:
		degreeValue1 = degreeFromCos
		degreeValue2 = 360 - degreeFromCos
		if sinValue>0:
			degreeValue = degreeValue1
		else:
			degreeValue = degreeValue2
	elif cosValue == 1:
		degreeValue == 0
	return [xPos, yPos, degreeValue]


def fromCfgToTransformXYRotZ(cfgArray, zHeight):
	xCfg = cfgArray[0]
	yCfg = cfgArray[1]
	zCfg = zHeight
	rotZCfg = cfgArray[2] # degree
	rotZCfgRadian = rotZCfg/180 * pi
	position = numpy.zeros([4,4])
	position[0:3,3] = [xCfg, yCfg, zCfg]
	objectTransform = giveRotationMatrix3D_4X4('z', rotZCfgRadian) + position
	return objectTransform

#############################################################################################################################
# PRM construct
#############################################################################################################################
def transitPRMConstruct(robot, area, objectList, env, transitPRMGraphParam):
	manip = robot.SetActiveManipulator("lwr")
	manip_tool = manip.GetEndEffector()
	numberOfJoints = len(manip.GetArmIndices())
	manipDOFLimitsMin = robot.GetDOFLimits()[0][manip.GetArmIndices()]
	manipDOFLimitsMax = robot.GetDOFLimits()[1][manip.GetArmIndices()]
	manipDOFLimit = [manipDOFLimitsMin, manipDOFLimitsMax]
	if env.GetBodies()[2].GetName() != 'table1':
		pdb.set_trace() #breakpoint3
	table1_Obj = env.GetBodies()[2]
	connectionRadius = transitPRMGraphParam[1]
	transitPRMVertices = []
	transitPRMEdges = []
	transitPRMGraph = [transitPRMVertices, transitPRMEdges] # Graph
	numInterpolation = transitPRMGraphParam[2]
	nodesForGrasps = giveGraspFromRandomObjPoses(robot, objectList, transitPRMGraphParam, area, env)
	# pdb.set_trace() #breakpoint
	nodesForSmoothingGrasps = giveNodeSmoothingGraspsFromGrasp(robot, nodesForGrasps, manipDOFLimit, env, transitPRMGraphParam)

	for i_NodeGrasp in range(shape(nodesForGrasps)[0]):
		transitPRMVertices.append(nodesForGrasps[i_NodeGrasp])
	for i_NodeSmoothingGrasp in range(shape(nodesForSmoothingGrasps)[0]):
		transitPRMVertices.append(nodesForSmoothingGrasps[i_NodeSmoothingGrasp])



	for i_node in range(transitPRMGraphParam[0]): # number of sampled nodes
		print 'sampling node:' + str(i_node)
		transitPRMVertices.append(sampleJointValueInsideWorkSpace(robot, manipDOFLimit, area, env))
	# pdb.set_trace() #breakpoint
	for i_sample in range(shape(transitPRMVertices)[0]):
		print str(i_sample)
		[nearestNodesIndex, nearestNodesDistance] = chooseNodesInsideBall(transitPRMGraph, i_sample, transitPRMGraphParam[1], robot, env)
		# pdb.set_trace() #breakpoint
		for i_nearNode in range(shape(nearestNodesIndex)[0]):
			if not checkCollisionBetweenTwoVertices(transitPRMGraph, nearestNodesIndex[i_nearNode], i_sample, numInterpolation, robot, table1_Obj, env):
				if nearestNodesIndex[i_nearNode] < i_sample:
					transitPRMGraph[1].append([nearestNodesIndex[i_nearNode], i_sample, nearestNodesDistance[i_nearNode]])
					# pdb.set_trace() #breakpoint
				else:
					transitPRMGraph[1].append([i_sample, nearestNodesIndex[i_nearNode], nearestNodesDistance[i_nearNode]])
	return transitPRMGraph


def findShortestPathUsingDijkstra(transitPRMGraph, startingNodeDOF, targetNodeDOF, transitPRMGraphParam, robot, cllsnCheckObjList, env):
	manip = robot.SetActiveManipulator("lwr")
	if env.GetBodies()[2].GetName() != 'table1':
		pdb.set_trace() #breakpoint3
	table1_Obj = env.GetBodies()[2]
	ikmodel = databases.inversekinematics.InverseKinematicsModel(robot = robot, iktype = IkParameterization.Type.Transform6D)
	if not ikmodel.load():
		ikmodel.autogenerate()
	
	# env.GetBodies()[4].SetTransform([[1,0,0,0.150],[0,1,0,-0.0411],[0,0,1,1.0072],[0,0,0,1]])
	# check collision at starting configuration
	robot.SetDOFValues(startingNodeDOF, manip.GetArmIndices())
	for i_cllsnCheckObj in range(shape(cllsnCheckObjList)[0]):
		if env.CheckCollision(robot, cllsnCheckObjList[i_cllsnCheckObj]):
			print "Collision with " + cllsnCheckObjList[i_cllsnCheckObj].GetName() + "at starting pose."
			pdb.set_trace() #breakpoint
			return []
	# check collision at target configuration
	robot.SetDOFValues(targetNodeDOF, manip.GetArmIndices())
	for i_cllsnCheckObj in range(shape(cllsnCheckObjList)[0]):
		if env.CheckCollision(robot, cllsnCheckObjList[i_cllsnCheckObj]):
			print "Collision with " + cllsnCheckObjList[i_cllsnCheckObj].GetName() + "at target pose."
			pdb.set_trace() #breakpoint
			return []
	# add nodes to graph
	transitPRMGraph[0].append(startingNodeDOF)
	indexStartingNodeDOF = shape(transitPRMGraph[0])[0]-1
	transitPRMGraph[0].append(targetNodeDOF)
	indexTargetNodeDOF = shape(transitPRMGraph[0])[0]-1
	# add edges for starting node
	distanceToStartVec = []
	foundAConnectionForStarting = False
	for i_sample in range(shape(transitPRMGraph[0])[0]-2):
		distanceToStartVec.append(LA.norm(startingNodeDOF-transitPRMGraph[0][i_sample]))
		if (distanceToStartVec[i_sample] < transitPRMGraphParam[1]):
			# pdb.set_trace() #breakpoint
			if not checkCollisionBetweenTwoVertices(transitPRMGraph, i_sample, indexStartingNodeDOF, transitPRMGraphParam[2], robot, table1_Obj, env):
				transitPRMGraph[1].append([i_sample, indexStartingNodeDOF, distanceToStartVec[i_sample]])
				foundAConnectionForStarting = True
		continue
	# pdb.set_trace() #breakpoint
	if foundAConnectionForStarting == False:
		transitPRMGraph[1].append([np.argmin(distanceToStartVec), indexStartingNodeDOF, distanceToStartVec[np.argmin(distanceToStartVec)]])
	# add edges for target node
	distanceToTargetVec = []
	foundAConnectionForTarget = False
	for i_sample in range(shape(transitPRMGraph[0])[0]-1):
		distanceToTargetVec.append(LA.norm(targetNodeDOF-transitPRMGraph[0][i_sample]))
		if (distanceToTargetVec[i_sample] < transitPRMGraphParam[1]):
			# pdb.set_trace() #breakpoint
			if not checkCollisionBetweenTwoVertices(transitPRMGraph, i_sample, indexTargetNodeDOF, transitPRMGraphParam[2], robot, table1_Obj, env):
				transitPRMGraph[1].append([i_sample, indexTargetNodeDOF, distanceToTargetVec[i_sample]])
				foundAConnectionForTarget = True
		continue
	if foundAConnectionForTarget == False:
		transitPRMGraph[1].append([np.argmin(distanceToTargetVec), indexTargetNodeDOF, distanceToTargetVec[np.argmin(distanceToStartVec)]])
	# pdb.set_trace() #breakpoint
	# loop for deleting the edges, where there is a collision
	while(True):
		g_Dijkstra = Graph()
		addPRMGraphToDijkstra(g_Dijkstra, transitPRMGraph)
		dijkstra(g_Dijkstra, g_Dijkstra.get_vertex(indexStartingNodeDOF)) # define starting node
		target = g_Dijkstra.get_vertex(indexTargetNodeDOF) # define target node
		pathUnsorted = [target.get_id()]
		shortest(target, pathUnsorted)
		path = pathUnsorted[::-1]
		# path = [6197, 1002, 6198]
		nodesIncludedInPath = shape(path)[0]
		if nodesIncludedInPath == 1: # there is no path found
			print "No path found"
			pdb.set_trace() #breakpoint
			return []
		else:
			# collisionAlongTotalPath = False
			for i_pathPiece in range(nodesIncludedInPath-1):
				collisionAlongPathPiece = False
				startNodeInPathPiece = path[i_pathPiece]
				targetNodeInPathPiece = path[i_pathPiece + 1]
				DOFVertex1 = transitPRMGraph[0][startNodeInPathPiece]
				DOFVertex2 = transitPRMGraph[0][targetNodeInPathPiece]
				numStep = 20.0
				diffDOF = (DOFVertex2 - DOFVertex1)/(numStep)
				for i_interpolator in range(int(numStep+1)):
					crrtDOF = DOFVertex1 + diffDOF * i_interpolator
					robot.SetDOFValues(crrtDOF, manip.GetArmIndices())
					for i_cllsnCheckObj in range(shape(cllsnCheckObjList)[0]):
						if env.CheckCollision(robot, cllsnCheckObjList[i_cllsnCheckObj]):
							collisionAlongPathPiece = True
							# pdb.set_trace() #breakpoint
							deleteAnEdgeFromGraph(transitPRMGraph, startNodeInPathPiece, targetNodeInPathPiece)
							break 
					if collisionAlongPathPiece == True:
						break
			if collisionAlongPathPiece == False:
				print 'Path found!'
				return  path # no collision along the total path

def	deleteAnEdgeFromGraph(transitPRMGraph, startNodeInPathPiece, targetNodeInPathPiece):
	# pdb.set_trace() #breakpoint
	for i_edge in range(shape(transitPRMGraph[1])[0]):
		if (transitPRMGraph[1][i_edge][0] == startNodeInPathPiece) and (transitPRMGraph[1][i_edge][1] == targetNodeInPathPiece):
			del transitPRMGraph[1][i_edge]
			break
		elif (transitPRMGraph[1][i_edge][0] == targetNodeInPathPiece) and (transitPRMGraph[1][i_edge][1] == startNodeInPathPiece):
			del transitPRMGraph[1][i_edge]
			break

def giveGraspFromRandomObjPoses(robot, objectList, transitPRMGraphParam, area, env):
	nodesForGrasps = []
	manip = robot.SetActiveManipulator("lwr")
	manip_tool = manip.GetEndEffector()
	#
	ikmodel = databases.inversekinematics.InverseKinematicsModel(robot = robot, iktype = IkParameterization.Type.Transform6D)
	if not ikmodel.load():
		ikmodel.autogenerate()
	#
	numRandomGenObjPoses = transitPRMGraphParam[3] # number of randomly generating object poses
	for i_randPoses in range(numRandomGenObjPoses):
		print 'random object poses: ' + str(i_randPoses)
		objListWithRandPoses = genrateRandomPositionObjectsInARectAreaWithZRot(area, objectList, env, 'random')
		numObj = shape(objListWithRandPoses)[0]
		for i_obj in range(numObj):
			# pdb.set_trace() #breakpoint
			graspForThisObj = giveGraspTransformsFromObject(objectList[i_obj], env)
			numOfGrasp = shape(graspForThisObj)[0]
			for i_grasp in range(numOfGrasp):
				sol_DOF = manip.FindIKSolution(graspForThisObj[i_grasp],  IkFilterOptions.IgnoreEndEffectorEnvCollisions)
				if sol_DOF is None:
					continue
					pdb.set_trace() #breakpoint
				robot.SetDOFValues(sol_DOF, manip.GetArmIndices())
				if env.CheckCollision(robot, objectList[i_obj]):
					continue
				if env.GetBodies()[2].GetName() != 'table1':
					pdb.set_trace() #breakpoint
				if env.CheckCollision(robot, env.GetBodies()[2]):
					continue
				nodesForGrasps.append(sol_DOF)			
	return nodesForGrasps

def giveNodeSmoothingGraspsFromGrasp(robot, nodesForGrasps, manipDOFLimit, env, transitPRMGraphParam):
	numOfSmoothingGrasps = transitPRMGraphParam[4]
	nodesForSmoothingGrasps = []
	manip = robot.SetActiveManipulator("lwr")
	manipDOFLimitMin = manipDOFLimit[0]
	manipDOFLimitMax = manipDOFLimit[1]
	diffDOFMaxMin = manipDOFLimitMax - manipDOFLimitMin
	maxVariationDOF = diffDOFMaxMin/100.0
	for i_grasp in range(shape(nodesForGrasps)[0]):
		DOFOfGrasp_Loop = nodesForGrasps[i_grasp]
		for i_smoothing in range(numOfSmoothingGrasps):
			smoothingGraspDOF = np.zeros(shape(DOFOfGrasp_Loop))
			# robot.SetDOFValues(DOFOfGrasp_Loop, manip.GetArmIndices())
			for i_DOF in range(shape(DOFOfGrasp_Loop)[0]):
				smoothingGraspDOFWithoutLimit = DOFOfGrasp_Loop[i_DOF] + np.random.uniform(-maxVariationDOF[i_DOF], maxVariationDOF[i_DOF], 1)
				smoothingGraspDOF[i_DOF] = max(min(smoothingGraspDOFWithoutLimit, manipDOFLimitMax[i_DOF]), manipDOFLimitMin[i_DOF])
			# pdb.set_trace() #breakpoint
			robot.SetDOFValues(smoothingGraspDOF, manip.GetArmIndices())
			if env.GetBodies()[2].GetName() != 'table1':
				pdb.set_trace() #breakpoint
			if env.CheckCollision(robot, env.GetBodies()[2]):
				continue
			nodesForSmoothingGrasps.append(smoothingGraspDOF)
	return nodesForSmoothingGrasps

def	sampleJointValueInsideWorkSpace(robot, manipDOFLimit, area, env):
	manip = robot.SetActiveManipulator("lwr")
	manip_tool = manip.GetEndEffector()
	xMin = area[0] - (area[2]/2.0) * 1.2
	xMax = area[0] + area[2]/2.0
	yMin = area[1] - area[3]/2.0
	yMax = area[1] + area[3]/2.0
	zMin = area[4]
	zMax = zMin + 0.5
	numberOfJoints = len(manipDOFLimit[0])
	sampledJointValues = np.zeros([numberOfJoints])
	validSampling = False
	while(not validSampling):
		for i_joint in range(numberOfJoints):
			sampledJointValues[i_joint] = np.random.uniform(manipDOFLimit[0][i_joint], manipDOFLimit[1][i_joint], 1)
		robot.SetDOFValues(sampledJointValues, manip.GetArmIndices())
		if env.GetBodies()[2].GetName() != 'table1':
				pdb.set_trace() #breakpoint			
		collisionWithTable1 = env.CheckCollision(robot, env.GetBodies()[2])
		xToolTransform = manip_tool.GetTransform()[0,3]
		yToolTransform = manip_tool.GetTransform()[1,3]
		zToolTransform = manip_tool.GetTransform()[2,3]
		ifInsideWorkspace = xToolTransform > xMin and xToolTransform < xMax and yToolTransform > yMin and yToolTransform < yMax and zToolTransform > zMin and zToolTransform < zMax
		validSampling = (not collisionWithTable1) and ifInsideWorkspace
		# print validSampling
		# pdb.set_trace() #breakpoint
	return sampledJointValues

def chooseNodesInsideBall(transitPRMGraph, vertexIndexInGraph, connectionRadius, robot, env):
	transitPRMVertices = transitPRMGraph[0]
	nearestNodesIndex = []
	nearestNodesDistance = []
	for i_vertex in range(shape(transitPRMVertices)[0]):		
		if i_vertex <= vertexIndexInGraph:
			continue
		distanceToVertex = LA.norm(transitPRMVertices[i_vertex] - transitPRMVertices[vertexIndexInGraph])
		# print distanceToVertex
		# pdb.set_trace() #breakpoint
		if distanceToVertex < connectionRadius:
			# print distanceToVertex
			# pdb.set_trace() #breakpoint
			nearestNodesIndex.append(i_vertex)
			nearestNodesDistance.append(distanceToVertex)
	return [nearestNodesIndex, nearestNodesDistance]
	# manip = robot.SetActiveManipulator("lwr")
	# robot.SetDOFValues(transitPRMVertices[i_vertex], manip.GetArmIndices())
	# robot.SetDOFValues(transitPRMVertices[vertexIndexInGraph], manip.GetArmIndices())

def checkCollisionBetweenTwoVertices(transitPRMGraph, vertex1, vertex2, numInterpolation, robot, checkCllsnWith, env):
	manip = robot.SetActiveManipulator("lwr")
	manip_tool = manip.GetEndEffector()
	DOFVertex1 = transitPRMGraph[0][vertex1]
	DOFVertex2 = transitPRMGraph[0][vertex2]
	diffDOF = (DOFVertex2 - DOFVertex1)/(numInterpolation+1)
	for i_interpolator in range(int(numInterpolation+2)):
		crrtDOF = DOFVertex1 + diffDOF * i_interpolator
		robot.SetDOFValues(crrtDOF, manip.GetArmIndices())
		# pdb.set_trace() #breakpoint
		if checkCllsnWith == 'env':
			if checkCollision(robot, env):
				return True # there is collision, no connection between these two vertices
		else:
			objToCheckCllsnWith = checkCllsnWith # data type is an object class
			if env.CheckCollision(robot, objToCheckCllsnWith):
				return True
	return False # no collision

def addPRMGraphToDijkstra(g_Dijkstra, transitPRMGraph):
	transitPRMVertices = transitPRMGraph[0]
	transitPRMEdges = transitPRMGraph[1]
	for i_node in range(shape(transitPRMVertices)[0]):
		g_Dijkstra.add_vertex(i_node)
	for i_edge in range(shape(transitPRMEdges)[0]):
		g_Dijkstra.add_edge(transitPRMEdges[i_edge][0], transitPRMEdges[i_edge][1], transitPRMEdges[i_edge][2])


#############################################################################################################################
# transit&transfer RRT (begin)
#############################################################################################################################
def transitTransferRRT(startingNodeDOF, targetNodeDOF, robot, objGrasped, cllsnCheckObjList, transitTransferRRTGraphParameter, area, env):
	numRRTIteration = transitTransferRRTGraphParameter[0] # 100
	collisionBackResidual = transitTransferRRTGraphParameter[1] # 0.3
	maxCollisionCheckDOFIncrement = transitTransferRRTGraphParameter[2] # 0.1
	goalSampleRate = transitTransferRRTGraphParameter[4]
	goalConnectionRate = transitTransferRRTGraphParameter[5]
	if objGrasped != []:
		initObjTransform = objGrasped.GetTransform()
	manip = robot.SetActiveManipulator("lwr")
	endTool = manip.GetEndEffector()
	manipDOFLimitsMin = robot.GetDOFLimits()[0][manip.GetArmIndices()]
	manipDOFLimitsMax = robot.GetDOFLimits()[1][manip.GetArmIndices()]
	manipDOFLimit = [manipDOFLimitsMin, manipDOFLimitsMax]
	robotArmInitDOF = robot.GetDOFValues(manip.GetArmIndices())
	if env.GetBodies()[2].GetName() != 'table1':
		pdb.set_trace() #breakpoint
	table1_Obj = env.GetBodies()[2]
	ikmodel = databases.inversekinematics.InverseKinematicsModel(robot = robot, iktype = IkParameterization.Type.Transform6D)
	if not ikmodel.load():
		ikmodel.autogenerate()

	startingVertex = [startingNodeDOF, LA.norm(startingNodeDOF - targetNodeDOF)]
	targetVertex = [targetNodeDOF, 0.0]
	RRTVertices = [startingVertex]
	RRTEdges = []
	RRTGraph = [RRTVertices, RRTEdges]

	# check collision at starting configuration
	robot.SetDOFValues(startingNodeDOF, manip.GetArmIndices())
	if objGrasped != []:
		relativeMatrixRobotToObjGrasped = dot(inv(endTool.GetTransform()), objGrasped.GetTransform())
	for i_cllsnCheckObj in range(shape(cllsnCheckObjList)[0]):
		if objGrasped == []:
			cllsnWithObjList = env.CheckCollision(robot, cllsnCheckObjList[i_cllsnCheckObj])
		else:
			cllsnWithObjList = env.CheckCollision(robot, cllsnCheckObjList[i_cllsnCheckObj]) or env.CheckCollision(objGrasped, cllsnCheckObjList[i_cllsnCheckObj])
		if cllsnWithObjList:
			print "Collision with " + cllsnCheckObjList[i_cllsnCheckObj].GetName() + "at starting pose."
			pdb.set_trace() #breakpoint
			robot.SetDOFValues(robotArmInitDOF, manip.GetArmIndices())
			return 'null'
	# check collision at target configuration
	robot.SetDOFValues(targetNodeDOF, manip.GetArmIndices())
	if objGrasped != []:
		objGrasped.SetTransform(dot(endTool.GetTransform(), relativeMatrixRobotToObjGrasped))
	for i_cllsnCheckObj in range(shape(cllsnCheckObjList)[0]):
		if objGrasped == []:
			cllsnWithObjList = env.CheckCollision(robot, cllsnCheckObjList[i_cllsnCheckObj])
		else:
			cllsnWithObjList = env.CheckCollision(robot, cllsnCheckObjList[i_cllsnCheckObj]) or env.CheckCollision(objGrasped, cllsnCheckObjList[i_cllsnCheckObj])
		if cllsnWithObjList:
			print "Collision with " + cllsnCheckObjList[i_cllsnCheckObj].GetName() + "at target pose."
			pdb.set_trace() #breakpoint
			robot.SetDOFValues(robotArmInitDOF, manip.GetArmIndices())
			if objGrasped != []:
				objGrasped.SetTransform(dot(endTool.GetTransform(), relativeMatrixRobotToObjGrasped))
			return 'null'

	# check if starting node and target node are connectable
	[connectivityTwoNodes, VertexBeforeCollision, noNewNodeAdded] = checkConnectivityRRT(startingNodeDOF, targetNodeDOF, targetNodeDOF, robot, objGrasped, cllsnCheckObjList, transitTransferRRTGraphParameter, env)
	if connectivityTwoNodes:
		path = [startingNodeDOF, targetNodeDOF]
		robot.SetDOFValues(robotArmInitDOF, manip.GetArmIndices())
		if objGrasped != []:
			objGrasped.SetTransform(dot(endTool.GetTransform(), relativeMatrixRobotToObjGrasped))
		return [path, RRTGraph]
	else:
		if not noNewNodeAdded:
			# pdb.set_trace() #breakpoint
			RRTGraph[0].append(VertexBeforeCollision)
			RRTGraph[1].append([0, 1])
	# pdb.set_trace() #breakpoint

	# not directly connectable, sample new nodes
	for i_RRTIteration in range(int(numRRTIteration)):
		# pdb.set_trace()
		sampleIsTarget = False
		if np.random.uniform(0,1,1) < goalSampleRate:
			DOFRandom = targetNodeDOF
			sampleIsTarget = True
		else:
			DOFRandom = sampleJointValueInsideWorkSpaceRRT(robot, objGrasped, manipDOFLimit, cllsnCheckObjList, area, env)
		indexMin = findNearestIndexInRRTGraph(DOFRandom, RRTGraph)
		nodeNearest = RRTGraph[0][indexMin][0]
		# pdb.set_trace() #breakpoint
		[nodeNew, isDOFRandom] = giveNewNodeRRT(nodeNearest, DOFRandom, transitTransferRRTGraphParameter)

		[connectivityNearestToNodeNew, VertexBeforeCollision, noNewNodeAdded] = checkConnectivityRRT(nodeNearest, nodeNew, targetNodeDOF, robot, objGrasped, cllsnCheckObjList, transitTransferRRTGraphParameter, env)
		if not connectivityNearestToNodeNew:
			continue
		# pdb.set_trace() #breakpoint
		
		if sampleIsTarget and isDOFRandom: # the new node is target
			pdb.set_trace()
			path = givePathFromRRTGraph(RRTGraph, targetNodeDOF)
			robot.SetDOFValues(robotArmInitDOF, manip.GetArmIndices())
			if objGrasped != []:
				objGrasped.SetTransform(dot(endTool.GetTransform(), relativeMatrixRobotToObjGrasped))
			return [path, RRTGraph]
		else:
			distanceToTarget = LA.norm(targetNodeDOF - nodeNew)
			vertexNew = [nodeNew, distanceToTarget]
			RRTGraph[0].append(vertexNew)
			RRTGraph[1].append([indexMin, shape(RRTGraph[0])[0]-1])
		# if LA.norm(VertexBeforeCollision[0]-targetNodeDOF)<1e-3: # connected to target

		if np.random.uniform(0,1,1) < goalConnectionRate:
			[connectivityNodeNewToTarget, VertexBeforeCollision, noNewNodeAdded] = checkConnectivityRRT(nodeNew, targetNodeDOF, targetNodeDOF, robot, objGrasped, cllsnCheckObjList, transitTransferRRTGraphParameter, env)
			if connectivityNodeNewToTarget:
				# pdb.set_trace()
				path = givePathFromRRTGraph(RRTGraph, targetNodeDOF)
				robot.SetDOFValues(robotArmInitDOF, manip.GetArmIndices())
				if objGrasped != []:
					objGrasped.SetTransform(dot(endTool.GetTransform(), relativeMatrixRobotToObjGrasped))
				return [path, RRTGraph]
		
		# [connectivityNewToTargetNodes, VertexBeforeCollision, noNewNodeAdded] = checkConnectivityRRT(VertexBeforeCollision[0], targetNodeDOF, targetNodeDOF, robot, cllsnCheckObjList, transitTransferRRTGraphParameter, env)
		# if connectivityNewToTargetNodes:
			# pdb.set_trace() #breakpoint
		#	path = givePathFromRRTGraph(RRTGraph, targetNodeDOF)
		#	robot.SetDOFValues(robotArmInitDOF, manip.GetArmIndices())
		#	return [path, RRTGraph]
		# else:
		#	if not noNewNodeAdded:
		#		RRTGraph[0].append(VertexBeforeCollision)
		#		RRTGraph[1].append([shape(RRTGraph[0])[0]-2, shape(RRTGraph[0])[0]-1])
	robot.SetDOFValues(robotArmInitDOF, manip.GetArmIndices())
	if objGrasped != []:
		objGrasped.SetTransform(dot(endTool.GetTransform(), relativeMatrixRobotToObjGrasped))
	return 'null'

def	executeTransitTransferRRT(startingNodeDOF, targetNodeDOF, rrtPath, robot, objGrasped, cllsnCheckObjList, transitTransferRRTGraphParameter, area, env):
	if rrtPath == []:
		[path, RRTGraph] = transitTransferRRT(startingNodeDOF, targetNodeDOF, robot, objGrasped, cllsnCheckObjList, transitTransferRRTGraphParameter, area, env)
	else: 
		path = rrtPath
	smoothedPath = smoothPath(path, robot, objGrasped, cllsnCheckObjList, targetNodeDOF, transitTransferRRTGraphParameter, env)
	runPathRRT(robot, objGrasped, smoothedPath, maxDOFIncrementInDemo = 0.1)



def givePathFromRRTGraph(RRTGraph, targetNodeDOF):
	pathIDUnsorted = []
	pathUnsorted = []
	pathUnsorted.append([targetNodeDOF])
	numEdge = shape(RRTGraph[1])[0]
	startingNodeVec = np.zeros(shape(RRTGraph[1])[0])
	endNodeVec = np.zeros(shape(RRTGraph[1])[0])
	for i_edge in range(numEdge):
		startingNodeVec[i_edge] = RRTGraph[1][i_edge][0]
		endNodeVec[i_edge] = RRTGraph[1][i_edge][1]

	edge_Loop = RRTGraph[1][-1]	
	pathIDUnsorted.append(edge_Loop[1])
	# pdb.set_trace() #breakpoint
	pathUnsorted.append(RRTGraph[0][edge_Loop[1]][0:-1])
	while(True):

		# endNode_Loop = edge_Loop[1]
		startNode_Loop = edge_Loop[0]
		DOFStartNode_Loop = RRTGraph[0][startNode_Loop]
		pathIDUnsorted.append(startNode_Loop)
		# pdb.set_trace() #breakpoint
		pathUnsorted.append(DOFStartNode_Loop[0:-1])

		if startNode_Loop == 0:
			break
		indexEdge_Loop = np.where(endNodeVec == startNode_Loop)

		edge_Loop = RRTGraph[1][indexEdge_Loop[0]]
	pathID = pathIDUnsorted[::-1]
	path = pathUnsorted[::-1]
	# pdb.set_trace() #breakpoint
	return path

def giveNewNodeRRT(nodeNearest, DOFRandom, transitTransferRRTGraphParameter):
	expansionDistance = transitTransferRRTGraphParameter[3]
	distanceNearestAndRandom = LA.norm(nodeNearest - DOFRandom)
	isDOFRandom = False
	if distanceNearestAndRandom < expansionDistance:
		nodeNew = DOFRandom
		isDOFRandom = True
	else:
		nodeNew = expansionDistance/distanceNearestAndRandom*(DOFRandom-nodeNearest) + nodeNearest
	return [nodeNew, isDOFRandom]


def	sampleJointValueInsideWorkSpaceRRT(robot, objGrasped, manipDOFLimit, cllsnCheckObjList, area, env):
	initDOFTotal = robot.GetDOFValues()
	manip = robot.SetActiveManipulator("lwr")
	endTool = manip.GetEndEffector()
	if objGrasped != []:
		relativeMatrixRobotToObjGrasped = dot(inv(endTool.GetTransform()), objGrasped.GetTransform())
	manip_tool = manip.GetEndEffector()
	xMin = area[0] - (area[2]/2.0) * 1.2
	xMax = area[0] + area[2]/2.0
	yMin = area[1] - area[3]/2.0
	yMax = area[1] + area[3]/2.0
	zMin = area[4]
	zMax = zMin + 0.5
	numberOfJoints = len(manipDOFLimit[0])
	sampledJointValues = np.zeros([numberOfJoints])
	validSampling = False
	while(not validSampling):
		ifCollisionFreeWithObjList = True
		for i_joint in range(numberOfJoints):
			sampledJointValues[i_joint] = np.random.uniform(manipDOFLimit[0][i_joint], manipDOFLimit[1][i_joint], 1)
		robot.SetDOFValues(sampledJointValues, manip.GetArmIndices())
		if objGrasped != []:
			objGrasped.SetTransform(dot(endTool.GetTransform(), relativeMatrixRobotToObjGrasped))
		if env.GetBodies()[2].GetName() != 'table1':
			pdb.set_trace() #breakpoint			
		"""
		if objGrasped is not []:
			collisionWithTable1 = env.CheckCollision(robot, env.GetBodies()[2]) or env.CheckCollision(objGrasped, env.GetBodies()[2])
		else: 
			collisionWithTable1 = env.CheckCollision(robot, env.GetBodies()[2])
		"""
		xToolTransform = manip_tool.GetTransform()[0,3]
		yToolTransform = manip_tool.GetTransform()[1,3]
		zToolTransform = manip_tool.GetTransform()[2,3]
		ifInsideWorkspace = xToolTransform > xMin and xToolTransform < xMax and yToolTransform > yMin and yToolTransform < yMax and zToolTransform > zMin and zToolTransform < zMax
		for i_cllsnCheckObj in range(shape(cllsnCheckObjList)[0]):
			if objGrasped != []:
				cllsnWithObjList = env.CheckCollision(robot, cllsnCheckObjList[i_cllsnCheckObj]) or env.CheckCollision(objGrasped, cllsnCheckObjList[i_cllsnCheckObj])
			else: 
				cllsnWithObjList = env.CheckCollision(robot, cllsnCheckObjList[i_cllsnCheckObj])
			if cllsnWithObjList:
				"""
				robot.SetDOFValues(sampledJointValues, manip.GetArmIndices())
				if objGrasped is not []:
					objGrasped.SetTransform(dot(endTool.GetTransform(), relativeMatrixRobotToObjGrasped))
				"""
				ifCollisionFreeWithObjList = False
				break
		validSampling = ifInsideWorkspace and ifCollisionFreeWithObjList
		# print validSampling
		# pdb.set_trace() #breakpoint
	robot.SetDOFValues(initDOFTotal)
	if objGrasped != []:
		objGrasped.SetTransform(dot(endTool.GetTransform(), relativeMatrixRobotToObjGrasped))
	return sampledJointValues

def	checkConnectivityRRT(startingNodeDOF, endNodeDOF, targetNodeDOF, robot, objGrasped, cllsnCheckObjList, transitTransferRRTGraphParameter, env):
	maxCollisionCheckDOFIncrement = transitTransferRRTGraphParameter[2] # 0.1
	noNewNodeAdded = False
	connectivityTwoNodes = False
	manip = robot.SetActiveManipulator("lwr")
	endTool = manip.GetEndEffector()
	if objGrasped != []:
		relativeMatrixRobotToObjGrasped = dot(inv(endTool.GetTransform()), objGrasped.GetTransform())
	DOFBeforeCollision = endNodeDOF
	distanceToTarget = LA.norm(targetNodeDOF - DOFBeforeCollision)
	VertexBeforeCollision = [DOFBeforeCollision, distanceToTarget]
	samplingRadius = 0

	stepNum = 1.0
	while(True):
		diffDOF = (endNodeDOF - startingNodeDOF)/stepNum
		DOFIncrement = LA.norm(diffDOF)
		if DOFIncrement > maxCollisionCheckDOFIncrement:
			stepNum = stepNum + 1
		else: 
			break
	# pdb.set_trace() #breakpoint
	for i_step in range(int(stepNum)-1):
		crrtDOF = startingNodeDOF + (i_step + 1) * diffDOF
		robot.SetDOFValues(crrtDOF, manip.GetArmIndices())
		if objGrasped != []:
			objGrasped.SetTransform(dot(endTool.GetTransform(), relativeMatrixRobotToObjGrasped))
		# time.sleep(0.1)
		if objGrasped == []:
			cllsnFreeWithObjList = checkCollisionWithList(robot, cllsnCheckObjList, env) == []
		else:
			cllsnFreeWithObjList = checkCollisionWithList(robot, cllsnCheckObjList, env) == [] and checkCollisionWithList(objGrasped, cllsnCheckObjList, env) == []
		if cllsnFreeWithObjList: # no collision with the objects in list
			continue
		else:
			DOFBeforeCollision = startingNodeDOF + i_step * diffDOF
			distanceToTarget = LA.norm(targetNodeDOF - DOFBeforeCollision)
			VertexBeforeCollision = [DOFBeforeCollision, distanceToTarget]
			connectivityTwoNodes = False
			if i_step == 0:
				noNewNodeAdded = True
				robot.SetDOFValues(VertexBeforeCollision[0], manip.GetArmIndices())
				if objGrasped != []:
					objGrasped.SetTransform(dot(endTool.GetTransform(), relativeMatrixRobotToObjGrasped))
				return [connectivityTwoNodes, VertexBeforeCollision, noNewNodeAdded]
			robot.SetDOFValues(VertexBeforeCollision[0], manip.GetArmIndices())
			if objGrasped != []:
				objGrasped.SetTransform(dot(endTool.GetTransform(), relativeMatrixRobotToObjGrasped))
			return [connectivityTwoNodes, VertexBeforeCollision, noNewNodeAdded]
	connectivityTwoNodes = True
	robot.SetDOFValues(VertexBeforeCollision[0], manip.GetArmIndices())
	if objGrasped != []:
			objGrasped.SetTransform(dot(endTool.GetTransform(), relativeMatrixRobotToObjGrasped))
	return [connectivityTwoNodes, VertexBeforeCollision, noNewNodeAdded]
	
def findNearestIndexInRRTGraph(DOFRandom, RRTGraph):
	RRTVertices = RRTGraph[0]
	indexMin = 0
	distanceMin = 100
	for i_vertex in range(shape(RRTVertices)[0]):
		vertex_Loop = RRTVertices[i_vertex][0]
		distance_Loop = LA.norm(DOFRandom - vertex_Loop)
		if distance_Loop < distanceMin:
			indexMin = i_vertex
			distanceMin = distance_Loop
	return indexMin

def runPathRRT(robot, objGrasped, path, maxDOFIncrementInDemo):
	manip = robot.SetActiveManipulator("lwr")
	endTool = manip.GetEndEffector()
	if objGrasped != []:
		relativeMatrixRobotToObjGrasped = dot(inv(endTool.GetTransform()), objGrasped.GetTransform())
	for i_piece in range(shape(path)[0]-1):
		time.sleep(1)
		# pdb.set_trace() # breakpoint
		if shape(path)[0] <= 2:
			startingNodeDOF = path[i_piece]
			endNodeDOF = path[i_piece+1]
		else:
			startingNodeDOF = path[i_piece][0]
			endNodeDOF = path[i_piece+1][0]
		stepNum = 1.0
		while(True):
			diffDOF = (endNodeDOF - startingNodeDOF)/stepNum
			DOFIncrement = LA.norm(diffDOF)
			if DOFIncrement > maxDOFIncrementInDemo:
				stepNum = stepNum + 1
			else: 
				break
		for i_step in range(int(stepNum)):
			crrtDOF = startingNodeDOF + (i_step + 1) * diffDOF
			robot.SetDOFValues(crrtDOF, manip.GetArmIndices())
			if objGrasped != []:
					objGrasped.SetTransform(dot(endTool.GetTransform(), relativeMatrixRobotToObjGrasped))
			time.sleep(0.1)

def transitTransferRRTBall(startingNodeDOF, targetNodeDOF, robot, objGrasped, cllsnCheckObjList, transitTransferRRTGraphParameter, area, env):
	numRRTIteration = transitTransferRRTGraphParameter[0] # 100
	collisionBackResidual = transitTransferRRTGraphParameter[1] # 0.3
	maxCollisionCheckDOFIncrement = transitTransferRRTGraphParameter[2] # 0.1
	initBallRadius = transitTransferRRTGraphParameter[3]

	manip = robot.SetActiveManipulator("lwr")
	manipDOFLimitsMin = robot.GetDOFLimits()[0][manip.GetArmIndices()]
	manipDOFLimitsMax = robot.GetDOFLimits()[1][manip.GetArmIndices()]
	manipDOFLimit = [manipDOFLimitsMin, manipDOFLimitsMax]
	robotArmInitDOF = robot.GetDOFValues(manip.GetArmIndices())
	if env.GetBodies()[2].GetName() != 'table1':
		pdb.set_trace() #breakpoint
	table1_Obj = env.GetBodies()[2]
	ikmodel = databases.inversekinematics.InverseKinematicsModel(robot = robot, iktype = IkParameterization.Type.Transform6D)
	if not ikmodel.load():
		ikmodel.autogenerate()

	startingVertex = [startingNodeDOF, LA.norm(startingNodeDOF - targetNodeDOF)]
	targetVertex = [targetNodeDOF, 0.0]
	RRTVertices = [startingVertex]
	RRTEdges = []
	RRTGraph = [RRTVertices, RRTEdges]

	# check collision at starting configuration
	robot.SetDOFValues(startingNodeDOF, manip.GetArmIndices())
	for i_cllsnCheckObj in range(shape(cllsnCheckObjList)[0]):
		if env.CheckCollision(robot, cllsnCheckObjList[i_cllsnCheckObj]):
			print "Collision with " + cllsnCheckObjList[i_cllsnCheckObj].GetName() + "at starting pose."
			pdb.set_trace() #breakpoint
			robot.SetDOFValues(robotArmInitDOF, manip.GetArmIndices())
			return 'null'
	# check collision at target configuration
	robot.SetDOFValues(targetNodeDOF, manip.GetArmIndices())
	for i_cllsnCheckObj in range(shape(cllsnCheckObjList)[0]):
		if env.CheckCollision(robot, cllsnCheckObjList[i_cllsnCheckObj]):
			print "Collision with " + cllsnCheckObjList[i_cllsnCheckObj].GetName() + "at target pose."
			pdb.set_trace() #breakpoint
			robot.SetDOFValues(robotArmInitDOF, manip.GetArmIndices())
			return 'null'

	# check if starting node and target node are connectable
	# pdb.set_trace() #breakpoint
	[connectivityTwoNodes, VertexBeforeCollision2, noNewNodeAdded] = checkConnectivityRRT(startingNodeDOF, targetNodeDOF, targetNodeDOF, robot, objGrasped, cllsnCheckObjList, transitTransferRRTGraphParameter, env)
	if connectivityTwoNodes:
		path = [[startingNodeDOF], [targetNodeDOF]]
		robot.SetDOFValues(robotArmInitDOF, manip.GetArmIndices())
		return [path, RRTGraph]
	else:
		if not noNewNodeAdded:
			# pdb.set_trace() #breakpoint
			RRTGraph[0].append(VertexBeforeCollision2)
			RRTGraph[1].append([0, 1])
	# pdb.set_trace() #breakpoint



	# not directly connectable, sample new nodes
	ballRadius_Loop = initBallRadius
	ballCenter_Loop = VertexBeforeCollision2[0]
	for i_RRTIteration in range(int(numRRTIteration)):
		DOFRandom = sampleJointValueInsideWorkSpaceAndDOFBallRRT(robot, manipDOFLimit, ballCenter_Loop, ballRadius_Loop, cllsnCheckObjList, area, env)
		# pdb.set_trace() #breakpoint
		indexMin = findNearestIndexInRRTGraph(DOFRandom, RRTGraph)
		nodeNearest = RRTGraph[0][indexMin][0]
		[connectivityNearestToRandNodes, VertexBeforeCollision, noNewNodeAdded] = checkConnectivityRRT(nodeNearest, DOFRandom, targetNodeDOF, robot, objGrasped, cllsnCheckObjList, transitTransferRRTGraphParameter, env)
		distanceToTarget1 = VertexBeforeCollision[1]
		if noNewNodeAdded:
			continue
		RRTGraph[0].append(VertexBeforeCollision)
		RRTGraph[1].append([indexMin, shape(RRTGraph[0])[0]-1])
		[connectivityNewToTargetNodes, VertexBeforeCollision2, noNewNodeAdded] = checkConnectivityRRT(VertexBeforeCollision[0], targetNodeDOF, targetNodeDOF, robot, objGrasped, cllsnCheckObjList, transitTransferRRTGraphParameter, env)
		distanceToTarget2 = VertexBeforeCollision2[1]
		if connectivityNewToTargetNodes:
			# pdb.set_trace() #breakpoint
			path = givePathFromRRTGraph(RRTGraph, targetNodeDOF)
			robot.SetDOFValues(robotArmInitDOF, manip.GetArmIndices())
			return [path, RRTGraph]
		else:
			if not noNewNodeAdded: # added some node
				RRTGraph[0].append(VertexBeforeCollision2)
				RRTGraph[1].append([shape(RRTGraph[0])[0]-2, shape(RRTGraph[0])[0]-1])
				if distanceToTarget1 >= distanceToTarget2:
					ballCenter_Loop = VertexBeforeCollision2[0]
					ballRadius_Loop = initBallRadius
				else:
					ballCenter_Loop = nodeNearest
					ballRadius_Loop = ballRadius_Loop + ballRadius_Loop
			else: # no new node added 
				ballCenter_Loop = nodeNearest
				ballRadius_Loop = ballRadius_Loop + ballRadius_Loop
	robot.SetDOFValues(robotArmInitDOF, manip.GetArmIndices())
	return 'null'

def	sampleJointValueInsideWorkSpaceAndDOFBallRRT(robot, manipDOFLimit, ballCenterDOF, ballRadius, cllsnCheckObjList, area, env):
	manip = robot.SetActiveManipulator("lwr")
	manip_tool = manip.GetEndEffector()
	xMin = area[0] - (area[2]/2.0) * 1.2
	xMax = area[0] + area[2]/2.0
	yMin = area[1] - area[3]/2.0
	yMax = area[1] + area[3]/2.0
	zMin = area[4]
	zMax = zMin + 0.5
	numberOfJoints = len(manipDOFLimit[0])
	sampledJointValues = np.zeros([numberOfJoints])
	validSampling = False
	while(not validSampling):
		ifCollisionFreeWithObjList = True
		for i_joint in range(numberOfJoints):
			# sample inside the ball around every individual joint value of the ball center DOF
			sampledJointValues[i_joint] = np.random.uniform(max(manipDOFLimit[0][i_joint], ballCenterDOF[i_joint]-ballRadius), min(manipDOFLimit[1][i_joint], ballCenterDOF[i_joint]+ballRadius), 1)
		robot.SetDOFValues(sampledJointValues, manip.GetArmIndices())
		if env.GetBodies()[2].GetName() != 'table1':
				pdb.set_trace() #breakpoint
		collisionWithTable1 = env.CheckCollision(robot, env.GetBodies()[2])
		xToolTransform = manip_tool.GetTransform()[0,3]
		yToolTransform = manip_tool.GetTransform()[1,3]
		zToolTransform = manip_tool.GetTransform()[2,3]
		ifInsideWorkspace = xToolTransform > xMin and xToolTransform < xMax and yToolTransform > yMin and yToolTransform < yMax and zToolTransform > zMin and zToolTransform < zMax
		ifInsideBall = LA.norm(sampledJointValues - ballCenterDOF) <= ballRadius
		for i_cllsnCheckObj in range(shape(cllsnCheckObjList)[0]):
			if env.CheckCollision(robot, cllsnCheckObjList[i_cllsnCheckObj]):
				robot.SetDOFValues(sampledJointValues, manip.GetArmIndices())
				ifCollisionFreeWithObjList = False
				break
		validSampling = (not collisionWithTable1) and ifInsideWorkspace and ifInsideBall and ifCollisionFreeWithObjList
		# print validSampling
	# pdb.set_trace() #breakpoint
	return sampledJointValues

def smoothPath(path, robot, objGrasped, cllsnCheckObjList, targetNodeDOF, transitTransferRRTGraphParameter, env):
	manip = robot.SetActiveManipulator("lwr")
	manip_tool = manip.GetEndEffector()
	smoothingIsOver = False
	while(not smoothingIsOver):
		pathDeleted = False
		numPathPiece = shape(path)[0]
		if numPathPiece <= 3:
			break
		for i_startingNode in range(numPathPiece-2):

			for i_endNode in range(numPathPiece-1, i_startingNode+1, -1):

				startingNodeDOF = path[i_startingNode][0]
				endNodeDOF = path[i_endNode][0]
				[connectivityTwoNodes, VertexBeforeCollision2, noNewNodeAdded] = checkConnectivityRRT(startingNodeDOF, endNodeDOF, targetNodeDOF, robot, objGrasped, cllsnCheckObjList, transitTransferRRTGraphParameter, env)
				if (not connectivityTwoNodes) and (i_startingNode == numPathPiece-3) and (i_endNode==i_startingNode+2):
					smoothingIsOver = True
				if not connectivityTwoNodes:
					continue
				else:
					for i_del in range(i_endNode-i_startingNode-1):
						del path[i_startingNode+1]
					pathDeleted = True
					break

			if pathDeleted == True:
				break
	return path


#############################################################################################################################
# transit&transfer RRT (End)
#############################################################################################################################
def giveGraspTransformsFromObject(objectInFocus, env):
	graspTransform = []
	# pdb.set_trace() # breakpoint
	if objectInFocus.GetName() == 'fork' or objectInFocus.GetName() == 'kitchen_knife':
		zOffset = -0.06
		xOffset = 0.12
		numGraspEachSide = 10.0
		YLength = 0.27
		deltaYTotal = YLength*8.0/9.0
		deltaY = deltaYTotal/(numGraspEachSide-1)
		translationCrrtFrameXYZ = [xOffset, -deltaYTotal/2.0, zOffset]
		# grasp 1
		graspTransform1 = objectInFocus.GetTransform()
		graspTransform1 = dot(graspTransform1, giveRotationMatrix3D_4X4('z', -pi/2))
		graspTransform1 = dot(graspTransform1, giveRotationMatrix3D_4X4('y', -pi/2))
		graspTransform1Final = executeTranslationFromCrrtFrame(graspTransform1, translationCrrtFrameXYZ)
		graspTransform.append(graspTransform1Final)
		for i_grasp in range(int(numGraspEachSide-1)):
			graspTransform1FinalStep = executeTranslationFromCrrtFrame(graspTransform1Final, [0, deltaY*(i_grasp+1), 0])
			graspTransform.append(graspTransform1FinalStep)
		# grasp 2
		graspTransform2 = dot(graspTransform1, giveRotationMatrix3D_4X4('x', pi))
		graspTransform2Final = executeTranslationFromCrrtFrame(graspTransform2, translationCrrtFrameXYZ)
		graspTransform.append(graspTransform2Final)
		for i_grasp in range(int(numGraspEachSide-1)):
			graspTransform2FinalStep = executeTranslationFromCrrtFrame(graspTransform2Final, [0, deltaY*(i_grasp+1), 0])
			graspTransform.append(graspTransform2FinalStep)
		# h4 = [], h4.append(PlotFrame(env, graspTransform1Final, 0.05))
		# pdb.set_trace() #breakpoint
	elif objectInFocus.GetName() == 'kitchen_glass':
		graspTransform1 = objectInFocus.GetTransform()
		graspTransform1 = dot(graspTransform1, giveRotationMatrix3D_4X4('z', -pi/2))
		graspTransform1 = dot(graspTransform1, giveRotationMatrix3D_4X4('y', -pi/2))
		graspTransform1 = executeTranslationFromCrrtFrame(graspTransform1, [0.222,0,0])
		genGraspInCircle(totalGrasps = graspTransform, inputGrasp = graspTransform1, numGrasp = 32, radius = -0.05, radiusOffsetFromAxis = 'z', distributionAxis = 'x')
	elif objectInFocus.GetName() == 'kitchen_cup':
		graspTransform1 = objectInFocus.GetTransform()
		graspTransform1 = dot(graspTransform1, giveRotationMatrix3D_4X4('z', -pi/2))
		graspTransform1 = dot(graspTransform1, giveRotationMatrix3D_4X4('y', -pi/2))
		graspTransform1 = executeTranslationFromCrrtFrame(graspTransform1, [0.17,0,0])
		genGraspInCircle(totalGrasps = graspTransform, inputGrasp = graspTransform1, numGrasp = 32, radius = -0.05, radiusOffsetFromAxis = 'z', distributionAxis = 'x')
		# pdb.set_trace() #breakpoint
	elif objectInFocus.GetName() == 'mug':
		graspTransform1 = objectInFocus.GetTransform()
		graspTransform1 = dot(graspTransform1, giveRotationMatrix3D_4X4('z', -pi/2))
		graspTransform1 = dot(graspTransform1, giveRotationMatrix3D_4X4('y', -pi/2))
		graspTransform1 = executeTranslationFromCrrtFrame(graspTransform1, [0.19,0,0])
		genGraspInCircle(totalGrasps = graspTransform, inputGrasp = graspTransform1, numGrasp = 32, radius = -0.05, radiusOffsetFromAxis = 'z', distributionAxis = 'x')
		# pdb.set_trace() #breakpoint
	else:
		print 'no such object!'
	return graspTransform

def	genGraspInCircle(totalGrasps, inputGrasp, numGrasp, radius, radiusOffsetFromAxis, distributionAxis):
	if radiusOffsetFromAxis == 'x':
		offSetGrasp = [radius, 0, 0]
	elif radiusOffsetFromAxis == 'y':
		offSetGrasp = [0, radius, 0]
	else: 
		offSetGrasp = [0, 0, radius]
	diffRadian = 2*pi/numGrasp
	for i_grasp in range(numGrasp):
		intermediateTransform = dot(inputGrasp,  giveRotationMatrix3D_4X4(distributionAxis, diffRadian * i_grasp))
		totalGrasps.append(executeTranslationFromCrrtFrame(intermediateTransform, offSetGrasp))

def	executeTranslationFromCrrtFrame(crrtTransform, translationCrrtFrameXYZ):
	transformMatrix = np.eye(4)
	transformMatrix[0:3,3] = translationCrrtFrameXYZ
	crrtTransform = dot(crrtTransform, transformMatrix)
	return crrtTransform

def runDemo(robot, path, transitPRMGraph, numInterpolation):
	manip = robot.SetActiveManipulator("lwr")
	manip_tool = manip.GetEndEffector()
	for i_piece in range(shape(path)[0]-1):
		DOFVertex1 = transitPRMGraph[0][path[i_piece]]
		DOFVertex2 = transitPRMGraph[0][path[i_piece+1]]
		diffDOF = (DOFVertex2 - DOFVertex1)/(numInterpolation+1)
		for i_interpolator in range(int(numInterpolation+2)):
			crrtDOF = DOFVertex1 + diffDOF * i_interpolator
			robot.SetDOFValues(crrtDOF, manip.GetArmIndices())
			time.sleep(0.2)
# i_interpolator = i_interpolator +1; crrtDOF = DOFVertex1 + diffDOF * i_interpolator; robot.SetDOFValues(crrtDOF, manip.GetArmIndices()); print i_interpolator


"""
RRT implementation
def sampleNodeNorm(area,objectList):
	numObj = shape(objectList)[0]
	randNode = []
	randNodeNorm = []
	for i_object in range(numObj):
		xRand = np.random.uniform(area[0] - area[2]/2, area[0] + area[2]/2, 1)
		yRand = np.random.uniform(area[1] - area[3]/2, area[1] + area[3]/2, 1)
		rotZRand = np.random.uniform(0, 360, 1)
		randNode.append([xRand, yRand, rotZRand])
		xRandNorm = (xRand - area[0] + area[2]/2)/area[2]
		yRandNorm = (yRand - area[1] + area[3]/2)/area[3]
		rotZRandNorm = rotZRand/360
		randNodeNorm.append([xRandNorm, yRandNorm, rotZRandNorm])

	return [randNode, randNodeNorm]

def extendTreeNode(Tree, TreeNorm, randNode, randNodeNorm)
	[nearestNodeNorm, nearestNodeIndex] = giveNearestNodeNorm(TreeNorm, randNodeNorm)

	return newTreeNode

def normalizeNode(node, area):
	numObj = shape(objectList)[0]
	nodeNorm = np.zeros(shape(node))
	for i_object in range(numObj):
		nodeNorm[i_object][0] = (node[i_object][0] + area[0])/area[2]
		nodeNorm[i_object][1] = (node[i_object][1] + area[1])/area[3]
		nodeNorm[i_object][2] = node[i_object][2]/360
	return nodeNorm

def giveNearestNodeNorm(TreeNorm, randNodeNorm):
	numNode = shape(TreeNorm)[0]
	diffBetNode = []
	for i_Node in range(numNode):
		nodeInLoop = TreeNorm[i_Node]
		diffBetNodeInLopp = np.linalg.norm(nodeInLoop - randNodeNorm)
		diffBetNode.append(diffBetNodeInLopp)
	nearestNodeIndex = np.argmin(diffBetNode)
	nearestNodeNorm = TreeNorm(nearestNodeIndex)
	return [nearestNodeNorm, nearestNodeIndex]
"""


