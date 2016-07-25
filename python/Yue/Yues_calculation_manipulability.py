from openravepy import *
from types import *
from datetime import datetime
import random
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
  sys.path.insert(0, '/home/yue/openrave/python/examples')
  from simplegrasping import *
# import for plot------------------------------------------------------------
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib import cm, colors, ticker
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib.mlab import bivariate_normal
import matplotlib.pyplot as plt
# import for plot------------------------------------------------------------






#############################################################################################################
# set up environment
#############################################################################################################
env = Environment() # create the environment
env.SetDebugLevel(DebugLevel.Fatal)
env.SetViewer('qtcoin') # start the viewer
# pdb.set_trace() #breakpoint
env.Load('../../robots/Kitchen_LASA/Scene_Kitchen_sampling_manipulability.env.xml') # load a scene
# robot
robot = env.GetRobots()[0]
robotRangeMax = 0.86 # maximum robot arm range in X-Y plane, from base origin to wrist
robotBasePosX_initi = 3
robotBasePosY_initi = 3
robotBasePos_initi = robot.GetTransform()
robotBasePos_initi[0:2,3] = array([robotBasePosX_initi, robotBasePosY_initi])
robot.SetTransform(robotBasePos_initi)
manipTool_Transform = robot.GetLinks()[15].GetTransform()
# generate inverse kinematics solver
ikmodel = databases.inversekinematics.InverseKinematicsModel(robot=robot, iktype=IkParameterizationType.Transform6D)
if not ikmodel.load():
  ikmodel.autogenerate()

# handle = []
# handle.append(PlotFrame(env, manipTool_Transform, 0.1))
# pdb.set_trace() #breakpoint

# tables
tableObj = env.GetBodies()[2]
length_table = 1.44 # x direction
width_table = 1.44 # y direction
height_table = 1.23
handle = []
# handle.append(env.drawlinestrip(points=array(((-2,0,height_table),(2,0,height_table))), linewidth=1.0, colors=array(((0.81,0.12,0.56),(0.81,0.12,0.56)))))
# handle.append(env.drawlinestrip(points=array(((0,2,height_table),(0,-2,height_table))), linewidth=1.0, colors=array(((0.81,0.12,0.56),(0.81,0.12,0.56)))))

# pdb.set_trace() #breakpoint

#############################################################################################################
# generate objects
#############################################################################################################
# Olive oil bottle
env.Load('../../robots/Kitchen_LASA/Kitchen_olive_oil_bottle.kinbody.xml')
obj = env.GetBodies()[-1]
numberSamplesInX = 50
numberSamplesInY = 50
xStep = length_table/numberSamplesInX
yStep = width_table/numberSamplesInY
objPos = []

X = np.arange(-length_table/2, length_table/2+xStep-0.001, xStep)
Y = np.arange(-width_table/2, width_table/2+yStep-0.001, yStep)
X, Y = np.meshgrid(X, Y)
numberSamplesTotal = shape(X)[0] * shape(X)[1]
computationCost1 = np.zeros(shape(X))
computationCost2 = np.zeros(shape(X))  
objPosOri = obj.GetTransform()
# pdb.set_trace() #breakpoint
for i_x in range(numberSamplesInX+1):
  for i_y in range(numberSamplesInY+1):
    x_thisExper = round(X[i_y][i_x],3)
    y_thisExper = round(Y[i_y][i_x],3)

    # pdb.set_trace() #breakpoint
    objPos.append((x_thisExper,y_thisExper))
    objPosOri[0:3,3] = array([x_thisExper, y_thisExper, height_table])
    obj.SetTransform(objPosOri)
    # calculate the computation time
    timeCalcBegin = datetime.now()
    objChosenPosXY = obj.GetTransform()[0:2,3]
    # pdb.set_trace() #breakpoint
    # with env begins
    pointsMovableAroundObjChosen, pointsHandle1 = generateMovablePointAreaAroundAnObj(center = objChosenPosXY, radius = robotRangeMax, granularityIndex = 10, robot = robot, env = env, typeChosen = 'object')
    #raw_input('Plot the reachable points on the ground without collision.')
    del pointsHandle1
    pointWithHighManipXY, pointManipulabilityHandle = findPointWithHighManipXY(robot, givenPointsMovable = pointsMovableAroundObjChosen, obj = obj, env = env, plotheight = 0.01, plotsSize = 0.03)  
    # with env ends
    timeCalcEnd = datetime.now()
    durationCalc1 = timeCalcEnd - timeCalcBegin
    if durationCalc1 == 0:
      pdb.set_trace() #breakpoint
    computationCost1[i_y][i_x] = durationCalc1.total_seconds()
    print "Robot used " + str(durationCalc1.total_seconds()) + " seconds to find the point with the highest XY manipulability in the first grob grid search."
    #raw_input('Now the robot calculates finer granularity of the grid around the point.\n')
    timeCalcBegin = datetime.now()
    # pdb.set_trace() #breakpoint
    # with env begins
    pointsMovableAroundPointChosen, pointsHandle2 = generateMovablePointAreaAroundAnObj(center = pointWithHighManipXY, radius = robotRangeMax/2, granularityIndex = 10, robot = robot, env = env, typeChosen = 'point')
    #raw_input('Plot the reachable points on the ground without collision in finer grid.')
    del pointsHandle2
    pointWithHighManipXY2, pointManipulabilityHandle2 = findPointWithHighManipXY(robot, givenPointsMovable = pointsMovableAroundPointChosen, obj = obj, env = env, plotheight = 0.2,  plotsSize = 0.02)
    # with env ends 
    timeCalcEnd = datetime.now()
    durationCalc2 = timeCalcEnd - timeCalcBegin
    computationCost2[i_y][i_x] = durationCalc2.total_seconds()
    print "Robot used " + str(durationCalc2.total_seconds()) + " seconds to find the point with the highest XY manipulability in the second grob grid search."
pdb.set_trace() #breakpoint

# save the data

# save the configuration data
with open('computingCostX.pickle','w') as ff: 
  pickle.dump(X, ff)
with open('computingCostY.pickle','w') as fff: 
  pickle.dump(Y, fff)
with open('computingCostXcomputationCost1.pickle','w') as ffff: 
  pickle.dump(computationCost1, ffff)
with open('computingCostXcomputationCost2.pickle','w') as fffff: 
  pickle.dump(computationCost2, fffff)

# plot 3D
"""
fig = plt.figure(figsize=plt.figaspect(0.5))

ax = fig.add_subplot(1, 2, 1, projection='3d')
surf = ax.plot_surface(X, Y, computationCost1, rstride=1, cstride=1, cmap=cm.coolwarm,linewidth=0, antialiased=False, label ='computationCost1')
ax.set_zlim3d(0, 10)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
fig.colorbar(surf, shrink=0.5, aspect=5)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('computationCost1')
ax.legend()

ax = fig.add_subplot(1, 2, 2, projection='3d')
surf = ax.plot_surface(X, Y, computationCost2, rstride=1, cstride=1, cmap=cm.coolwarm,linewidth=0, antialiased=False, label ='computationCost1', interpolation=nearest)
ax.set_zlim3d(0, 10)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
fig.colorbar(surf, shrink=0.5, aspect=5)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('computationCost2')
ax.legend()

plt.show()
"""
#  plot 2D

fig = plt.figure(figsize=plt.figaspect(0.5))
ax0 = fig.add_subplot(1, 2, 1)
surf = ax0.contourf(X, Y, computationCost1, cmap=cm.PiYG)
# ax0.set_zlim(0, 10)
ax0.set_xlabel('X')
ax0.set_ylabel('Y')
# fig.colorbar(surf, ax=ax0)
ax0.set_title('computation Cost 1')
fig.colorbar(surf, ax=ax0)
# plt.colorbar()

ax1 = fig.add_subplot(1, 2, 2)
surf2 = ax1.contourf(X, Y, computationCost2, cmap=cm.PiYG)
# ax1.set_zlim(0, 10)
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_title('computation Cost 2')
fig.colorbar(surf2, ax=ax1)
# fig.tight_layout()

plt.show()


