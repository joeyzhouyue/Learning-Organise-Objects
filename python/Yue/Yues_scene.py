from openravepy import *
from types import *
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


env = Environment() # create the environment
env.SetDebugLevel(DebugLevel.Fatal)

env.SetViewer('qtcoin') # start the viewer

env.Load('../../robots/Kitchen_LASA/Scene_Kitchen_without_objects_LASA.env.xml') # load a scene
robot = env.GetRobots()[0]
manipTool_Transform = robot.GetLinks()[15].GetTransform()
handle = []
handle.append(PlotFrame(env, manipTool_Transform, 0.1))
tableObj1 = env.GetBodies()[3]
tableObj2 = env.GetBodies()[4]
# pdb.set_trace() #breakpoint
length_table1 = 2.4
width_table1 = 1.2
height_table1 = 1.4
length_table2 = 2
width_table2 = 2
height_table2 = 1.4
table1 = (tableObj1, length_table1, width_table1, height_table1)
table2 = (tableObj2, length_table2, width_table2, height_table2)
#############################################################################################################
# generate random placement of objects
# with env:
objectList = []
giveGraspPosOri = []
env.Load('../../robots/Kitchen_LASA/Kitchen_glass.kinbody.xml')
objectList.append([env.GetBodies()[5],'glass', giveGraspPosOri])


env.Load('../../robots/Kitchen_LASA/Kitchen_olive_oil_bottle.kinbody.xml')
objectList.append([env.GetBodies()[6],'oliveOilBottle', giveGraspPosOri])

# env.Load('../../robots/Kitchen_LASA/Kitchen_wine_bottle.kinbody.xml')
# objectList.append([env.GetBodies()[6],'wineBottle'])

# env.Load('../../robots/Kitchen_LASA/Kitchen_plate2.kinbody.xml')
# objectList.append([env.GetBodies()[5],'plate2'])


env.Load('../../robots/Kitchen_LASA/Kitchen_knife.kinbody.xml')
objectList.append([env.GetBodies()[7],'knife', giveGraspPosOri])

env.Load('../../robots/Kitchen_LASA/Kitchen_cup.kinbody.xml')
objectList.append([env.GetBodies()[8],'cup', giveGraspPosOri])

#env.Load('../../robots/Kitchen_LASA/Kitchen_drink_bottle.kinbody.xml')
#objectList.append([env.GetBodies()[8],'drinkBottle'])

#pdb.set_trace() #breakpoint
for i_item in range(shape(objectList)[0]):
  newTransform = gerateRandomPositionObjects(table1, table2, objectList[i_item][0])
  thereIsCollision = True
  while(thereIsCollision):
    objectList[i_item][0].SetTransform(newTransform)
    env.UpdatePublishedBodies()
    thereIsCollision = checkCollision(objectList[i_item][0], env)
    if not thereIsCollision:
      handle.append(PlotFrame(env, objectList[i_item][0].GetTransform(), 0.05))
      newTransform = gerateRandomPositionObjects(table1, table2, objectList[i_item][0])
      objectList[i_item][2] = giveGraspPosOriFromObj(objectList[i_item][0:2], handle, env)
      env.UpdatePublishedBodies()
############################################################################################################
# generate grid point on the ground
pdb.set_trace() #breakpoint

stepNumberX = 100
stepNumberY = 100
# infocylinder = KinBody.Link.GeometryInfo()
# table1_GeometryInfo = env.GetBodies()[3].GeometryInfo()
# table2_KinBody = env.GetBodies()[4]
stepLengthX = 10.0/stepNumberX
stepLengthY = 10.0/stepNumberY
X0 = -5 + stepLengthX/2
Y0 = -5 + stepLengthY/2
groundReachabilityGridInfo = np.zeros([stepNumberX,stepNumberY]) # if the robot can move to this piece of land
objectReachabilityGridInfo = np.zeros([stepNumberX,stepNumberY]) # if the robot can reach any object on fixed base position
#with env:
for i_x in range(stepNumberX):
  for i_y in range(stepNumberY):
    # if i_x == 13 and i_y == 18: pdb.set_trace() #breakpoint
    # if i_x == 10 and i_y == 18: pdb.set_trace() #breakpoint
    # time.sleep(0.1)
    robotBasePosX = X0 + i_x * stepLengthX
    robotBasePosY = Y0 + i_y * stepLengthY
    robotBasePos = robot.GetTransform()
    robotBasePos[0:2,3] = array([robotBasePosX, robotBasePosY])
    robot.SetTransform(robotBasePos)
    # pdb.set_trace() #breakpoint
    if checkCollision(robot, env) or math.fabs(robotBasePosX) > 4.5 or math.fabs(robotBasePosY) > 4.5 : # there is collision between the robot and other objects, shows red point
      handle.append(env.plot3(points=robot.GetTransform()[0:3,3], pointsize=0.05, colors=array(((1,0,0))),drawstyle = 1))
    else: # there is no collision between the robot and other objects till now
      groundReachabilityGridInfo[i_x,i_y] = 1
      if canReachAnyObjectOnTable(robot, objectList, env):
        objectReachabilityGridInfo[i_x,i_y] = 1
        handle.append(env.plot3(points=robot.GetTransform()[0:3,3], pointsize=0.05, colors=array(((0,0,1))),drawstyle = 1))
      else:
        handle.append(env.plot3(points=robot.GetTransform()[0:3,3], pointsize=0.05, colors=array(((0,1,0))),drawstyle = 1))

pdb.set_trace() #breakpoint
#------------------------------------------------------------------------------------------------
# maybe useful
# manipprob.MoveManipulator(goal=[-0.75,1.24,-0.064,2.33,-1.16,-1.548,1.19]) # call motion planner with goal joint angles
# res = manipprob.MoveToHandPosition(matrices=[Tgoal],seedik=10) # with end effector
#--------------------------------------------------------------------------------------------------



# handle = env.RegisterCollisionCallback(collisioncallback)
# report1 = CollisionReport()

handles.append(PlotFrame(env, robot.GetTransform(), 0.5)) # robot base
floorCenter = env.GetBodies()[1]
handles.append(PlotFrame(env, floorCenter.GetTransform(), 0.5)) # floor center


# glass = env.GetBodies()[5]


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







obj_in_focus = env.GetBodies()[13] # the plate 1
transform_old = obj_in_focus.GetTransform() # location before transform
transform_new = array([[ 0.38676739, -0.04232826, -0.92120535,  0.03661492],
 [-0.45307359,  0.86134552, -0.22980039,  0.46699503],
 [ 0.80320315,  0.50625312,  0.31396254,  2.12730765],
 [ 0.        ,  0.        ,  0.        ,  1.        ]])
obj_in_focus.SetTransform(transform_new)
objFrame = obj_in_focus.GetTransform()
framescale=0.15

To_go = array([[  0.00000000e+00,   1.00000000e+00,  -6.84852132e-22,
  1.26753521e-05],
  [ -1.00000000e+00,   0.00000000e+00,   0.00000000e+00,
  2.15658039e-01],
  [  0.00000000e+00,   6.84852132e-22,   1.00000000e+00,
  2.28416693e+00],
  [  0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
  1.00000000e+00]])


htg1,htg2,htg3= PlotFrame(env, To_go, framescale)



# move the manipulator

manip = robot.SetActiveManipulator("lwr")

print manip.GetArmIndices()

manipBase_Transform = robot.GetLinks()[1].GetTransform()
manipTool_Transform = robot.GetLinks()[15].GetTransform()
h1,h2,h3 = PlotFrame(env, manipBase_Transform, 0.2)
h4,h5,h6 = PlotFrame(env, manipTool_Transform, 0.2)
#handles.append(h1)
#handles.append(h2)
#handles.append(h3)


pdb.set_trace() #breakpoint


ikmodel = databases.inversekinematics.InverseKinematicsModel(robot=robot, freeindices=[4], iktype=IkParameterization.Type.Transform6D)
if not ikmodel.load():
 ikmodel.autogenerate()
#lower,upper = [v[ikmodel.manip.GetArmIndices()] for v in ikmodel.robot.GetDOFLimits()]
#robot.SetDOFValues(random.rand()*(upper-lower)+lower,ikmodel.manip.GetArmIndices()) # set random values
#if not robot.CheckSelfCollision():

sol_DOF = manip.FindIKSolution(To_go,IkFilterOptions.IgnoreJointLimits)
CurrJntDOF = robot.GetActiveDOFValues()[0:7]
diffDOF = sol_DOF-CurrJntDOF
lastStep = 0;

while LA.norm(diffDOF) > 0.25:
  #print LA.norm(diffDOF)
  cmd = CurrJntDOF + diffDOF * 0.1
  robot.SetDOFValues(cmd, manip.GetArmIndices())
  env.UpdatePublishedBodies()
  time.sleep(0.1)
  CurrJntDOF = robot.GetActiveDOFValues()[0:7]
  diffDOF = sol_DOF - CurrJntDOF

  robot.SetDOFValues(sol_DOF, manip.GetArmIndices())

  env.UpdatePublishedBodies() 

  raw_input('press enter to finish the game')
pdb.set_trace() #breakpoint
