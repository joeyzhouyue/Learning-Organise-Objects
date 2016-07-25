from openravepy import *
from types import *
from datetime import datetime
import IPython
import random
import math, time
import numpy as np
import sys
sys.path.insert(0, '../')
sys.path.insert(0, 'Yues_class_definition')
from Yues_GUI_module import *
from openravepy import misc
from scipy import linalg
# if not __openravepy_build_doc__:
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
from Yues_functions import * 
import pdb
sys.path.insert(0, '/home/yue/openrave/python/examples')
from simplegrasping import *

#############################################################################################################
# set up environment
#############################################################################################################
pdb.set_trace() #breakpoint
env = Environment() # create the environment
env.SetDebugLevel(DebugLevel.Fatal)
env.SetViewer('qtcoin') # start the viewer
# env.SetViewer('RViz')
# IPython.embed()
# env.SetViewer('InteractiveMarker')
# pdb.set_trace() #breakpoint
env.Load('../../robots/Kitchen_LASA/Scene_Kitchen_demon.env.xml') # load a scene
# env.Load('../../robots/Kitchen_LASA/Kitchen_Robot.robot.xml') # load a scene

# pdb.set_trace() #breakpoint

# robot
robot = env.GetRobots()[0]
manip = robot.SetActiveManipulator("lwr")
manip_tool = manip.GetEndEffector()
robotRangeMax = 0.86 # maximum robot arm range in X-Y plane, from base origin to wrist
robotBasePosX_initi = -1
robotBasePosY_initi = 3
robotBasePos_initi = robot.GetTransform()
robotBasePos_initi[0:2,3] = array([robotBasePosX_initi, robotBasePosY_initi])
robot.SetTransform(robotBasePos_initi)
manipTool_Transform = manip_tool.GetTransform()
# ManipVecXYZHandleOri = drawManipXYZ(robot,manip_tool,env)

#handle = []
#handle.append(PlotFrame(env, manipTool_Transform, 0.1))
# tables
tableObj1 = env.GetBodies()[3]
tableObj2 = env.GetBodies()[4]
length_table1 = 2.4
width_table1 = 1.2
height_table1 = 1.4
length_table2 = 2
width_table2 = 2
height_table2 = 1.4
table1 = (tableObj1, length_table1, width_table1, height_table1)
table2 = (tableObj2, length_table2, width_table2, height_table2)


#############################################################################################################
# generate objects
#############################################################################################################
objectList = []
giveGraspPosOri = []
# 1. glass
env.Load('../../robots/Kitchen_LASA/Kitchen_glass.kinbody.xml')
obj = env.GetBodies()[9]
objectList.append([obj,'glass', giveGraspPosOri])
objPosOri = obj.GetTransform()
objPosOri[0:3,3] = array([2.32, 0.76, 0.93])
obj.SetTransform(objPosOri)
# 2. olive oil bottle
env.Load('../../robots/Kitchen_LASA/Kitchen_olive_oil_bottle.kinbody.xml')
obj = env.GetBodies()[10]
objectList.append([obj,'oliveOilBottle', giveGraspPosOri])
objPosOri = obj.GetTransform()
objPosOri[0:3,3] = array([1.7, -0.25, 1.23])

obj.SetTransform(objPosOri)
# 3. knife
env.Load('../../robots/Kitchen_LASA/Kitchen_knife.kinbody.xml')
obj = env.GetBodies()[11]
objectList.append([obj,'knife', giveGraspPosOri])
objPosOri = obj.GetTransform()
objPosOri[0:3,3] = array([2, 1.4, 0.922])
obj.SetTransform(objPosOri)
# 4. cup
env.Load('../../robots/Kitchen_LASA/Kitchen_cup.kinbody.xml')
obj = env.GetBodies()[12]
objectList.append([obj,'cup', giveGraspPosOri])
objPosOri = obj.GetTransform()
objPosOri[0:3,3] = array([1.9, -1.2, 1.23])
obj.SetTransform(objPosOri)
# 5. hook
env.Load('../../robots/Kitchen_LASA/Kitchen_hook.kinbody.xml')
obj = env.GetBodies()[13]
objectList.append([obj,'hook', giveGraspPosOri])
objPosOri = obj.GetTransform()
objPosOri[0:3,3] = array([-2.9, 1.37, 1.16])
obj.SetTransform(objPosOri)
# 6. ladle
env.Load('../../robots/Kitchen_LASA/Kitchen_ladle.kinbody.xml')
obj = env.GetBodies()[14]
objectList.append([obj,'ladle', giveGraspPosOri])
objPosOri = obj.GetTransform()
objPosOri[0:3,3] = array([-2, -0.35, 1.58])
obj.SetTransform(objPosOri)
# 7. mug
env.Load('../../robots/Kitchen_LASA/Kitchen_mug.kinbody.xml')
obj = env.GetBodies()[15]
objectList.append([obj,'mug', giveGraspPosOri])
objPosOri = obj.GetTransform()
objPosOri[0:3,3] = array([2.3, 1.32, 0.93])
obj.SetTransform(objPosOri)
# 8. pasta can
env.Load('../../robots/Kitchen_LASA/Kitchen_pasta_can.kinbody.xml')
obj = env.GetBodies()[16]
objectList.append([obj,'pasta_can', giveGraspPosOri])
objPosOri = obj.GetTransform()
objPosOri[0:3,3] = array([-2.57, -0.94, 1.77])
obj.SetTransform(objPosOri)
# 9. plate
env.Load('../../robots/Kitchen_LASA/Kitchen_plate2.kinbody.xml')
obj = env.GetBodies()[17]
objectList.append([obj,'plate', giveGraspPosOri])
objPosOri = obj.GetTransform()
objPosOri[0:3,3] = array([2.07, 1, 0.93])
obj.SetTransform(objPosOri)
# 10. wine bottle
env.Load('../../robots/Kitchen_LASA/Kitchen_wine_bottle.kinbody.xml')
obj = env.GetBodies()[18]
objectList.append([obj,'wine_bottle', giveGraspPosOri])
objPosOri = obj.GetTransform()
objPosOri[0:3,3] = array([-0.16, -2.43, 1.51])
obj.SetTransform(objPosOri)
raw_input('The robot is facing a kitchen to be cleaned up.')




# generate inverse kinematics solver
ikmodel = databases.inversekinematics.InverseKinematicsModel(robot=robot, iktype=IkParameterizationType.Transform6D)
if not ikmodel.load():
  ikmodel.autogenerate()
raw_input('Assume that we could generate a plan to organise the kitchen in a smart fashion.')


#############################################################################################################
# navigate to table/object
#############################################################################################################
raw_input('In this demo, the robot decides to handle the olive oil bottle first.')
objChosen = env.GetBodies()[10] # olive oil bottle
raw_input('Show the direction of the robot movement.')
rangeHandle1 = drawPathRangeToObj(robot, objPosXY = objChosen.GetTransform()[0:2,3], robotRangeMax = robotRangeMax, env = env)
raw_input('Then the robot begins to move towards the object.')
posRobot = robot.GetTransform()[0:3,3]
pathPointHandle = slowlyMoveToAPoint(robot, destination = objChosen.GetTransform()[0:2,3], segmentOfRoute = 1./3, stepNumber = 10, env = env, drawOrNotDraw = 'draw', withOrWithoutTimeSleep = 'with') 
# pathPointHandle = slowlyMoveToAPoint(robot, destination = objChosen.GetTransform()[0:2,3], 1./3, 10, env) 
raw_input('In the meantime, the robot tries to find the position near the object with the highest horizontal manipulability.')
# calculate the computation time
timeCalcBegin = datetime.now()
objChosenPosXY = objChosen.GetTransform()[0:2,3]
# pdb.set_trace() #breakpoint
# with env:
pointsMovableAroundObjChosen, pointsHandle1 = generateMovablePointAreaAroundAnObj(center = objChosenPosXY, radius = robotRangeMax, granularityIndex = 10, robot = robot, env = env, typeChosen = 'object', pointsType = 'nearRob')
raw_input('Plot the reachable points on the ground without collision.')
del pointsHandle1
pointWithHighManipXY, pointManipulabilityHandle = findPointWithHighManipXY(robot, givenPointsMovable = pointsMovableAroundObjChosen, obj = objChosen, env = env, plotheight = 0.01, plotsSize = 0.03)  
# with env ends
timeCalcEnd = datetime.now()
durationCalc1 = timeCalcEnd - timeCalcBegin
print "Robot used " + str(durationCalc1.total_seconds()) + " seconds to find the point with the highest XY manipulability in the first grob grid search."
raw_input('Then the robot begins to move towards that position.')
pathPointHandle2 = slowlyMoveToAPoint(robot, destination = pointWithHighManipXY, segmentOfRoute = 1./2, stepNumber = 10, env = env, drawOrNotDraw = 'draw',  withOrWithoutTimeSleep = 'with')
raw_input('Now the robot calculates finer granularity of the grid around the point.\n')
# pdb.set_trace() #breakpoint
timeCalcBegin = datetime.now()
# pdb.set_trace() #breakpoint
# with env begins
pointsMovableAroundPointChosen, pointsHandle2 = generateMovablePointAreaAroundAnObj(center = pointWithHighManipXY, radius = robotRangeMax/2, granularityIndex = 10, robot = robot, env = env, typeChosen = 'point', pointsType = 'nearRob')
raw_input('Plot the reachable points on the ground without collision in finer grid.')
del pointsHandle2
pointWithHighManipXY2, pointManipulabilityHandle2 = findPointWithHighManipXY(robot, givenPointsMovable = pointsMovableAroundPointChosen, obj = objChosen, env = env, plotheight = 0.1,  plotsSize = 0.02)
# with env ends

timeCalcEnd = datetime.now()
durationCalc2 = timeCalcEnd - timeCalcBegin
print "Robot used " + str(durationCalc2.total_seconds()) + " seconds to find the point with the highest XY manipulability in the second grob grid search."
raw_input('Now the robot begins to move towards that position and arrives there.')
pathPointHandle3 = slowlyMoveToAPoint(robot, destination = pointWithHighManipXY2, segmentOfRoute = 1, stepNumber = 15, env = env, drawOrNotDraw = 'draw',  withOrWithoutTimeSleep = 'with')
raw_input('After arrival, the robot chooses that grasp configuration with the highest Z manipulability.')
with env:
  grapsTranf, solDOF, graspManipulabilityHandle = findGraspWithHighManipZ(robot, givenRobPos = pointWithHighManipXY2, obj = objChosen, env = env)
#############################################################################################################
# grasp the object
#############################################################################################################
raw_input('The robot moves the arm to that grasp configuration.')
# pdb.set_trace() #breakpoint
ifPrintManipulabilityEllipsoid = True
if ifPrintManipulabilityEllipsoid:
	drawTimes = 5
	slowlyMoveManipToAnObjWithManipulability(robot, solDOF, env, drawTimes)
else:
	slowlyMoveManipToAnObj(robot, solDOF, env)

raw_input('The robot closes the fingers to grasp the object.')
slowlyGraspAnObjForDemo(robot, env)
raw_input('Show the end-effector manipulability in x,y,z directions.')
# ManipVecXYZHandle = drawManipXYZ(robot,manip_tool,env,drawtype = 'semiaxes')
# del ManipVecXYZHandle
# pdb.set_trace() #breakpoint
raw_input('Now the robot has the object in the hand.')
del rangeHandle1, pathPointHandle, pathPointHandle2, pathPointHandle3, pointManipulabilityHandle, pointManipulabilityHandle2, graspManipulabilityHandle
#############################################################################################################
# move object 
#############################################################################################################
raw_input('Some samples from previous experiences on placement.')
# distribution 1
samplePosition1 = [-2.3, -1.14, 1.48]
covMatrix1 = [[0.02, -0.01], [-0.01, 0.02]]
pointNumber1 = 100
possiblePlacementHandle1 = possiblePlacement(samplePosition1, covMatrix1, pointNumber = pointNumber1, env = env)
# distribution 2
samplePosition2 = [0.2, -2.4, 1.51]
covMatrix2 = [[0.08, 0], [0, 0.005]]
pointNumber2 = 500
possiblePlacementHandle2 = possiblePlacement(samplePosition2, covMatrix2, pointNumber = pointNumber2, env = env)
# distribution 3
samplePosition3 = [-2.3, -1.14, 0.673]
covMatrix3 = [[0.02, 0.01], [-0.01, 0.02]]
pointNumber3 = 100
possiblePlacementHandle3 = possiblePlacement(samplePosition3, covMatrix3, pointNumber = pointNumber3, env = env)
# distribution 4
samplePosition4 = [-2.31, -2.18, 0.001]
covMatrix4 = [[0.04, 0.01], [-0.01, 0.002]]
pointNumber4 = 50
possiblePlacementHandle4 = possiblePlacement(samplePosition4, covMatrix4, pointNumber = pointNumber4, env = env)
# distribution 5
samplePosition5 = [-2, 1, 0.001]
covMatrix5 = [[0.02, 0.01], [0.01, 0.02]]
pointNumber5 = 50
possiblePlacementHandle5 = possiblePlacement(samplePosition5, covMatrix5, pointNumber = pointNumber5, env = env)
raw_input('The robot decides where to put the object. In this demo, the robot chooses the shelf at the wall.')
raw_input('The robot first slides the object on the table surface.')

# pdb.set_trace() #breakpoint
slowlyRotateTwoObjAroundAnAxisNormalToGround(obj1 = robot, obj2 = env.GetBodies()[10], posAxis = robot.GetTransform()[0:3,3], radiens = -pi/4, env = env)
robDestination = array([0, -1.7])
posDiff = robDestination - robot.GetTransform()[0:2,3]
oliveOilBottleDestination = env.GetBodies()[10].GetTransform()[0:2,3] + posDiff
raw_input('Then, the robot moves towards the shelf with the object.')
slowlyMoveTwoObjToAPoint(obj1 = robot, obj2 = env.GetBodies()[10], destination1_2D = robDestination, destination2_2D = oliveOilBottleDestination, stepNum = 20, env = env, drawOrNotDraw1 = 'notDraw', drawOrNotDraw2 = 'notDraw')
slowlyRotateTwoObjAroundAnAxisNormalToGround(obj1 = robot, obj2 = env.GetBodies()[10], posAxis = robot.GetTransform()[0:3,3], radiens = pi/6, env = env)

raw_input('The robot places the object to the position.')
oliveOilBottleTransform = env.GetBodies()[10].GetTransform()
manipToolTransform = manip_tool.GetTransform()
transformFromBottToTool = np.dot(inv(oliveOilBottleTransform), manipToolTransform)
oliveOilBottleDestinationCabinet = array([[1,0,0,0.22],[0,1,0,-2.4],[0,0,1,1.51],[0,0,0,1]])
rotBotFrame = giveRotationMatrix3D_4X4('z', -pi/2)
oliveOilBottleDestinationCabinet[0:3,0:3] = np.dot(rotBotFrame[0:3,0:3] ,oliveOilBottleDestinationCabinet[0:3,0:3])
oliveOilBottlePosIntermedStep = env.GetBodies()[10].GetTransform()
oliveOilBottlePosIntermedStep[2,3] = 1.65

slowlyPlaceAnObjToAPosition(robot = robot, obj = env.GetBodies()[10], objNewTransform = oliveOilBottlePosIntermedStep, stepNumber = 10, env = env)
# pdb.set_trace() #breakpoint
slowlyPlaceAnObjToAPosition(robot = robot, obj = env.GetBodies()[10], objNewTransform = oliveOilBottleDestinationCabinet, stepNumber = 10, env = env)
raw_input('The robot release the grasping configuration.')
slowlyReleaseGraspAnObjForDemo(robot, env)
slowlyReleaseMoveManipToAnObj(robot, env)
#############################################################################################################
# move towards the next object...
#############################################################################################################
raw_input('Show the end effector manipulability on the current robot arm configuration.')
ManipVecXYZHandleOri = drawManipXYZ(robot,manip_tool,env)
raw_input('The robot turns around to handle the second object.')
for i_turn in range(10):
  rotateAroundAnAxisNormalToGround(robot, robot.GetTransform()[0:3,3], 3.0/40*pi)
  time.sleep(0.2)



# handle.append(PlotFrame(env, oliveOilBottleTransform, 0.1))





pdb.set_trace() #breakpoint

PosOriOld = env.GetBodies()[0].GetTransform() 
rotationMatrix = giveRotationMatrix3D_4X4('z', -pi/4) 
for i_index in range(4):
  for j_index in range(4):
    rotationMatrix[i_index][j_index] = round(rotationMatrix[i_index][j_index],3)
rotationMatrix[0:3,3] = PosOriOld[0:3,3]
env.GetBodies()[0].SetTransform(rotationMatrix)

PosOriOld = env.GetBodies()[10].GetTransform() 
# del rangeHandle1
# rangeHandle1 = drawPathRangeToObj(robot, pointWithHighManipXY, robotRangeMax, env)


pdb.set_trace() #breakpoint
env.GetBodies()[8]





