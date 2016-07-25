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
# pdb.set_trace() #breakpoint
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
areaList = [] # object class, area name, area image, object position when positioned inside, area length and width
areaList.append([env.GetBodies()[3],'kitchen body', 'kitchenBody_Area.png',[0.3,0,1.456],[0.972, 2.05]])
areaList.append([env.GetBodies()[5],'shelf','shelf_Area.png',[0,0.1,1.51], [1.4, 0.4]])
areaList.append([env.GetBodies()[7],'table 1','table1_Area.png',[0,0,1.25], [1.44, 1.6]])
areaList.append([env.GetBodies()[8],'table 2','table2_Area.png',[0,0,0.925],[1.09, 1.09]])
areaList.append([env.GetBodies()[10],'trash bin','trash_bin.png',[0,0,0.2],[0.372, 0.372]])

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
objectList = [] # [0]. obj class, [1]. obj name, [2]. obj image, [3]. Z-rotation sample?
enumObjBegin = 10
enumobj = 0
# glass
# pdb.set_trace() #breakpoint

env.Load('../../robots/Kitchen_LASA/Kitchen_glass.kinbody.xml')
enumobj = enumobj + 1 
enumGlass = enumobj
obj = env.GetBodies()[enumObjBegin+enumGlass]
objectList.append([obj,'green glass', 'greenGlass_Obj.png']) 
objPosOri = obj.GetTransform()
objPosOri[0:3,3] = array([1.58793, -1.17354, 1.24])
obj.SetTransform(objPosOri)
# plate
env.Load('../../robots/Kitchen_LASA/Kitchen_plate2.kinbody.xml')
enumobj = enumobj + 1 
enumPlate = enumobj
obj = env.GetBodies()[enumObjBegin+enumPlate]
objectList.append([obj,'plate', 'plate_Obj.png'])
objPosOri = obj.GetTransform()
objPosOri[0:3,3] = array([2.39525, -0.80829, 1.24])
obj.SetTransform(objPosOri)
# fork
env.Load('../../robots/Kitchen_LASA/Kitchen_fork.kinbody.xml')
enumobj = enumobj + 1 
enumFork = enumobj
obj = env.GetBodies()[enumObjBegin+enumFork]
objectList.append([obj,'fork', 'fork.png'])
objPosOri = obj.GetTransform()
objPosOri[0:3,3] = array([1.62389, -0.25461, 1.24])
obj.SetTransform(objPosOri)
# cup
env.Load('../../robots/Kitchen_LASA/Kitchen_cup.kinbody.xml')
enumobj = enumobj + 1 
enumCup = enumobj
obj = env.GetBodies()[enumObjBegin+enumCup]
objectList.append([obj,'cup', 'cup_Obj.png'])
objPosOri = obj.GetTransform()
objPosOri[0:3,3] = array([1.9, -1.2, 1.23])
obj.SetTransform(objPosOri)

# knife
env.Load('../../robots/Kitchen_LASA/Kitchen_knife.kinbody.xml')
enumobj = enumobj + 1 
enumKnife = enumobj
obj = env.GetBodies()[enumObjBegin+enumKnife]
objectList.append([obj,'knife', 'knife_Obj.png'])
objPosOri = obj.GetTransform()
objPosOri[0:3,3] = array([1.97853, -0.60885, 1.24])
obj.SetTransform(objPosOri)

# mug
env.Load('../../robots/Kitchen_LASA/Kitchen_mug.kinbody.xml')
enumobj = enumobj + 1 
enumMug = enumobj
obj = env.GetBodies()[enumObjBegin+enumMug]
objectList.append([obj,'mug', 'mug_Obj.png'])
objPosOri = obj.GetTransform()
objPosOri[0:3,3] = array([2.526, -0.335, 1.24])
obj.SetTransform(objPosOri)
"""
# ladle
env.Load('../../robots/Kitchen_LASA/Kitchen_ladle.kinbody.xml')
enumobj = enumobj + 1 
enumLadle = enumobj
obj = env.GetBodies()[enumObjBegin+enumLadle]
objectList.append([obj,'ladle', 'ladle_Obj.png'])
objPosOri = obj.GetTransform()
objPosOri[0:3,3] = array([2.03383, -0.97494, 1.24])
obj.SetTransform(objPosOri)
"""
# monitor
env.Load('../../robots/Kitchen_LASA/Kitchen_monitor.kinbody.xml')
enumobj = enumobj + 1 
enumMonitor = enumobj
obj = env.GetBodies()[enumObjBegin+enumMonitor]
objectList.append([obj,'monitor', 'monitor_Obj.png'])
objPosOri = obj.GetTransform()
objPosOri[0:3,3] = array([2.03383, -0.97494, 1.24])
obj.SetTransform(objPosOri)
"""
# keyboard
env.Load('../../robots/Kitchen_LASA/Kitchen_keyboard.kinbody.xml')
enumobj = enumobj + 1 
enumKeyboard = enumobj
obj = env.GetBodies()[enumObjBegin+enumKeyboard]
objectList.append([obj,'keyboard', 'keyboard_Obj.png'])
objPosOri = obj.GetTransform()
objPosOri[0:3,3] = array([2.23383, -0.67494, 1.24])
obj.SetTransform(objPosOri)

# table lamp
env.Load('../../robots/Kitchen_LASA/Kitchen_table_lamp.kinbody.xml')
enumTableLamp = 3
obj = env.GetBodies()[enumObjBegin+enumTableLamp]
objectList.append([obj,'table lamp', 'table_lamp.png'])
objPosOri = obj.GetTransform()
objPosOri[0:3,3] = array([1.814, -0.6556, 1.24])
obj.SetTransform(objPosOri)


# sauce pan
env.Load('../../robots/Kitchen_LASA/Kitchen_saucepan.kinbody.xml')
enumSaucepan = 1
obj = env.GetBodies()[enumObjBegin+enumSaucepan]
objectList.append([obj,'sauce pan', 'saucePan_Obj.png'])
objPosOri = obj.GetTransform()
objPosOri[0:3,3] = array([2.33884, -1.28367, 1.24])
obj.SetTransform(objPosOri)

#  wine bottle
env.Load('../../robots/Kitchen_LASA/Kitchen_wine_bottle.kinbody.xml')
enumWineBottle = 3
obj = env.GetBodies()[enumObjBegin+enumWineBottle]
objectList.append([obj,'wine bottle', 'wineBottle_Obj.png'])
objPosOri = obj.GetTransform()
objPosOri[0:3,3] = array([1.56, -0.491, 1.24])
obj.SetTransform(objPosOri)

# pasta can
env.Load('../../robots/Kitchen_LASA/Kitchen_pasta_can.kinbody.xml')
obj = env.GetBodies()[enumObjBegin+7]
objectList.append([obj,'pasta can', 'pastaCan_Obj.png'])
objPosOri = obj.GetTransform()
objPosOri[0:3,3] = array([1.724, -0.68172, 1.24])
obj.SetTransform(objPosOri)


# olive oil bottle
env.Load('../../robots/Kitchen_LASA/Kitchen_olive_oil_bottle.kinbody.xml')
obj = env.GetBodies()[enumObjBegin+1]
objectList.append([obj,'olive oil bottle', 'oliveOilBottle_Obj.png'])
objPosOri = obj.GetTransform()
objPosOri[0:3,3] = array([1.98247, -0.40261, 1.24])
obj.SetTransform(objPosOri)
"""

# pdb.set_trace() #breakpoint

## placement result example
"""
v = raw_input('Organising example or not (y/n):\n')
if v == 'y':
	# plate
	obj = env.GetBodies()[enumObjBegin+8]
	objPosOri = obj.GetTransform()
	objPosOri[0:3,3] = array([2.02982, 1.02692, 0.925])
	obj.SetTransform(objPosOri)
	# 11. fork
	obj = env.GetBodies()[enumObjBegin+10]
	objPosOri = obj.GetTransform()
	objPosOri[0:3,3] = array([1.9861, 0.72187, 0.925])
	obj.SetTransform(objPosOri)
	# 3. knife
	obj = env.GetBodies()[enumObjBegin+2]
	objPosOri = obj.GetTransform()
	objPosOri[0:3,3] = array([1.98756, 0.633, 0.925])
	obj.SetTransform(objPosOri)
	# 1. glass
	obj = env.GetBodies()[enumObjBegin]
	objPosOri = obj.GetTransform()
	objPosOri[0:3,3] = array([2.27452, 1.30707, 0.925])
	obj.SetTransform(objPosOri)
	# 10. wine bottle
	obj = env.GetBodies()[enumObjBegin+9]
	objPosOri = obj.GetTransform()
	objPosOri[0:3,3] = array([2.38124, 1.17741, 0.925])
	obj.SetTransform(objPosOri)
	# 8. pasta can
	obj = env.GetBodies()[enumObjBegin+7]
	objPosOri = obj.GetTransform()
	objPosOri[0:3,3] = array([2.37480, 0.92938, 0.925])
	obj.SetTransform(objPosOri)
	# 4. cup
	obj = env.GetBodies()[enumObjBegin+3]
	objPosOri = obj.GetTransform()
	objPosOri[0:3,3] = array([2.53546, 0.77, 0.925])
	obj.SetTransform(objPosOri)
	# 7. mug
	obj = env.GetBodies()[enumObjBegin+6]
	objPosOri = obj.GetTransform()
	objPosOri[0:3,3] = array([2.30767, 0.71284, 0.925])
	obj.SetTransform(objPosOri)

v = raw_input('Show some placement points or not (y/n):\n')
if v == 'y':
	samplePosition1 = [2.268, 0.70, 0.925]
	covMatrix1 = [[0.02, 0], [0, 0.005]]
	pointNumber1 = 300
	possiblePlacementHandle1 = possiblePlacement(samplePosition1, covMatrix1, pointNumber = pointNumber1, env = env)
	samplePosition2 = [2.268, 1.316, 0.925]
	covMatrix2 = [[0.02, 0], [0, 0.005]]
	pointNumber2 = 300
	possiblePlacementHandle2 = possiblePlacement(samplePosition2, covMatrix2, pointNumber = pointNumber2, env = env)
	samplePosition3 = [2.62, 1.036, 0.925]
	covMatrix3 = [[0.005, 0], [0, 0.02]]
	pointNumber3 = 300
	possiblePlacementHandle3 = possiblePlacement(samplePosition3, covMatrix3, pointNumber = pointNumber3, env = env)
	samplePosition4 = [1.989, 1.0375, 0.925]
	covMatrix4 = [[0.005, 0], [0, 0.02]]
	pointNumber4 = 300
	possiblePlacementHandle4 = possiblePlacement(samplePosition4, covMatrix4, pointNumber = pointNumber4, env = env)
"""

# pdb.set_trace() #breakpoint
# random generate object placement
v = raw_input('Randomly generate object placement on table 1? (y/n):\n')
if v == 'y':
	area = np.zeros([5,1])
	area[0] = env.GetBodies()[7].GetTransform()[0][3] # table 1
	area[1] = env.GetBodies()[7].GetTransform()[1][3] # table 1
	area[2] = areaList[2][4][0] # table 1
	area[3] = areaList[2][4][1]
	area[4] = areaList[2][3][2]
	gerateRandomPositionObjectsInARectAreaWithZRot(area,objectList,env)

root = Tk()
root.geometry("1000x700+300+300")
controlPanel = objectControlPanel(root, objectList, areaList)
# app.objectList = objectList
pdb.set_trace() #breakpoint
 # print 'abs'    
v = raw_input('Save the current object placement? (y/n):\n')
if v == 'y':
        with open('placementData.pickle') as ff: 
            placementMatrixOld = pickle.load(ff)
            posTable1 = env.GetBodies()[7].GetTransform()[0:2,3]

        placementMatrix



        with open('placementData.pickle','w') as ff: 
            pickle.dump(placementMatrix, ff)
        
# root.mainloop()  
