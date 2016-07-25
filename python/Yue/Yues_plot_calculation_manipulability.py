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
  # import rospy
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


with open('computingCostX.pickle') as ff: 
	X = pickle.load(ff)
with open('computingCostY.pickle') as fff: 
	Y = pickle.load(fff)
with open('computingCostXcomputationCost1.pickle') as ffff: 
	computationCost1 = pickle.load(ffff)
with open('computingCostXcomputationCost2.pickle') as fffff: 
	computationCost2 = pickle.load(fffff)
# pdb.set_trace() #breakpoint

t1_average = sum(computationCost1)/size(computationCost1)
t2_average = sum(computationCost2)/size(computationCost2)


fig = plt.figure(figsize=plt.figaspect(0.5))
ax0 = fig.add_subplot(1, 2, 1)
surf = ax0.contourf(X, Y, computationCost1, cmap=cm.PiYG)
# ax0.set_zlim(0, 10)
ax0.set_xlabel('Object position in X [m]',fontsize = 18)
ax0.set_ylabel('Object position in Y [m]',fontsize = 18)

# fig.colorbar(surf, ax=ax0)
ax0.set_title('Computing Time 1 [s]', fontsize = 20)
fig.colorbar(surf, ax=ax0)
# plt.colorbar()

ax1 = fig.add_subplot(1, 2, 2)
surf2 = ax1.contourf(X, Y, computationCost2, cmap=cm.PiYG)
# ax1.set_zlim(0, 10)
ax1.set_xlabel('Object position in X [m]',fontsize = 18)
ax1.set_ylabel('Object position in Y [m]',fontsize = 18)
ax1.set_title('Computing Time 2 [s]', fontsize = 20)
fig.colorbar(surf2, ax=ax1)
# fig.tight_layout()

plt.show()
