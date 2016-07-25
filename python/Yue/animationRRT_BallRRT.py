import pdb
import time
from datetime import datetime
import random
import math
import copy
from numpy import linalg as LA
import numpy as np
import matplotlib.pyplot as plt
from classRRT import *

obstacleList = [
        (9, 12, 2),
        (9, 2, 2),
        (5, 7, 4),
    ]  # [x,y,size]
"""
obstacleList = [
        (5, 7, 4),
    ]  # [x,y,size]
# obstacleList = []
"""
start = [0.0,7.0]
goal=[12.0,7.0]
randArea=[-2.0,15.0]
numExperiment = 100
computingTimeRRT = np.zeros(numExperiment)
nodeExploredRRT = np.zeros(numExperiment)
computingTimeBallRRT = np.zeros(numExperiment)
nodeExploredBallRRT = np.zeros(numExperiment)


# pdb.set_trace() #breakpoint
for i_experiment in range(numExperiment):
	
	# pdb.set_trace() #breakpoint
	# RRT
	rrt=RRT(start, goal, obstacleList, randArea, expandDis=0.5, goalSampleRate=5)
	pdb.set_trace() #breakpoint
	rrt.DrawGraph()

	timeCalcBegin = datetime.now()
	path=rrt.Planning(animation=False)
	timeCalcEnd = datetime.now()
	durationRRT = timeCalcEnd - timeCalcBegin
	computingTimeRRT[i_experiment] = durationRRT.total_seconds()
	nodeExploredRRT[i_experiment] = np.shape(rrt.nodeList)[0]
	rrt.DrawGraph()
	# pdb.set_trace() #breakpoint
	plt.plot([x for (x,y) in path], [y for (x,y) in path],'-r')
	## Path smoothing
	maxIter = 1000
	smoothedPath = PathSmoothing(path, maxIter, obstacleList)
	plt.plot([x for (x,y) in smoothedPath], [y for (x,y) in smoothedPath],'-b')
	plt.grid(True)
	# plt.show()
	pdb.set_trace() #breakpoint
	
	
	# ballRRT
	ballrrt=ballRRT(start, goal, randArea, obstacleList, ballRadius = 0.5, maxCollisionCheckIncrement = 0.1, goalConnectionRate = 0.95)
	pdb.set_trace() #breakpoint
	ballrrt.DrawGraph()
	timeCalcBegin = datetime.now()
	path=ballrrt.Planning(animation=False)
	timeCalcEnd = datetime.now()
	durationBallRRT = timeCalcEnd - timeCalcBegin
	computingTimeBallRRT[i_experiment] = durationBallRRT.total_seconds()
	nodeExploredBallRRT[i_experiment] = np.shape(ballrrt.nodeList)[0]
	# Draw final path
	ballrrt.DrawGraph()
	# pdb.set_trace() #breakpoint
	plt.plot([x for (x,y) in path], [y for (x,y) in path],'-r')
	## Path smoothing
	maxIter=1000
	smoothedPath = PathSmoothing(path, maxIter, obstacleList)
	plt.plot([x for (x,y) in smoothedPath], [y for (x,y) in smoothedPath],'-b')
	# pdb.set_trace() #breakpoint
	plt.grid(True)
	# plt.show()
	# pdb.set_trace() #breakpoint
	# plt.close()
	
print 'RRT time: ' + str(computingTimeRRT)
print 'mean(time RRT): ' + str(np.mean(computingTimeRRT))
print 'variance(time RRT): ' + str(np.var(computingTimeRRT))
print 'BallRRT time: ' + str(computingTimeBallRRT)
print 'mean(time ballRRT): ' + str(np.mean(computingTimeBallRRT))
print 'variance(time ballRRT): ' + str(np.var(computingTimeBallRRT))

print 'node RRT: ' + str(nodeExploredRRT)
print 'mean(node RRT): ' + str(np.mean(nodeExploredRRT))
print 'variance(node RRT): ' + str(np.var(nodeExploredRRT))
print 'node ballRRT: ' + str(nodeExploredBallRRT)
print 'mean(node ballRRT): ' + str(np.mean(nodeExploredBallRRT))
print 'variance(node ballRRT): ' + str(np.var(nodeExploredBallRRT))

pdb.set_trace() #breakpoint
