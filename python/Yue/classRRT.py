import pdb
import time
from datetime import datetime
import random
import math
import copy
from numpy import linalg as LA
import matplotlib.pyplot as plt
import numpy as np
############################################################################
# some RRT algorithms: RRT, ballRRT
############################################################################

############################################################################
# RRT
class RRT():
    """
    Class for RRT Planning
    """

    def __init__(self, start, goal, obstacleList, randArea,expandDis=0.2,goalSampleRate=5,maxIter=500):
        u"""
        Setting Parameter

        start:Start Position [x,y]
        goal:Goal Position [x,y]
        obstacleList:obstacle Positions [[x,y,size],...]
        randArea:Ramdom Samping Area [min,max]

        """
        self.start=NodeRRT(start[0],start[1])
        self.end=NodeRRT(goal[0],goal[1])
        self.nodeList = [self.start]
        self.minrand = randArea[0]
        self.maxrand = randArea[1]
        self.expandDis = expandDis
        self.goalSampleRate = goalSampleRate
        self.maxIter = maxIter
        self.obstacleList = obstacleList

    def Planning(self,animation=True):
        u"""
        Pathplanning 

        animation: flag for animation on or off
        """

        while True:
            # Random Sampling
            if random.randint(0, 100) > self.goalSampleRate:
                rnd = [random.uniform(self.minrand, self.maxrand), random.uniform(self.minrand, self.maxrand)]
            else:
                rnd = [self.end.x, self.end.y]

            # Find nearest node
            nind = self.GetNearestListIndex(self.nodeList, rnd)
            # print(nind)

            # expand tree
            nearestNode =self.nodeList[nind]
            theta = math.atan2(rnd[1] - nearestNode.y, rnd[0] - nearestNode.x)

            newNode = copy.deepcopy(nearestNode)
            newNode.x += self.expandDis * math.cos(theta)
            newNode.y += self.expandDis * math.sin(theta)
            newNode.parent = nind

            if not self.__CollisionCheck(newNode, self.obstacleList):
                continue

            self.nodeList.append(newNode)

            # check goal
            dx = newNode.x - self.end.x
            dy = newNode.y - self.end.y
            d = math.sqrt(dx * dx + dy * dy)
            if d <= self.expandDis:
                # pdb.set_trace() #breakpoint
                print("Goal!!")
                break

            if animation:
                self.DrawGraph(rnd)

            
        path=[[self.end.x,self.end.y]]
        lastIndex = len(self.nodeList) - 1
        while self.nodeList[lastIndex].parent is not None:
            node = self.nodeList[lastIndex]
            path.append([node.x,node.y])
            lastIndex = node.parent
        path.append([self.start.x, self.start.y])

        return path

    def DrawGraph(self,rnd=None):
        import matplotlib.pyplot as plt
        plt.clf()
        if rnd is not None:
            plt.plot(rnd[0], rnd[1], "^k")
        for node in self.nodeList:
            if node.parent is not None:
                plt.plot([node.x, self.nodeList[node.parent].x], [node.y, self.nodeList[node.parent].y], "-g")
        for (x,y,size) in self.obstacleList:
            self.PlotCircle(x,y,size)

        plt.plot(self.start.x, self.start.y, "xr")
        plt.plot(self.end.x, self.end.y, "xr")
        plt.axis([-2, 15, -2, 15])
        plt.grid(True)
        plt.pause(0.01)

    def PlotCircle(self,x,y,size):
        deg=range(0,360,5)
        deg.append(0)
        xl=[x+size*math.cos(math.radians(d)) for d in deg]
        yl=[y+size*math.sin(math.radians(d)) for d in deg]
        plt.plot(xl, yl, "-k")

    def GetNearestListIndex(self, nodeList, rnd):
        dlist = [(node.x - rnd[0]) ** 2 + (node.y - rnd[1]) ** 2 for node in nodeList]
        minind = dlist.index(min(dlist))
        return minind

    def __CollisionCheck(self, node, obstacleList):

        for (ox, oy, size) in obstacleList:
            dx = ox - node.x
            dy = oy - node.y
            d = math.sqrt(dx * dx + dy * dy)
            if d <= size:
                return False  # collision

        return True  # safe

class NodeRRT():
    """
    RRT Node
    """

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None
# RRT
############################################################################

############################################################################
# ball RRT

class ballRRT():
    u"""
    Class for ballRRT Planning
    """

    def __init__(self, start, goal, randArea, obstacleList, ballRadius, maxCollisionCheckIncrement, goalConnectionRate):
        """
        Setting Parameter

        start:Start Position [x,y]
        goal:Goal Position [x,y]
        obstacleList:obstacle Positions [[x,y,size],...]
        randArea:Ramdom Samping Area [min,max]
        """
        # pdb.set_trace() #breakpoint
        self.start=NodeBallRRT(start[0],start[1], goal)
        self.end=NodeBallRRT(goal[0],goal[1], goal)
        self.minrand = randArea[0]
        self.maxrand = randArea[1]
        self.initBallRadius = ballRadius
        self.ballRadius = ballRadius
        self.maxCollisionCheckIncrement = maxCollisionCheckIncrement
        # self.maxSamples = maxSamples
        self.nodeList = [self.start]
        self.obstacleList = obstacleList
        self.goalConnectionRate = goalConnectionRate

    def Planning(self, animation=True):
        """
        Pathplanning 
        animation: flag for animation on or off
        """

        """
        # check if start and end nodes are connectable
        if self.LineCollisionCheck(self.start, self.end, self.obstacleList): # no collision between start and goal
            pdb.set_trace() #breakpoint
            path=[[self.end.x,self.end.y]]
            path.append([self.start.x, self.start.y])
            return path
        else: # there is collision between start and goal
            # pdb.set_trace() #breakpoint
            [interMNode, noNewNodeAdded] = self.giveIntermediateNode(self.start, self.end, self.obstacleList)
            if not noNewNodeAdded:
                interMNode.parent = 0
                self.nodeList.append(interMNode)
        """
        ballRadius_Loop = self.initBallRadius
        ballCenterNode_Loop = self.start
        # pdb.set_trace() #breakpoint
        # for i_RRTIteration in range(self.maxSamples):
        while(True):
            [rnd, rndNode] = self.sampleBallRRT(ballCenterNode_Loop, ballRadius_Loop, self.obstacleList)
            # Find nearest node
            nind = self.GetNearestListIndex(self.nodeList, rnd)
            nearestNode =self.nodeList[nind]
            # print(nind)

            # expand tree
            if self.LineCollisionCheck(nearestNode, rndNode, self.obstacleList): # no collision between both nodes
                rndNode.parent = nind
                self.nodeList.append(rndNode)
                newNode = rndNode
            else: # collision between two nodes
                continue
                """
                [interMNode, noNewNodeAdded] = self.giveIntermediateNode(nearestNode, rndNode, self.obstacleList)
                if noNewNodeAdded:
                    continue
                interMNode.parent = nind
                self.nodeList.append(interMNode)
                newNode = interMNode
                """
            if np.random.uniform(0,1,1) > self.goalConnectionRate: # probability to connect to the target
                ballCenterNode_Loop = newNode
                ballRadius_Loop = self.initBallRadius
                continue
            ## try to connect to the target

            # pdb.set_trace() #breakpoint
            # if self.LineCollisionCheck(newNode, self.end, self.obstacleList): # no collision between new node and goal
            #    print("Goal!!")
            #    break
            # else: # there is collision between two nodes
                # pdb.set_trace() #breakpoint
                # goalNode = NodeBallRRT()
            
            if self.LineCollisionCheck(newNode, self.end, self.obstacleList): # no collision between both nodes
                print("Goal!!")
                break
            else:
                if newNode.distanceToTarget < nearestNode.distanceToTarget:
                    ballCenterNode_Loop = newNode
                    ballRadius_Loop = self.initBallRadius
                else: 
                    ballCenterNode_Loop = nearestNode
                    ballRadius_Loop = ballRadius_Loop * 2
                
            """
            [newNode2, noNewNodeAdded] = self.giveIntermediateNode(newNode, self.end, self.obstacleList)
            if not noNewNodeAdded: # added some node
                if newNode2.distanceToTarget <= 1e-3:
                    # pdb.set_trace() #breakpoint
                    print("Goal!!")
                    break
                newNode2.parent = len(self.nodeList)-1
                self.nodeList.append(newNode2)
                if newNode2.distanceToTarget < nearestNode.distanceToTarget:
                    ballCenterNode_Loop = newNode2
                    ballRadius_Loop = self.initBallRadius
                else:
                    ballCenterNode_Loop = nearestNode
                    # ballRadius_Loop = ballRadius_Loop + ballRadius_Loop
                    ballRadius_Loop = ballRadius_Loop * 2
            else:
                ballCenterNode_Loop = nearestNode
                # ballRadius_Loop = ballRadius_Loop + ballRadius_Loop
                ballRadius_Loop = ballRadius_Loop * 2
            """
            # forVideoRecord = True
            if animation:
                # pdb.set_trace() #breakpoint
                # time.sleep(0.1)
                self.DrawGraph(rnd)

        # pdb.set_trace() #breakpoint
        # return 'null'



            """
            theta = math.atan2(rnd[1] - nearestNode.y, rnd[0] - nearestNode.x)

            newNode = copy.deepcopy(nearestNode)
            newNode.x += self.expandDis * math.cos(theta)
            newNode.y += self.expandDis * math.sin(theta)
            newNode.parent = nind

            if not self.__CollisionCheck(newNode, obstacleList):
                continue

            self.nodeList.append(newNode)
            

            # check goal
            dx = newNode.x - self.end.x
            dy = newNode.y - self.end.y
            d = math.sqrt(dx * dx + dy * dy)
            if d <= self.expandDis:
                print("Goal!!")
                break

            if animation:
                self.DrawGraph(rnd)
            """
        # if i_RRTIteration == self.maxSamples - 1:
        #     pdb.set_trace() #breakpoint

        path=[[self.end.x,self.end.y]]
        lastIndex = len(self.nodeList) - 1
        while self.nodeList[lastIndex].parent is not None:
            node = self.nodeList[lastIndex]
            path.append([node.x,node.y])
            lastIndex = node.parent
        path.append([self.start.x, self.start.y])

        return path

    def DrawGraph(self,rnd=None):
        import matplotlib.pyplot as plt
        plt.clf()
        if rnd is not None:
            plt.plot(rnd[0], rnd[1], "^k")
        for node in self.nodeList:
            if node.parent is not None:
                plt.plot([node.x, self.nodeList[node.parent].x], [node.y, self.nodeList[node.parent].y], "-g")
        for (x,y,size) in self.obstacleList:
            self.PlotCircle(x,y,size)

        plt.plot(self.start.x, self.start.y, "xr")
        plt.plot(self.end.x, self.end.y, "xr")
        plt.axis([-2, 15, -2, 15])
        plt.grid(True)
        plt.pause(0.01)

    def PlotCircle(self,x,y,size):
        deg=range(0,360,5)
        deg.append(0)
        xl=[x+size*math.cos(math.radians(d)) for d in deg]
        yl=[y+size*math.sin(math.radians(d)) for d in deg]
        plt.plot(xl, yl, "-k")

    def GetNearestListIndex(self, nodeList, rnd):
        dlist = [(node.x - rnd[0]) ** 2 + (node.y - rnd[1]) ** 2 for node in nodeList]
        minind = dlist.index(min(dlist))
        return minind

    def __CollisionCheck(self, node, obstacleList):

        for (ox, oy, size) in obstacleList:
            dx = ox - node.x
            dy = oy - node.y
            d = math.sqrt(dx * dx + dy * dy)
            if d <= size:
                return False  # collision

        return True  # safe

    def LineCollisionCheck(self, node1, node2, obstacleList):
        # Line Equation
        x1 = node1.x
        y1 = node1.y
        x2 = node2.x
        y2 = node2.y

        try:
            deltaY = y2-y1
            deltaX = -(x2-x1)
            cParam = y2*(x2-x1)-x2*(y2-y1)
        except ZeroDivisionError:
            return False

        for (ox,oy,size) in obstacleList:
            distance = abs(deltaY*ox+deltaX*oy+cParam)/(math.sqrt(deltaY*deltaY+deltaX*deltaX))
            #  print((ox,oy,size,distance))
            if distance <= (size):
                #  print("NG")
                return False # collision
        return True  # no collision

    def giveIntermediateNode(self, firstNode, secondNode, obstacleList):
        noNewNodeAdded = False
        first = np.array([firstNode.x, firstNode.y])
        second = np.array([secondNode.x, secondNode.y])
        stepNum = 1.0
        while(True):
            diff = (second - first)/stepNum
            XYincrement = LA.norm(diff)
            if XYincrement > self.maxCollisionCheckIncrement:
                stepNum = stepNum + 1
            else:
                break
        # pdb.set_trace() #breakpoint
        for i_step in range(int(stepNum)-1):
            crrt = first + (i_step + 1) * diff
            crrtNode = NodeBallRRT(crrt[0], crrt[1], [self.end.x, self.end.y])
            # pdb.set_trace() #breakpoint
            if self.__CollisionCheck(crrtNode, obstacleList): # no collision
                continue
            else:
                # pdb.set_trace() #breakpoint
                if i_step == 0:
                    noNewNodeAdded = True
                    interMNode = firstNode
                    return [interMNode, noNewNodeAdded]
                beforeCllsn = first + i_step * diff
                nodeBeforeCllsn = NodeBallRRT(beforeCllsn[0], beforeCllsn[1], [self.end.x, self.end.y])
                interMNode = nodeBeforeCllsn
                return [interMNode, noNewNodeAdded]
        interMNode = secondNode
        return [interMNode, noNewNodeAdded]

    def sampleBallRRT(self, centerNode, ballRadius, obstacleList):
        validSampling = False
        while(not validSampling):
            ifCollisionFreeWithObstacles = True
            sampledPointX = np.random.uniform(max(self.minrand,centerNode.x - ballRadius/math.sqrt(2)), min(self.maxrand,centerNode.x + ballRadius/math.sqrt(2)), 1)
            sampledPointY = np.random.uniform(max(self.minrand,centerNode.y - ballRadius/math.sqrt(2)), min(self.maxrand,centerNode.y + ballRadius/math.sqrt(2)), 1)
            sampledNode = NodeBallRRT(sampledPointX[0], sampledPointY[0], [self.end.x, self.end.y])

            ifCollisionFreeWithObstacles = self.__CollisionCheck(sampledNode, obstacleList)
            ifLargeEnough = (LA.norm([sampledPointX - centerNode.x, sampledPointY - centerNode.y]) >= ballRadius/2.0)
            validSampling = ifCollisionFreeWithObstacles and ifLargeEnough 
        return [[sampledPointX, sampledPointY], sampledNode]

class NodeBallRRT():
    u"""
    BallRRT Node
    """

    def __init__(self, x, y, goal):
        self.x = x
        self.y = y
        self.parent = None
        self.distanceToTarget = LA.norm([goal[0]-x, goal[1]-y])
# ball RRT definition
############################################################################
def GetPathLength(path):
    l = 0
    for i in range(len(path) - 1):
        dx = path[i + 1][0] - path[i][0]
        dy = path[i + 1][1] - path[i][1]
        d = math.sqrt(dx * dx + dy * dy)
        l += d

    return l


def GetTargetPoint(path, targetL):
    l = 0
    ti = 0
    lastPairLen = 0
    for i in range(len(path) - 1):
        dx = path[i + 1][0] - path[i][0]
        dy = path[i + 1][1] - path[i][1]
        d = math.sqrt(dx * dx + dy * dy)
        l += d
        if l >= targetL:
            ti = i-1
            lastPairLen = d
            break

    partRatio = (l - targetL) / lastPairLen
    #  print(partRatio)
    #  print((ti,len(path),path[ti],path[ti+1]))

    x = path[ti][0] + (path[ti + 1][0] - path[ti][0]) * partRatio
    y = path[ti][1] + (path[ti + 1][1] - path[ti][1]) * partRatio
    #  print((x,y))

    return [x, y, ti]


def LineCollisionCheck(first, second, obstacleList):
    # Line Equation

    x1=first[0]
    y1=first[1]
    x2=second[0]
    y2=second[1]

    try:
        a=y2-y1
        b=-(x2-x1)
        c=y2*(x2-x1)-x2*(y2-y1)
    except ZeroDivisionError:
        return False

    #  print(first)
    #  print(second)

    for (ox,oy,size) in obstacleList:
        d=abs(a*ox+b*oy+c)/(math.sqrt(a*a+b*b))
        #  print((ox,oy,size,d))
        if d<=(size):
            #  print("NG")
            return False

    #  print("OK")

    return True  # OK


def PathSmoothing(path, maxIter, obstacleList):
    #  print("PathSmoothing")

    l = GetPathLength(path)

    for i in range(maxIter):
        # Sample two points
        pickPoints = [random.uniform(0, l), random.uniform(0, l)]
        pickPoints.sort()
        #  print(pickPoints)
        first = GetTargetPoint(path, pickPoints[0])
        #  print(first)
        second = GetTargetPoint(path, pickPoints[1])
        #  print(second)

        if first[2]<=0 or second[2]<=0:
            continue

        if (second[2]+1) > len(path):
            continue

        if second[2]==first[2]:
            continue

        # collision check
        if not LineCollisionCheck(first, second, obstacleList):
            continue

        #Create New path
        newPath=[]
        newPath.extend(path[:first[2]+1])
        newPath.append([first[0],first[1]])
        newPath.append([second[0],second[1]])
        newPath.extend(path[second[2]+1:])
        path=newPath
        l = GetPathLength(path)

    return path

"""
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    #====Search Path with RRT====
    # Parameter
    obstacleList = [
        (5, 5, 1),
        (6, 6, 2),
        (4, 8, 2),
        (5, 10, 1.8),
        (7, 3, 2),
    ]  # [x,y,size]
    rrt=RRT(start=[0.0,7.0],goal=[8.0,9.0],randArea=[-2.0,15.0],obstacleList=obstacleList)

    timeCalcBegin = datetime.now()
    path=rrt.Planning(animation=True)
    timeCalcEnd = datetime.now()
    durationBallRRT = timeCalcEnd - timeCalcBegin
    print 'RRT calculation computing time: ' + str(durationBallRRT.total_seconds()) + 'seconds'


    # Draw final path
    rrt.DrawGraph()
    plt.plot([x for (x,y) in path], [y for (x,y) in path],'-r')

    #Path smoothing
    maxIter=1000
    smoothedPath = PathSmoothing(path, maxIter, obstacleList)
    plt.plot([x for (x,y) in smoothedPath], [y for (x,y) in smoothedPath],'-b')

    plt.grid(True)
    plt.pause(0.01)  # Need for Mac
    plt.show()
"""