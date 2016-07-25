#!/usr/bin/python

import rospy
from std_msgs.msg import String
import random
import numpy as np
import time
from sensor_msgs.msg import JointState
from math import radians


def talker():
    joint1 = radians(90)
    pub = rospy.Publisher('/kukaAllegroHand/robot_cmd', JointState)
    rospy.init_node('talker', anonymous=True)
    r = rospy.Rate(10000) # 10hz
    msg = JointState()
    msg.position = [0,joint1,0,0,0,0,0]
    pub.publish(msg)
    r.sleep()
    time.sleep(.4)
    
    while True:
        for acc in range(600,601):
            print '************************* acceleratioin: %d *************************' % acc
            # acc = 200
            trjFile = '../trj/acc_new_' + str(acc) + '.dat'
            trj = np.genfromtxt(trjFile, delimiter="\t")
            done = True
            i = 1
            while not rospy.is_shutdown() and i < len(trj):
                print i
                msg = JointState()
                msg.position = [0,joint1,0,0,0,0,0]
                msg.position[0] = radians(trj[i][0])
                msg.position[2] = radians(trj[i][0])
                msg.position[3] = radians(trj[i][0])
                msg.position[4] = radians(trj[i][0])
                msg.position[5] = radians(trj[i][0])
                msg.position[6] = radians(trj[i][0])
                pub.publish(msg)
                r.sleep()
                time.sleep((trj[i][2] - trj[i-1][2])/10)
                i += 1

if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException: pass
