from Tkinter import * 
# Tk,N,S,W,E,BOTH,Listbox,StringVar,DoubleVar,END,RAISED,RIGHT,LEFT,TOP, BOTTOM, X,Y
from ttk import Frame, Button, Style, Label, Entry, Scale
from PIL import Image, ImageTk
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


class objectControlPanel(Frame):

    def __init__(self, parent, objectList, areaList):
        Frame.__init__(self, parent)
        
        self.parent = parent
        self.objectList = objectList
        self.objectNames = []
        self.objectInitRotMatrix = []
        self.objectImages = []
        self.objectSelected = self.objectList[0][0]  # default chosen object 
        self.objectSelectedIndex = 0                 # default chosen object index
        self.objectRotDegCurrent = np.zeros([shape(objectList)[0],3])
        self.objectInAreaName = []
        self.areaList = areaList
        self.configurationMatrix = []

        for i_objName in range(shape(objectList)[0]):
            self.objectNames.append(objectList[i_objName][1])
            self.objectImages.append(objectList[i_objName][2])
            self.objectInitRotMatrix.append(objectList[i_objName][0].GetTransform()[0:3,0:3])
            # print objectList[i_objName][1]
            self.objectInAreaName.append(self.findObjectArea(objectList[i_objName][0]))
        self.areaNames = []
        self.areaImages = []
        self.placementPosInArea = []
        # print self.areaList[0][0].GetTransform[0:3,3]
        for i_areaName in range(shape(areaList)[0]):
            self.areaNames.append(areaList[i_areaName][1])
            self.areaImages.append(areaList[i_areaName][2]) 
            self.placementPosInArea.append(self.areaList[i_areaName][0].GetTransform()[0:3,3]+areaList[i_areaName][3])
        self.xRotDegree = 0.0
        self.yRotDegree = 0.0
        self.zRotDegree = 0.0

        self.initUI()

    def initUI(self):
        self.parent.title("Object Control Panel")
        self.pack(fill=BOTH, expand=10)
        
        frame1 = Frame(self, relief=RAISED, borderwidth = 1)
        frame1.pack(side=LEFT,fill = Y, padx = 10, pady = 10)
        objectNameLbl = Label(frame1, text="Objects", width=6)
        objectNameLbl.pack(anchor = S, padx=5, pady=3)
        objectlb = Listbox(frame1)
        for i in self.objectNames:
            objectlb.insert(END, i)
        objectlb.bind("<<ListboxSelect>>", self.onSelectObj)
        objectlb.pack(anchor=N, pady=5)
        path = 'Yues_class_definition/GUI_images/' + self.objectImages[0]
        imageInLbl = ImageTk.PhotoImage(Image.open(path).resize((150,150)))
        self.objectChosenLbl = Label(frame1, image=imageInLbl)
        self.objectChosenLbl.image = imageInLbl
        self.objectChosenLbl.pack(anchor = N)
        frame1a = Frame(frame1)
        frame1a.pack(anchor = N)
        self.objectChosenName = StringVar()
        self.objectChosenName = self.objectList[0][1]# default chosen object name
        objectChosenTextLbl = Label(frame1a, text=0, width=20, textvariable = self.objectChosenName)
        objectChosenTextLbl.pack(side = LEFT, pady = 10, padx = 10)
        #objectChosenStaticTextLbl = Label(frame1a, width=20, text = " is selected!")
        #objectChosenStaticTextLbl.pack(side=LEFT)
        
        frame2 = Frame(self, relief=RAISED, borderwidth = 1)
        frame2.pack(fill = X, padx = 10, pady = 10)
        frame2a = Frame(frame2)
        frame2a.pack(side=LEFT)
        areaCurrentToLbl = Label(frame2a, text="This object is this area currently:", width=15)
        areaCurrentToLbl.pack(anchor = N,fill = X, padx = 10)
        path = 'Yues_class_definition/GUI_images/' + self.objectInAreaName[0][1]
        imageInLbl = ImageTk.PhotoImage(Image.open(path).resize((200,200)))
        self.areaCurrentLbl = Label(frame2a, image=imageInLbl)
        self.areaCurrentLbl.image = imageInLbl
        self.areaCurrentLbl.pack(side=LEFT,padx = 10)

        frame2b = Frame(frame2, relief=RAISED, borderwidth = 0.5)
        frame2b.pack(side = LEFT, fill = X, padx = 10, pady = 10)
        moveToLbl = Label(frame2b, text="move to area", width=10)
        moveToLbl.pack(side=TOP,padx = 10)
        moveTolb = Listbox(frame2b)
        for i in self.areaNames:
            moveTolb.insert(END, i)
        moveTolb.bind("<<ListboxSelect>>", self.onSelectArea)
        moveTolb.pack(side = LEFT, fill = X, pady=1, padx=10)
        path = 'Yues_class_definition/GUI_images/' + self.areaImages[0]
        imageInLbl = ImageTk.PhotoImage(Image.open(path).resize((160,160)))
        self.areaChosenLbl = Label(frame2b, image=imageInLbl)
        self.areaChosenLbl.image = imageInLbl
        self.areaChosenLbl.pack(side=LEFT, fill = X)
        areaMoveYesButton = Button(frame2b, text = 'Back!', width = 5, command = self.areaMoveNo)
        areaMoveYesButton.pack(side = BOTTOM, pady = 10, padx = 20)
        areaMoveNoButton = Button(frame2b, text = 'Move!', width = 5, command = self.areaMoveYes)
        areaMoveNoButton.pack(side = BOTTOM, pady = 10, padx = 20)
        # self.areaChosen = StringVar()
        # areaChosenLbl = Label(frame2, text=0, width=12, textvariable = self.areaChosen)
        # areaChosenLbl.pack(side =LEFT)
        # frame2bb1 = Frame(frame2b)
        # frame2c.pack(side = LEFT, fill = X, padx = 10, pady = 10)
        
        """
        canvas = Canvas(frame2b, width=10, height=20, bd=1, highlightthickness=1, name='arrow')
        canvas.create_polygon('0 0 1 1 2 2', fill='DeepSkyBlue3', tags=('poly', ), outline='black')
        canvas.pack(side = RIGHT,padx=15)
        self.canvas = Canvas(self, width=400, height=400)
        self.canvas.pack(side="top", fill="both", expand=True)
        self.canvas.create_line(200,200, 200,200, tags=("line",), arrow="last")
        """

        frame3 = Frame(self, relief=RAISED, borderwidth = 1)
        frame3.pack(fill = X, padx = 10, pady = 10, anchor = N)
        translationLbl = Label(frame3, text="Translation (in meters)", width=20)
        translationLbl.pack(side = LEFT, anchor = N)
        frame3a = Frame(frame3)
        frame3a.pack(anchor = N, fill = BOTH)
        frame3a.columnconfigure(0, pad=3)
        frame3a.columnconfigure(1, pad=3)
        frame3a.columnconfigure(2, pad=3)
        frame3a.columnconfigure(3, pad=20)
        frame3a.columnconfigure(4, pad=3)
        frame3a.rowconfigure(0, pad=3)
        frame3a.rowconfigure(1, pad=3)
        frame3a.rowconfigure(2, pad=3)
        xTransLbl = Label(frame3a,text='X:', width = 2)
        xTransLbl.grid(row = 0, column = 0)
        xTransMinusButton = Button(frame3a, text = '-', width = 2, command = self.xTransMinus)
        xTransMinusButton.grid(row = 0, column = 1)
        self.xTransEntry = Entry(frame3a, width = 10)
        self.xTransEntry.insert(0,0.05)
        self.xTransEntry.grid(row = 0, column = 2)
        xTransPlusButton = Button(frame3a, text = '+', width = 2, command = self.xTransPlus)
        xTransPlusButton.grid(row = 0, column = 3)
        self.xPosDisp = StringVar()
        xPosDisplLbl = Label(frame3a, textvariable=self.xPosDisp, width = 10)
        xPosDisplLbl.grid(row = 0, column = 4)
        self.xPosDisp.set(round(self.objectSelected.GetTransform()[0][3])) # display X position 
        yTransLbl = Label(frame3a,text='Y:', width = 2)
        yTransLbl.grid(row = 1, column = 0)
        yTransMinusButton = Button(frame3a, text = '-', width = 2, command = self.yTransMinus)
        yTransMinusButton.grid(row = 1, column = 1)
        self.yTransEntry = Entry(frame3a, width = 10)
        self.yTransEntry.insert(0,0.05)
        self.yTransEntry.grid(row = 1, column = 2)
        yTransPlusButton = Button(frame3a, text = '+', width = 2, command = self.yTransPlus)
        yTransPlusButton.grid(row = 1, column = 3)
        self.yPosDisp = StringVar()
        yPosDisplLbl = Label(frame3a, textvariable=self.yPosDisp, width = 10)
        yPosDisplLbl.grid(row = 1, column = 4)
        self.yPosDisp.set(round(self.objectSelected.GetTransform()[1][3])) # display Y position 
        zTransLbl = Label(frame3a, text='Z:', width = 2)
        zTransLbl.grid(row = 2, column = 0)
        zTransMinusButton = Button(frame3a, text = '-', width = 2, command = self.zTransMinus)
        zTransMinusButton.grid(row = 2, column = 1)
        self.zTransEntry = Entry(frame3a, width = 10)
        self.zTransEntry.insert(0,0.05)
        self.zTransEntry.grid(row = 2, column = 2)
        zTransPlusButton = Button(frame3a, text = '+', width = 2, command = self.zTransPlus)
        zTransPlusButton.grid(row = 2, column = 3)
        self.zPosDisp = StringVar()
        zPosDisplLbl = Label(frame3a, textvariable=self.zPosDisp, width = 10)
        zPosDisplLbl.grid(row = 2, column = 4)
        self.zPosDisp.set(round(self.objectSelected.GetTransform()[2][3])) # display Z position         

        frame4 = Frame(self, relief=RAISED, borderwidth = 1)
        frame4.pack(fill = X, padx = 10, pady = 10, anchor = N)
        rotationLbl = Label(frame4, text="Rotation (in degrees)", width=20)
        rotationLbl.pack(side = LEFT, anchor = N)

        frame4a = Frame(frame4)
        frame4a.pack(anchor = N, fill = X)
        xRotLbl = Label(frame4a,text='X:', width = 2)
        xRotLbl.pack(side=LEFT, padx = 10, pady = 5)
        self.xRotScale = Scale(frame4a,from_=0,to=360,length=200,command = self.xRotOnScale)
        self.xRotScale.pack(side=LEFT)
        self.xRotDisp = DoubleVar()
        xRotScaleLbl = Label(frame4a, textvariable=self.xRotDisp, width = 5)
        xRotScaleLbl.pack(side=LEFT, padx = 10)

        frame4b = Frame(frame4)
        frame4b.pack(anchor = N, fill = X)
        yRotLbl = Label(frame4b,text='Y:', width = 2)
        yRotLbl.pack(side=LEFT, padx = 10, pady = 5)
        self.yRotScale = Scale(frame4b, from_=0, to=360,length=200,command = self.yRotOnScale)
        self.yRotScale.pack(side=LEFT)
        self.yRotDisp = DoubleVar()
        yRotScaleLbl = Label(frame4b, textvariable=self.yRotDisp, width = 5)
        yRotScaleLbl.pack(side=LEFT, padx = 10)

        frame4c = Frame(frame4)
        frame4c.pack(anchor = N, fill = X)
        zRotLbl = Label(frame4c, text='Z:', width = 2)
        zRotLbl.pack(side=LEFT, padx = 10, pady = 5)
        self.zRotScale = Scale(frame4c, from_=0, to=360,length=200,command = self.zRotOnScale)
        self.zRotScale.pack(side=LEFT)
        self.zRotDisp = DoubleVar()
        zRotScaleLbl = Label(frame4c, textvariable=self.zRotDisp, width = 5)
        zRotScaleLbl.pack(side=LEFT, padx = 10)

        frame5 = Frame(self, relief=RAISED, borderwidth = 0)
        frame5.pack(fill = X, padx = 10, pady = 10, anchor = N)
        doneButton = Button(frame5, text = 'Save data!', width = 5, command=self.saveData)
        doneButton.pack(side = RIGHT, padx = 10, pady = 10)

    def onSelectObj(self, val):
        sender = val.widget
        idx = sender.curselection()
        value = sender.get(idx)
        self.objectSelected = self.objectList[idx[0]][0]
        self.objectSelectedIndex = idx[0]
        # shows the picture of object
        imageChosen = self.objectImages[idx[0]]
        path = 'Yues_class_definition/GUI_images/' + imageChosen
        imageInLbl = ImageTk.PhotoImage(Image.open(path).resize((200,200)))
        self.objectChosenLbl.configure(image=imageInLbl)
        self.objectChosenLbl.image = imageInLbl
        # shows the picture of area
        imageChosen = self.objectInAreaName[idx[0]][1]
        path = 'Yues_class_definition/GUI_images/' + imageChosen
        imageInLbl = ImageTk.PhotoImage(Image.open(path).resize((200,200)))
        self.areaCurrentLbl.configure(image=imageInLbl)
        self.areaCurrentLbl.image = imageInLbl
        
        # self.objectChosenName = self.objectNames[self.objectSelectedIndex]
        # print self.objectChosenName
        self.xPosDisp.set(round(self.objectSelected.GetTransform()[0][3])) # display X position 
        self.yPosDisp.set(round(self.objectSelected.GetTransform()[1][3])) # display Y position 
        self.zPosDisp.set(round(self.objectSelected.GetTransform()[2][3])) # display Z position 
        self.xRotScale.set(self.objectRotDegCurrent[idx[0]][0])     # display X rotation 
        self.yRotScale.set(self.objectRotDegCurrent[idx[0]][1])     # display Y rotation 
        self.zRotScale.set(self.objectRotDegCurrent[idx[0]][2])     # display Z rotation 

    def onSelectArea(self, val):
        sender = val.widget
        idx = sender.curselection()
        value = sender.get(idx)
        self.areaSelectedIndex = idx[0]
        imageChosen = self.areaList[idx[0]][2]
        path = 'Yues_class_definition/GUI_images/' + imageChosen
        imageInLbl = ImageTk.PhotoImage(Image.open(path).resize((200,200)))
        self.areaChosenLbl.configure(image=imageInLbl)
        self.areaChosenLbl.image = imageInLbl

    def xTransMinus(self):
        xNew = self.objectSelected.GetTransform()[0][3] - float(self.xTransEntry.get())
        objectTransformOld = self.objectSelected.GetTransform()
        objectTransformOld[0,3] = xNew
        self.objectSelected.SetTransform(objectTransformOld)
        self.xPosDisp.set(round(self.objectSelected.GetTransform()[0][3],2))
        self.objectInAreaName[self.objectSelectedIndex] = self.findObjectArea(self.objectSelected)
        # update the area picture
        imageChosen = self.objectInAreaName[self.objectSelectedIndex][1]
        path = 'Yues_class_definition/GUI_images/' + imageChosen
        imageInLbl = ImageTk.PhotoImage(Image.open(path).resize((200,200)))
        self.areaCurrentLbl.configure(image=imageInLbl)
        self.areaCurrentLbl.image = imageInLbl

    def xTransPlus(self):
        xNew = self.objectSelected.GetTransform()[0][3] + float(self.xTransEntry.get())
        objectTransformOld = self.objectSelected.GetTransform()
        objectTransformOld[0,3] = xNew
        self.objectSelected.SetTransform(objectTransformOld)
        self.xPosDisp.set(round(self.objectSelected.GetTransform()[0][3],2))
        self.objectInAreaName[self.objectSelectedIndex] = self.findObjectArea(self.objectSelected)
        # update the area picture
        imageChosen = self.objectInAreaName[self.objectSelectedIndex][1]
        path = 'Yues_class_definition/GUI_images/' + imageChosen
        imageInLbl = ImageTk.PhotoImage(Image.open(path).resize((200,200)))
        self.areaCurrentLbl.configure(image=imageInLbl)
        self.areaCurrentLbl.image = imageInLbl

    def yTransMinus(self):
        yNew = self.objectSelected.GetTransform()[1][3] - float(self.yTransEntry.get())
        objectTransformOld = self.objectSelected.GetTransform()
        objectTransformOld[1,3] = yNew
        self.objectSelected.SetTransform(objectTransformOld)
        self.yPosDisp.set(round(self.objectSelected.GetTransform()[1][3],2))
        self.objectInAreaName[self.objectSelectedIndex] = self.findObjectArea(self.objectSelected)
        # update the area picture
        imageChosen = self.objectInAreaName[self.objectSelectedIndex][1]
        path = 'Yues_class_definition/GUI_images/' + imageChosen
        imageInLbl = ImageTk.PhotoImage(Image.open(path).resize((200,200)))
        self.areaCurrentLbl.configure(image=imageInLbl)
        self.areaCurrentLbl.image = imageInLbl

    def yTransPlus(self):
        yNew = self.objectSelected.GetTransform()[1][3] + float(self.yTransEntry.get())
        objectTransformOld = self.objectSelected.GetTransform()
        objectTransformOld[1,3] = yNew
        self.objectSelected.SetTransform(objectTransformOld)
        self.yPosDisp.set(round(self.objectSelected.GetTransform()[1][3],2))
        self.objectInAreaName[self.objectSelectedIndex] = self.findObjectArea(self.objectSelected)
        # update the area picture
        imageChosen = self.objectInAreaName[self.objectSelectedIndex][1]
        path = 'Yues_class_definition/GUI_images/' + imageChosen
        imageInLbl = ImageTk.PhotoImage(Image.open(path).resize((200,200)))
        self.areaCurrentLbl.configure(image=imageInLbl)
        self.areaCurrentLbl.image = imageInLbl


    def zTransMinus(self):
        zNew = self.objectSelected.GetTransform()[2][3] - float(self.zTransEntry.get())
        objectTransformOld = self.objectSelected.GetTransform()
        objectTransformOld[2,3] = zNew
        self.objectSelected.SetTransform(objectTransformOld)
        self.zPosDisp.set(round(self.objectSelected.GetTransform()[2][3],2))
        self.objectInAreaName[self.objectSelectedIndex] = self.findObjectArea(self.objectSelected)
        # update the area picture
        imageChosen = self.objectInAreaName[self.objectSelectedIndex][1]
        path = 'Yues_class_definition/GUI_images/' + imageChosen
        imageInLbl = ImageTk.PhotoImage(Image.open(path).resize((200,200)))
        self.areaCurrentLbl.configure(image=imageInLbl)
        self.areaCurrentLbl.image = imageInLbl

    def zTransPlus(self):
        zNew = self.objectSelected.GetTransform()[2][3] + float(self.zTransEntry.get())
        objectTransformOld = self.objectSelected.GetTransform()
        objectTransformOld[2,3] = zNew
        self.objectSelected.SetTransform(objectTransformOld)
        self.zPosDisp.set(round(self.objectSelected.GetTransform()[2][3],2))
        self.objectInAreaName[self.objectSelectedIndex] = self.findObjectArea(self.objectSelected)
        # update the area picture
        imageChosen = self.objectInAreaName[self.objectSelectedIndex][1]
        path = 'Yues_class_definition/GUI_images/' + imageChosen
        imageInLbl = ImageTk.PhotoImage(Image.open(path).resize((200,200)))
        self.areaCurrentLbl.configure(image=imageInLbl)
        self.areaCurrentLbl.image = imageInLbl

    def xRotOnScale(self, val):
        self.xRotDegree = float(val)/180*pi
        self.xRotDisp.set(int(float(val)))
        self.objectRotDegCurrent[self.objectSelectedIndex][0] = float(val)
        self.rotateObject()
        # update the area picture
        imageChosen = self.objectInAreaName[self.objectSelectedIndex][1]
        path = 'Yues_class_definition/GUI_images/' + imageChosen
        imageInLbl = ImageTk.PhotoImage(Image.open(path).resize((200,200)))
        self.areaCurrentLbl.configure(image=imageInLbl)
        self.areaCurrentLbl.image = imageInLbl

    def yRotOnScale(self, val):
        self.yRotDegree = float(val)/180*pi
        self.yRotDisp.set(int(float(val)))
        self.objectRotDegCurrent[self.objectSelectedIndex][1] = float(val)
        self.rotateObject()
        # update the area picture
        imageChosen = self.objectInAreaName[self.objectSelectedIndex][1]
        path = 'Yues_class_definition/GUI_images/' + imageChosen
        imageInLbl = ImageTk.PhotoImage(Image.open(path).resize((200,200)))
        self.areaCurrentLbl.configure(image=imageInLbl)
        self.areaCurrentLbl.image = imageInLbl

    def zRotOnScale(self, val):
        self.zRotDegree = float(val)/180*pi
        self.zRotDisp.set(int(float(val)))
        self.objectRotDegCurrent[self.objectSelectedIndex][2] = float(val)
        self.rotateObject()
        # update the area picture
        imageChosen = self.objectInAreaName[self.objectSelectedIndex][1]
        path = 'Yues_class_definition/GUI_images/' + imageChosen
        imageInLbl = ImageTk.PhotoImage(Image.open(path).resize((200,200)))
        self.areaCurrentLbl.configure(image=imageInLbl)
        self.areaCurrentLbl.image = imageInLbl

    def rotateObject(self):
        objectTransformOld = np.eye(4)
        position = numpy.zeros([4,4])
        position[0:3,3] = self.objectSelected.GetTransform()[0:3,3]
        objectTransformOld[0:3,0:3] = self.objectInitRotMatrix[self.objectSelectedIndex]
        dotProduct1 = dot(giveRotationMatrix3D_4X4('y',self.yRotDegree), giveRotationMatrix3D_4X4('x',self.xRotDegree))
        dotProduct2 = dot(giveRotationMatrix3D_4X4('z',self.zRotDegree), dotProduct1)
        objectTransformNew = dotProduct2 + position
        self.objectSelected.SetTransform(objectTransformNew)

    def areaMoveYes(self):
        self.posBeforeMove = self.objectSelected.GetTransform()
        objectTransformAfterMove = self.objectSelected.GetTransform()
        objectTransformAfterMove[0:3,3] = self.placementPosInArea[self.areaSelectedIndex][0:3]
        self.objectSelected.SetTransform(objectTransformAfterMove)
        self.objectInAreaName[self.objectSelectedIndex] = self.findObjectArea(self.objectSelected)

    def areaMoveNo(self):
        self.objectSelected.SetTransform(self.posBeforeMove)

    def findObjectArea(self, obj):
        for i_area in range(shape(self.areaList)[0]):
            xDeltaInArea = self.areaList[i_area][4][0]
            yDeltaInArea = self.areaList[i_area][4][1]
            xMinInArea = self.areaList[i_area][0].GetTransform()[0][3]-float(xDeltaInArea/2)
            xMaxInArea = self.areaList[i_area][0].GetTransform()[0][3]+float(xDeltaInArea/2)
            yMinInArea = self.areaList[i_area][0].GetTransform()[1][3]-float(yDeltaInArea/2)
            yMaxInArea = self.areaList[i_area][0].GetTransform()[1][3]+float(yDeltaInArea/2)
            objPosX = obj.GetTransform()[0][3]
            objPosY = obj.GetTransform()[1][3]
            if objPosX<xMaxInArea and objPosX>xMinInArea and objPosY<yMaxInArea and objPosY>yMinInArea:
                return [self.areaList[i_area][1], self.areaList[i_area][2]]
        return ['None', 'other_areas.png']

    def saveData(self):
        print 'Done!'
        #parent.destroy()