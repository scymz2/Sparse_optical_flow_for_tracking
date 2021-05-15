#!/usr/bin/python
# -*- coding: UTF-8 -*-

import sys

from PyQt5 import QtGui, QtCore
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QMessageBox, QDesktopWidget, QFileDialog, \
    QLabel, QRadioButton, QHBoxLayout
from optical_flow.standard_flow import*
from optical_flow.trackPoint import *
from optical_flow.trackObject import *
from optical_flow.short_flow import *


class App(QWidget):
    def __init__(self):
        super().__init__()
        self.title = "Optical Flow"
        self.videoPath = 'campus1.mp4'
        self.NUM_OF_FEATURE_POINTS = 200
        self.left = 10
        self.top = 10
        self.width = 500
        self.height = 350
        self.button1 = QPushButton('Standard', self)
        self.button2 = QPushButton('Track Points', self)
        self.button3 = QPushButton('Track Selected Objects', self)
        self.button4 = QPushButton('Track Moving Objects', self)
        self.btn1 = QRadioButton('Video1')
        self.btn2 = QRadioButton('Video2')
        self.initUI()
        self.center()

    def center(self):
        # Put the window  to the center of the screen
        screen = QDesktopWidget().screenGeometry()
        size = self.geometry()
        self.move((screen.width() - size.width()) / 2, (screen.height() - size.height()) / 2)

    def initUI(self):

        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        # One vertical layout
        layout = QVBoxLayout()
        # Two horizontal layout
        hlayout = QHBoxLayout()
        hlayout2 = QHBoxLayout()

        # Init picture 1
        label1 = QLabel()
        label1.setFixedSize(300, 200)
        pix1 = QtGui.QPixmap('campus1.jpg')
        label1.setPixmap(pix1)
        label1.setScaledContents(True)

        # Init picture 2
        label2 = QLabel()
        label2.setFixedSize(300, 200)
        pix2 = QtGui.QPixmap('test_video2.jpg')
        label2.setPixmap(pix2)
        label2.setScaledContents(True)

        # Init Radio Button
        self.btn1.setChecked(True)
        self.btn1.toggled.connect(lambda: self.btnstate(self.btn1))
        self.btn2.toggled.connect(lambda: self.btnstate(self.btn2))

        # Initial button 1
        self.button1.setToolTip("This is the mode for a standard optical flow")
        self.button1.clicked.connect(lambda: self.onClickButton())
        # Initial button 2
        self.button2.setToolTip("You can click to set the points to track in this mode")
        self.button2.clicked.connect(lambda: self.onClickButton())
        # Initial button 3
        self.button3.setToolTip("You can draw a rectangle and track the optical flow of the objects in it")
        self.button3.clicked.connect(lambda: self.onClickButton())
        # Initial button 4
        self.button4.setToolTip("The moving objects are tracked and their optical flow are displayed in this mode")
        self.button4.clicked.connect(lambda: self.onClickButton())

        # Init title text for the box
        label3 = QLabel("SPARSE OPTICAL FLOW FOR TRACKING")
        label3.setFont(QFont('Times', 12, QFont.Bold))
        layout.addWidget(label3, alignment=QtCore.Qt.AlignCenter)

        # Add the pictures and radio button to the horizontal layouts
        hlayout.addWidget(label1, alignment=QtCore.Qt.AlignCenter)
        hlayout.addWidget(label2, alignment=QtCore.Qt.AlignCenter)
        hlayout2.addWidget(self.btn1, alignment=QtCore.Qt.AlignCenter)
        hlayout2.addWidget(self.btn2, alignment=QtCore.Qt.AlignCenter)
        hwg = QWidget()
        hwg1 = QWidget()
        hwg.setLayout(hlayout)
        hwg1.setLayout(hlayout2)

        # Add the horizontal layout to the vertical layout
        layout.addWidget(hwg)
        layout.addWidget(hwg1)

        # Add the mode buttons to the vertical layout
        layout.addWidget(self.button1)
        layout.addWidget(self.button2)
        layout.addWidget(self.button3)
        layout.addWidget(self.button4)

        # Add author information to the vertical layout
        label4 = QLabel("Author: Mochuan Zhan", alignment=QtCore.Qt.AlignRight)
        label5 = QLabel("Email: scymz2@nottingham.edu.cn", alignment=QtCore.Qt.AlignRight)
        label6 = QLabel("ID: 20030024", alignment=QtCore.Qt.AlignRight)
        layout.addWidget(label4)
        layout.addWidget(label5)
        layout.addWidget(label6)

        # Set app layout
        self.setLayout(layout)
        self.show()

    # button listener to select mode
    def onClickButton(self):
        sender = self.sender()
        clickEvent = sender.text()
        if clickEvent == 'Standard':
            reply = QMessageBox.information(self, 'Notice', 'This is standard sparse optical flow, press Q to quit', QMessageBox.Ok, QMessageBox.Cancel)
            if reply == QMessageBox.Ok:
                run_standard(self.videoPath, self.NUM_OF_FEATURE_POINTS)
            elif reply == QMessageBox.Cancel:
                print("")
        elif clickEvent == 'Track Points':
            reply = QMessageBox.information(self, 'Notice', 'You can click on the video to set points', QMessageBox.Ok, QMessageBox.Cancel)
            if reply == QMessageBox.Ok:
                run_track_points(self.videoPath)
            elif reply == QMessageBox.Cancel:
                print("")
        elif clickEvent == 'Track Selected Objects':
            reply = QMessageBox.information(self, 'Notice', 'You can press SPACE to pause the video and draw a rectangle to track selected object ', QMessageBox.Ok, QMessageBox.Cancel)
            if reply == QMessageBox.Ok:
                run_track_object(self.videoPath)
            elif reply == QMessageBox.Cancel:
                print("")
        elif clickEvent == 'Track Moving Objects':
            reply = QMessageBox.information(self, 'Notice', 'The moving object will be automatically tracked and their optical flow will be displayed', QMessageBox.Ok, QMessageBox.Cancel)
            if reply == QMessageBox.Ok:
                run_moving_objects(self.videoPath, self.NUM_OF_FEATURE_POINTS)
            elif reply == QMessageBox.Cancel:
                print("")

    # Button listner for radio buttons to select video
    def btnstate(self, btn):
        if btn.text() == 'Video1':
            if btn.isChecked() == True:
                self.videoPath = 'campus1.mp4'
                self.NUM_OF_FEATURE_POINTS = 200

        if btn.text() == "Video2":
            if btn.isChecked() == True:
                self.videoPath = 'test_video2.mp4'
                self.NUM_OF_FEATURE_POINTS = 10000


if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    app.setStyleSheet("QPushButton { padding: 2ex;}")
    ex = App()
    sys.exit(app.exec_())

