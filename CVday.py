####################################
####search(Ctrl+F1) here1, here2####
####################################

import sys
import os
import time
from tkinter import messagebox
from tkinter.filedialog import askopenfilename
from tkinter import *
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtGui import *
from PyQt5.QtGui import QStandardItem, QStandardItemModel, QPixmap, QColor
from PyQt5.QtWidgets import *
from PyQt5.QtWidgets import (QApplication, QMainWindow, QMessageBox, QWidget, QHBoxLayout, QVBoxLayout, QGridLayout, QLabel, QListView, QLineEdit, QComboBox, QProgressBar, QSlider, QLCDNumber, QPushButton)
from PIL import Image
from PIL.ImageQt import *
import cv2
import numpy as np
import tensorflow as tf
tf.set_random_seed(19)
from tools.cyclegan.model import cyclegan
from tools.pix2pix import pix2pix as ptp
from tools.neuraldoodle import face_labeling
from tools.neuraldoodle import doodle
import argparse

srcPath='./src/img.jpg'
stlPath='./src/stlimg.jpg'
dstPath=''
crtPath=''
imgLabel=''
imgLabelBack=''
imgEdit=''
winLabel=''
winLcd=''
imgStyle='vangogh'
imgEffect='└hue'
imgValue=50
imgFormat='jpg'
img=''
pix_class = ptp.pix2pixClass()

class MainWindow(QDialog):

    def __init__(self):
        super().__init__()
        self.init_UI()
        self.setGeometry(240, 135, 1440, 810)
        self.setWindowFlags(Qt.FramelessWindowHint)
        self.oldPos = self.pos()
        self.show()

    def mousePressEvent(self, event):
        self.oldPos = event.globalPos()

    def mouseMoveEvent(self, event):
        delta = QPoint (event.globalPos() - self.oldPos)
        self.move(self.x() + delta.x(), self.y() + delta.y())
        self.oldPos = event.globalPos()

    def init_UI(self):
        self.w1 = QWidget(parent=self, flags=Qt.Widget)
        self.w1.setGeometry(0, 25, 1440, 750)
        self.w1.setStyleSheet("background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,"
                              "stop: 0 rgb(30,30,30), stop: 1 rgb(45,45,45));"
                              "border-style: solid;"
                              "border-color: rgb(15,15,15);"
                              "border-width: 1px;")

        self.w2 = QWidget(parent=self, flags=Qt.Widget)
        self.w2.setGeometry(1, 722, 1440, 45)
        self.w2.setStyleSheet("background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,"
                              "stop: 0 rgb(55,55,55), stop: 1 rgb(60,60,60));")

        self.w3 = QWidget(parent=self, flags=Qt.Widget)
        self.w3.setGeometry(1090, 25, 350, 750)
        self.w3.setStyleSheet("background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,"
                              "stop: 0 rgb(70,70,70), stop: 1 rgb(85,85,85));"
                              "border-style: solid;"
                              "border-color: rgb(15,15,15);"
                              "border-width: 1px;")
        self.shadow1 = QGraphicsDropShadowEffect()
        self.shadow1.setBlurRadius(6)
        self.shadow1.setColor(QtGui.QColor(0, 0, 0, 100))
        self.shadow1.setXOffset(-1)
        self.shadow1.setYOffset(0)
        self.w3.setGraphicsEffect(self.shadow1)

        self.w4 = QWidget(parent=self, flags=Qt.Widget)
        self.w4.setGeometry(0, 766, 1440, 44)
        self.w4.setStyleSheet("background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,"
                              "stop: 0 rgb(35,35,35), stop: 1 rgb(20,20,20));"
                              "border-style: solid;"
                              "border-color: rgb(15,15,15);"
                              "border-width: 1px;")
        self.shadow2 = QGraphicsDropShadowEffect()
        self.shadow2.setBlurRadius(6)
        self.shadow2.setColor(QtGui.QColor(0, 0, 0, 100))
        self.shadow2.setXOffset(0)
        self.shadow2.setYOffset(-1)
        self.w4.setGraphicsEffect(self.shadow2)

        self.w5 = QWidget(parent=self, flags=Qt.Widget)
        self.w5.setGeometry(0, 0, 1440, 27)
        self.w5.setStyleSheet("background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,"
                              "stop: 0 rgb(55,55,55), stop: 1 rgb(40,40,40));"
                              "border-style: solid;"
                              "border-color: rgb(15,15,15);"
                              "border-width: 1px;")
        self.shadow3 = QGraphicsDropShadowEffect()
        self.shadow3.setBlurRadius(3)
        self.shadow3.setColor(QtGui.QColor(0, 0, 0, 50))
        self.shadow3.setXOffset(0)
        self.shadow3.setYOffset(1)
        self.w5.setGraphicsEffect(self.shadow3)

        mBtn1 = QPushButton("Photo Style Transfer Software", self)
        mBtn1.move(4, 2)
        mBtn1.setStyleSheet("QPushButton{background-color: rgba(0,0,0,0);"
                              "color: rgb(190,190,190);"
                              "font: bold;"
                              "font-size: 12px;"
                              "font-family: Arial;}")

        mBtn2 = QPushButton("About", self)
        mBtn2.move(1339, 2)
        mBtn2.clicked.connect(self.on_about)
        mBtn2.setStyleSheet("QPushButton{background-color: rgba(0,0,0,0);"
                            "color: rgb(190,190,190);"
                            "font: bold;"
                            "font-size: 12px;"
                            "font-family: Arial;}")

        mBtn3 = QPushButton("Exit", self)
        mBtn3.move(1381, 2)
        mBtn3.clicked.connect(self.on_exit)
        mBtn3.setStyleSheet("QPushButton{background-color: rgba(0,0,0,0);"
                            "color: rgb(190,190,190);"
                            "font: bold;"
                            "font-size: 12px;"
                            "font-family: Arial;}")

        label=QLabel(self)
        global imgLabelBack
        imgLabelBack = QLabel(self)
        global imgLabel
        imgLabel = QLabel(self)
        pixmap = QPixmap("main.png")#interface
        label.setPixmap(pixmap)
        label.setGeometry(20, 45, 1050, 636)

        global imgEdit
        imgEdit = QLineEdit(self)
        imgEdit.setText("C:/")
        imgEdit.setGeometry(20,777,546,22)
        imgEdit.setStyleSheet("QLineEdit{background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,"
                            "stop: 0 rgb(180,180,180), stop: 1 rgb(220,220,220));"
                            "border-style: solid;"
                            "border-color: rgb(5,5,5);"
                            "border-width: 1px;"
                            "border-radius: 2px;"
                            "color: rgb(35,35,35);"
                            "font: bold;"
                            "font-size: 10px;"
                            "font-family: Arial;}")

        combo = QComboBox(self)
        combo.addItem("JPG")
        combo.addItem("BMP")
        combo.addItem("PNG")
        combo.addItem("GIF")
        combo.setGeometry(1111,777,200,22)
        combo.activated[str].connect(self.on_format)
        combo.setStyleSheet("QComboBox{background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,"
                           "stop: 0 rgb(180,180,180), stop: 1 rgb(220,220,220));"
                           "border-style: solid;"
                           "border-color: rgb(5,5,5);"
                           "border-width: 1px;"
                           "border-radius: 2px;"
                           "color: rgb(35,35,35);"
                           "font: bold;"
                           "font-size: 10px;"
                           "font-family: Arial;}")

        ###################
        ####here1 start####
        ## 콤보박스에 넣을 이름, addItem에 넣으면 됨 모델명으로 하는 거 추천
        ## 초기값이 안들어가 있어서 vangogh로 되어있어도 맨처음에 하나 클릭 해야됨
        list = QListWidget(self)
        list.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        list.setViewMode(QtWidgets.QListView.IconMode)
        list.setIconSize(QtCore.QSize(220, 170))
        for image, text in [("widget.png", "Vangogh"),
                            ("widget (2).png", "Ukiyoe"),
                            ("widget (3).png", "Night To Day"),
                            ("widget (4).png", "Black-White To Color"),
                            ("widget (5).png", "Neural Doodle")]:
            item = QtWidgets.QListWidgetItem(text, list)
            icon = QtGui.QIcon()
            icon.addPixmap(QtGui.QPixmap(image), QtGui.QIcon.Normal, QtGui.QIcon.Off)
            item.setIcon(icon)
            item.setSizeHint(QtCore.QSize(240, 200))
        list.setGeometry(1108, 45, 310, 660)
        list.itemClicked.connect(self.on_styleList)
        list.setStyleSheet("QListWidget{background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,"
                           "stop: 0 rgb(220,220,220), stop: 1 rgb(180,180,180));"
                           "border-style: solid;"
                           "border-color: rgb(55,55,55);"
                           "border-width: 1px;"
                           "border-radius: 3px;"
                           "color: rgb(35,35,35);"
                           "font: bold;"
                           "font-size: 12px;"
                           "font-family: Arial;}")
        btn7 = QPushButton("Apply", self)
        btn7.setGeometry(1108, 718,150,30)
        btn7.clicked.connect(self.on_styleBtn)
        btn7.setStyleSheet("QPushButton{background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,"
                           "stop: 0 rgb(95,95,95), stop: 1 rgb(85,85,85));"
                           "border-style: solid;"
                           "border-color: rgb(65,65,65);"
                           "border-width: 1px;"
                           "border-radius: 3px;"
                           "color: rgb(190,190,190);"
                           "font: bold;"
                           "font-size: 12px;"
                           "font-family: Arial;}")
        self.shadow5 = QGraphicsDropShadowEffect()
        self.shadow5.setBlurRadius(4)
        self.shadow5.setColor(QtGui.QColor(0, 0, 0, 80))
        self.shadow5.setXOffset(1)
        self.shadow5.setYOffset(1)
        btn7.setGraphicsEffect(self.shadow5)
        ####here1 end####
        #################

        btn1 = QPushButton("●", self)
        btn1.setGeometry(1035, 692,35,20)
        btn1.clicked.connect(self.on_scale)
        btn1.setStyleSheet("QPushButton{background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,"
                           "stop: 0 rgb(55,55,55), stop: 1 rgb(45,45,45));"
                           "border-style: solid;"
                           "border-color: rgb(25,25,25);"
                           "border-width: 1px;"
                           "border-radius: 3px;"
                           "color: rgb(180,180,180);"
                           "font: bold;"
                           "font-size: 10px;"
                           "font-family: Arial;}")

        btn2 = QPushButton("More", self)
        btn2.setGeometry(1268, 718,150,30)
        btn2.clicked.connect(self.on_effect)
        btn2.setStyleSheet("QPushButton{background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,"
                           "stop: 0 rgb(95,95,95), stop: 1 rgb(85,85,85));"
                           "border-style: solid;"
                           "border-color: rgb(65,65,65);"
                           "border-width: 1px;"
                           "border-radius: 3px;"
                           "color: rgb(190,190,190);"
                           "font: bold;"
                           "font-size: 12px;"
                           "font-family: Arial;}")
        self.shadow4 = QGraphicsDropShadowEffect()
        self.shadow4.setBlurRadius(4)
        self.shadow4.setColor(QtGui.QColor(0, 0, 0, 80))
        self.shadow4.setXOffset(1)
        self.shadow4.setYOffset(1)
        btn2.setGraphicsEffect(self.shadow4)

        btn3 = QPushButton("Open", self)
        btn3.setGeometry(587, 777,35,21)
        btn3.clicked.connect(self.on_open)
        btn3.setStyleSheet("QPushButton{background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,"
                           "stop: 0 rgb(35,35,35), stop: 1 rgb(20,20,20));"
                           "border-style: solid;"
                           "border-color: rgb(5,5,5);"
                           "border-width: 1px;"
                           "border-radius: 2px;"
                           "color: rgb(170,170,170);"
                           "font: bold;"
                           "font-size: 10px;"
                           "font-family: Arial;}")

        btn4 = QPushButton("Save", self)
        btn4.setGeometry(1327, 777,35,21)
        btn4.clicked.connect(self.on_save)
        btn4.setStyleSheet("QPushButton{background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,"
                           "stop: 0 rgb(35,35,35), stop: 1 rgb(20,20,20));"
                           "border-style: solid;"
                           "border-color: rgb(5,5,5);"
                           "border-width: 1px;"
                           "border-radius: 2px;"
                           "color: rgb(170,170,170);"
                           "font: bold;"
                           "font-size: 10px;"
                           "font-family: Arial;}")

        btn5 = QPushButton("◀", self)
        btn5.setGeometry(484, 690,60,22)
        btn5.clicked.connect(self.on_bef)
        btn5.setStyleSheet("QPushButton{background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,"
                           "stop: 0 rgb(55,55,55), stop: 1 rgb(45,45,45));"
                           "border-style: solid;"
                           "border-color: rgb(25,25,25);"
                           "border-width: 1px;"
                           "border-radius: 3px;"
                           "color: rgb(180,180,180);"
                           "font: bold;"
                           "font-size: 10px;"
                           "font-family: Arial;}")

        btn6 = QPushButton("▶", self)
        btn6.setGeometry(548, 690,60,22)
        btn6.clicked.connect(self.on_aft)
        btn6.setStyleSheet("QPushButton{background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,"
                           "stop: 0 rgb(55,55,55), stop: 1 rgb(45,45,45));"
                           "border-style: solid;"
                           "border-color: rgb(25,25,25);"
                           "border-width: 1px;"
                           "border-radius: 3px;"
                           "color: rgb(180,180,180);"
                           "font: bold;"
                           "font-size: 10px;"
                           "font-family: Arial;}")

    def on_about(self):
        self.next = AboutWindow()

    def on_exit(self):
        message = QMessageBox.question(self, 'Notice', "Are you sure to quit?", QMessageBox.Yes|QMessageBox.No, QMessageBox.No)
        if message == QMessageBox.Yes:
            self.close()
        else:
            pass

    def on_bef(self):
        global srcPath
        global crtPath
        crtPath=srcPath
        pixmap = QPixmap(srcPath)
        smaller_pixmap = pixmap.scaled(1050, 636, Qt.KeepAspectRatio, Qt.FastTransformation)
        global imgLabel
        imgLabel.setPixmap(smaller_pixmap)
        imgLabel.setGeometry(20, 45, 1050, 636)
        imgLabel.show()

    def on_aft(self):
        global dstPath
        global crtPath
        crtPath=dstPath
        pixmap = QPixmap(dstPath)
        smaller_pixmap = pixmap.scaled(1050, 636, Qt.KeepAspectRatio, Qt.FastTransformation)
        global imgLabel
        imgLabel.setPixmap(smaller_pixmap)
        imgLabel.setGeometry(20, 45, 1050, 636)
        imgLabel.show()

    def on_scale(self):
        self.next = ScaleWindow()

    def on_open(self):
        try:
            root = Tk()
            root.withdraw()
            root.filename = filedialog.askopenfilename(initialdir="C:/", title="Open", filetypes=(("jpeg files", "*.jpg"), ("all files", "*.*")))
            global srcPath
            pixmap = QPixmap(root.filename)
            img=pixmap
            img.save(srcPath)
            smaller_pixmap = pixmap.scaled(1050, 636, Qt.KeepAspectRatio, Qt.FastTransformation)
            global imgLabelBack
            pixmap2 = QPixmap("main (2).png")  # interface
            imgLabelBack.setPixmap(pixmap2)
            imgLabelBack.setGeometry(19, 44, 1063, 656)
            imgLabelBack.show()
            global imgLabel
            imgLabel.setAlignment(Qt.AlignCenter)
            imgLabel.setPixmap(smaller_pixmap)
            imgLabel.setGeometry(20, 45, 1050, 636)
            imgLabel.show()
            global imgEdit
            imgEdit.setText(root.filename)
            imgEdit.setGeometry(20,777,546,22)
            imgEdit.show()
        except:
             pass

    def on_effect(self):
        self.next = EffectWindow()

    def on_format(self, item):
        global imgFormat
        if item == "JPG":
            imgFormat = "jpg"
        elif item == "BMP":
            imgFormat = "bmp"
        elif item == "PNG":
            imgFormat = "png"
        elif item == "GIF":
            imgFormat = "gif"

    ###################
    ####here2 start####
    ## 내경우는 item이랑 imgstyle이랑 통일함
    def on_styleList(self, item):
        global imgStyle
        global stlPath
        global srcPath
        if item.text() == "Vangogh":
            imgStyle = "vangogh"
        elif item.text() == "Ukiyoe":
            imgStyle = "ukiyoe"
        elif item.text() == "Night To Day":
            imgStyle = "night2day"
        elif item.text() == "Black-White To Color":
            imgStyle = "bw2color"
        elif item.text() == "Neural Doodle":
            imgStyle = "neuraldoodle" # 명석오빠꺼
            message = QMessageBox.question(self, 'Notice', "You must choose a style image!", QMessageBox.Yes | QMessageBox.No,QMessageBox.No)
            if message == QMessageBox.Yes:
                try:
                    ## style_img 선택 (input_img는 맨 처음 open 버튼으로 선택되어있는 상태)
                    root = Tk()
                    root.withdraw()
                    root.filename = filedialog.askopenfilename(initialdir="C:/", title="Open",filetypes=(("jpeg files", "*.jpg"), ("all files", "*.*")))
                    global imgEdit
                    imgEdit.setText(imgEdit.text()+"; "+root.filename)
                    imgEdit.setGeometry(20, 777, 546, 22)
                    imgEdit.show()
                    ## style_img는 src 폴더에 stlimg 파일로 저장됨 (input_img는 src 폴더에 img 파일로 저장되어있는 상태)
                    pixmap = QPixmap(root.filename)
                    img = pixmap
                    img.save(stlPath)
                    img1 = Image.open(srcPath)
                    img2 = Image.open(stlPath)
                    (width1, height1) = img1.size
                    (width2, height2) = img2.size
                    width3 = width1 + width2 + 20
                    height3 = max(height1, height2)
                    img3 = Image.new('RGB', (width3, height3))
                    img3.paste(im=img1, box=(0, 0))
                    img3.paste(im=img2, box=(width1+20, 0))
                    qimg=ImageQt(img3)
                    pimg=QPixmap.fromImage(qimg).copy()
                    smaller_pixmap = pimg.scaled(1050, 636, Qt.KeepAspectRatio, Qt.FastTransformation)
                    global imgLabel
                    imgLabel.setAlignment(Qt.AlignCenter)
                    imgLabel.setPixmap(smaller_pixmap)
                    imgLabel.setGeometry(20, 45, 1050, 636)
                    imgLabel.show()
                except:
                    pass
            else:
                pass

    def on_styleBtn(self):
        global imgStyle
        global imgLabel
        global dstPath
        global srcPath
        global stlPath
        # sp=os.path.split(srcPath)
        # sp[0] directory
        # sp[1] filename
        if imgStyle == "vangogh":
            modelname='vangogh2photo'
            tfconfig=tf.ConfigProto(allow_soft_placement=True)
            tfconfig.gpu_options.allow_growth=True
            with tf.Session(config=tfconfig) as sess:
                model=cyclegan(sess,modelname,srcPath)
                dstPath=model.test()
            tf.contrib.keras.backend.clear_session()
            pixmap = QPixmap(dstPath)
            smaller_pixmap = pixmap.scaled(1050, 636, Qt.KeepAspectRatio, Qt.FastTransformation)
            imgLabel.setPixmap(smaller_pixmap)
            imgLabel.setGeometry(20, 45, 1050, 636)

        if imgStyle == "ukiyoe":
            modelname='ukiyoe2photo'
            tfconfig=tf.ConfigProto(allow_soft_placement=True)
            tfconfig.gpu_options.allow_growth=True
            tfconfig.gpu_options.per_process_gpu_memory_fraction=0.6
            with tf.Session(config=tfconfig) as sess:
                model=cyclegan(sess,modelname,srcPath)
                dstPath=model.test()
            ## tensorflow 쓰는 경우 아래처럼 session초기화 해줘야 에러 안남
            tf.contrib.keras.backend.clear_session()
            ## 밑에 주석 참고
            pixmap = QPixmap(dstPath)
            smaller_pixmap = pixmap.scaled(1050, 636, Qt.KeepAspectRatio, Qt.FastTransformation)
            imgLabel.setPixmap(smaller_pixmap)
            imgLabel.setGeometry(20, 45, 1050, 636)

        ## 나는 애초에 모델에서 저장을 해줌,
        ## img=QPixmap(srcPath)
        ## img.save(dstPath)
        ## 그대로 두면 원본이 덮어 씌워서
        ## pixmap = QPixmap(dstPath)
        ## imgLabel.setPixmap(pixmap)
        ## imgLabel.setGeometry(20, 45, 1050, 636)
        ## 로 바꿨음

        ## by 소연 : 인자는 input 경로, output 경로, checkpoint 경로만 받음
        ## 사진 띄우는거는 민기오빠 따라함^^
        if imgStyle == "night2day":
            dstPath = pix_class.main(input_dir=srcPath, output_dir='dst/night2day', checkpoint='model/night2day_train')
            if os.path.exists(dstPath):
                pixmap = QPixmap(dstPath)
                smaller_pixmap = pixmap.scaled(1050, 636, Qt.KeepAspectRatio, Qt.FastTransformation)
                imgLabel.setPixmap(smaller_pixmap)
                imgLabel.setGeometry(20, 45, 1050, 636)

        if imgStyle == "bw2color":
            dstPath = pix_class.main(input_dir=srcPath, output_dir='dst/bw2color', checkpoint='model/bw2color_train')
            if os.path.exists(dstPath):
                pixmap = QPixmap(dstPath)
                smaller_pixmap = pixmap.scaled(1050, 636, Qt.KeepAspectRatio, Qt.FastTransformation)
                imgLabel.setPixmap(smaller_pixmap)
                imgLabel.setGeometry(20, 45, 1050, 636)

        ## 명석오빠 붙이세요~~
        if imgStyle == "neuraldoodle":
            face_labeling.pix_class.main(input_dir='src/img.jpg', output_dir='src', checkpoint='model/facial_train')
            face_labeling.pix_class.main(input_dir='src/stlimg.jpg', output_dir='src', checkpoint='model/facial_train')
            input_img = cv2.imread('src/img.jpg')
            input_img = cv2.resize(input_img, dsize=(256, 256))
            cv2.imwrite('src/img.jpg', input_img)
            style_img = cv2.imread("src/stlimg.jpg")
            style_img = cv2.resize(style_img, dsize=(256, 256))
            cv2.imwrite('src/stlimg.jpg', style_img)
            doodle.args.content='src/img.jpg'
            doodle.args.style='src/stlimg.jpg'
            if not os.path.exists('dst/neuraldoodle'):
                os.makedirs('dst/neuraldoodle')
            doodle.args.output='dst/neuraldoodle/output.png'
            generator = doodle.NeuralGenerator()
            generator.run()
            pixmap = QPixmap('dst/neuraldoodle/output.png')
            smaller_pixmap = pixmap.scaled(1050, 636, Qt.KeepAspectRatio, Qt.FastTransformation)
            imgLabel.setPixmap(smaller_pixmap)
            imgLabel.setGeometry(20, 45, 1050, 636)
    ####here2 end####
    #################

    def on_save(self):
        global dstPath
        pixmap = QPixmap(dstPath)
        img=pixmap
        global imgFormat
        if imgFormat == "jpg":
            img.save('./result/img.jpg')
        if imgFormat == "bmp":
            img.save('./result/img.bmp')
        if imgFormat == "png":
            img.save('./result/img.png')
        if imgFormat == "gif":
            img.save('./result/img.gif')

class AboutWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setGeometry(571, 205, 778, 670)
        self.setWindowFlags(Qt.FramelessWindowHint)
        label = QLabel(self)
        pixmap = QPixmap("about.jpg")#splash
        label.setPixmap(pixmap)
        label.setGeometry(0, 0, 778, 670)
        self.init_window()

    def init_window(self):
        self.show()
        self.counter = 0
        self.timer = QTimer()
        self.timer.setInterval(1000)
        self.timer.timeout.connect(self.on_time)
        self.timer.start()

    def on_time(self):
        self.counter +=1
        if self.counter ==6:
            time.sleep(3)
            self.close()

class ScaleWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Scale")
        self.setGeometry(0, 0, 1920, 1080)
        self.init_window()

    def init_window(self):
        global crtPath
        pixmap = QPixmap(crtPath)
        bigger_pixmap = pixmap.scaled(1920, 1080, Qt.KeepAspectRatio, Qt.FastTransformation)
        label=QLabel(self)
        label.setAlignment(Qt.AlignCenter)
        label.setPixmap(bigger_pixmap)
        label.setGeometry(0, 0, 1920, 1080)
        self.show()

class EffectWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Effect")
        self.setGeometry(1218, 338, 171, 404)
        cv2.namedWindow("Effect", flags=cv2.WINDOW_GUI_NORMAL)
        cv2.moveWindow("Effect", 531, 307)
        cv2.resizeWindow("Effect", 667, 404)
        self.init_window()

    def closeEvent(self, *args, **kwargs):
        cv2.destroyAllWindows()

    def init_window(self):
        list = QListWidget(self)
        label = ('--Image Processing--',
                 '└filter',
                 '　└warm',
                 '　└cool',
                 '　└sepia',
                 '　└vintage',
                 '└color',
                 '└color-map',
                 '└color-inverse',
                 '└brightness',
                 '└contrast',
                 '└saturation',
                 '└sharp',
                 '└blur',
                 '--Photo Decoration--',
                 '└vignette')
        list.addItems(label)
        list.setGeometry(20, 20, 131, 284)
        list.itemClicked.connect(self.on_effectList)
        global winLcd
        winLcd=QLCDNumber(self)
        winLcd.display(50)
        winLcd.setGeometry(-14, 303, 60, 60)
        winLcd.setStyleSheet("""QLCDNumber{background-color:rgba(0,0,0,0);}""")
        sld = QSlider(Qt.Horizontal, self)
        sld.setRange(1,100)
        sld.setValue(50)
        sld.setGeometry(56, 324, 95, 20)
        sld.valueChanged.connect(self.on_effectSld)
        btn = QPushButton("Apply", self)
        btn.setGeometry(20, 354, 131, 30)
        btn.clicked.connect(self.on_effectBtn)
        self.show()

    def on_effectList(self, item):
        global imgEffect
        if item.text() == "　└warm":
            imgEffect = "warm"
        elif item.text() == "　└cool":
            imgEffect = "cool"
        elif item.text() == "　└sepia":
            imgEffect = "sepia"
        elif item.text() == "　└vintage":
            imgEffect = "vintage"
        elif item.text() == "└color":
            imgEffect = "color"
        elif item.text() == "└color-map":
            imgEffect = "colormap"
        elif item.text() == "└color-inverse":
            imgEffect = "colorinverse"
        elif item.text() == "└brightness":
            imgEffect = "brightness"
        elif item.text() == "└contrast":
            imgEffect = "contrast"
        elif item.text() == "└saturation":
            imgEffect = "saturation"
        elif item.text() == "└sharp":
            imgEffect = "sharp"
        elif item.text() == "└blur":
            imgEffect = "blur"
        elif item.text() == "└vignette":
            imgEffect = "vignette"
        else:
            imgEffect=''

    def on_effectSld(self,value):
        global imgValue
        imgValue = value
        global winLcd
        winLcd.display(imgValue)
        global imgEffect
        global img
        global dstPath
        global srcPath
        if imgEffect=="vignette":
            img = cv2.imread(dstPath, cv2.IMREAD_COLOR)
        else:
            img=cv2.imread(srcPath,cv2.IMREAD_COLOR)
        if imgEffect == "color":
            hsvimg = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsvimg)
            h.fill(imgValue*1.8)
            hsvimg = cv2.merge([h, s, v])
            img = cv2.cvtColor(hsvimg, cv2.COLOR_HSV2BGR)
        if imgEffect=="colormap":
            (B, G, R) = cv2.split(img)
            M = np.maximum(np.maximum(R, G), B)
            R[R < M] = 0
            G[G < M] = 0
            B[B < M] = 0
            img = cv2.merge([B, G, R])
        if imgEffect=="colorinverse":
            img=255-img
        if imgEffect == "brightness":
            hsvimg = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsvimg)
            v.fill(imgValue*2.55)
            hsvimg = cv2.merge([h, s, v])
            img = cv2.cvtColor(hsvimg, cv2.COLOR_HSV2BGR)
        if imgEffect=="contrast":
            labimg = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(labimg)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            labimg = cv2.merge((cl, a, b))
            img = cv2.cvtColor(labimg, cv2.COLOR_LAB2BGR)
        if imgEffect=="saturation":
            hsvimg = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsvimg)
            s.fill(imgValue*2.55)
            hsvimg = cv2.merge([h, s, v])
            img = cv2.cvtColor(hsvimg, cv2.COLOR_HSV2BGR)
        if imgEffect=="sharp":
            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            img = cv2.filter2D(img, -1, kernel)
        if imgEffect=="blur":
            img = cv2.blur(img, (imgValue, imgValue), anchor=(-1, -1), borderType=cv2.BORDER_DEFAULT)
        if imgEffect == "vignette":
            rows, cols = img.shape[:2]
            kernelx = cv2.getGaussianKernel(cols, 200)
            kernely = cv2.getGaussianKernel(rows, 200)
            kernel = kernely * kernelx.T
            mask = 255 * kernel / np.linalg.norm(kernel)
            for i in range(3):
                img[:, :, i] = img[:, :, i] * mask
        cv2.imshow('Effect', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def on_effectBtn(self):
        global imgEffect
        global img
        global dstPath
        global srcPath
        global imgLabel
        if imgEffect=="vignette":
            cv2.imwrite(dstPath, img)
            pixmap = QPixmap(dstPath)
        else:
            cv2.imwrite(srcPath,img)
            pixmap = QPixmap(srcPath)
        smaller_pixmap = pixmap.scaled(1050, 636, Qt.KeepAspectRatio, Qt.FastTransformation)
        imgLabel.setPixmap(smaller_pixmap)

class StartWindow(QMainWindow):

    def __init__(self, *args, **kwargs):
        super(StartWindow, self).__init__(*args, **kwargs)
        self.setGeometry(560, 240, 900, 600)
        self.setWindowFlags(Qt.FramelessWindowHint)
        label = QLabel(self)
        pixmap = QPixmap("start.jpg")#splash
        label.setPixmap(pixmap)
        label.setGeometry(0, 0, 900, 600)
        global winLabel
        winLabel = QLabel("Starting..",self)
        winLabel.setGeometry(407, 195, 200, 30)
        winLabel.setStyleSheet("color: rgb(255,255,255);"
                           "font-size: 12px;"
                           "font-family: Arial;}")
        self.show()
        self.counter = 0
        self.timer = QTimer()
        self.timer.setInterval(100)
        self.timer.timeout.connect(self.on_time)
        self.timer.start()

    def on_time(self):
        self.counter +=1
        if self.counter==1:
            winLabel.setText("Directed by MinKi,Song")
        if self.counter == 2:
            winLabel.setText("Directed by MyeongSuk,Yoon")
        if self.counter == 3:
            winLabel.setText("Directed by SoYeon,Jeon")
        if self.counter == 4:
            winLabel.setText("Directed by JuHeon,Jeong")
        if self.counter == 5:
            winLabel.setText("Completed..")
        if self.counter ==6:
            time.sleep(1)
            self.close()
            window = MainWindow()
            window.exec_()

app = QApplication([])
window = StartWindow()
app.exec_()