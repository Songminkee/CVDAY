####line175,348####

import sys
import os
import time
from tkinter import messagebox
from tkinter.filedialog import askopenfilename
from tkinter import *
from PyQt5.QtCore import *
from PyQt5.QtCore import Qt, QPoint
from PyQt5 import QtGui, QtWidgets
from PyQt5.QtGui import *
from PyQt5.QtGui import QStandardItem, QStandardItemModel, QPixmap, QColor
from PyQt5.QtWidgets import *
from PyQt5.QtWidgets import (QApplication, QMainWindow, QMessageBox, QWidget, QHBoxLayout, QVBoxLayout, QGridLayout, QLabel, QListView, QLineEdit, QComboBox, QProgressBar, QSlider, QLCDNumber, QPushButton)
from PIL import Image
import cv2
import tensorflow as tf
tf.set_random_seed(19)
from cyclegan.model import cyclegan

srcPath='./src/img.jpg'
dstPath=''
crtPath=''
imgLabel=''
imgEdit=''
winLabel=''
winLcd=''
imgStyle='A'
imgEffect='└hue'
imgValue=50
imgFormat='jpg'
img=''

class MainWindow(QDialog):

    def __init__(self):
        super().__init__()
        self.init_UI()
        self.setGeometry(240, 135, 1440, 810)
        self.setWindowFlags(Qt.FramelessWindowHint)
        self.show()

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
        self.shadow1.setXOffset(0)
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
        self.shadow2.setBlurRadius(15)
        self.shadow2.setColor(QtGui.QColor(0, 0, 0, 150))
        self.shadow2.setXOffset(0)
        self.shadow2.setYOffset(0)
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

        global imgLabel
        imgLabel = QLabel(self)
        pixmap = QPixmap("main.png")#interface
        imgLabel.setPixmap(pixmap)
        imgLabel.setGeometry(20, 45, 1050, 636)

        self.myQListWidget = QtWidgets.QListWidget(self)#others
        for index, name, icon in [
            ('No.1', 'a', 'icon.png'),
            ('No.2', 'b', 'icon.png'),
            ('No.3', 'c', 'icon.png')]:
            myQCustomQWidget = QCustomQWidget()
            myQCustomQWidget.setTextUp(index)
            myQCustomQWidget.setTextDown(name)
            myQCustomQWidget.setIcon(icon)
            myQListWidgetItem = QtWidgets.QListWidgetItem(self.myQListWidget)
            myQListWidgetItem.setSizeHint(myQCustomQWidget.sizeHint())
            self.myQListWidget.addItem(myQListWidgetItem)
            self.myQListWidget.setItemWidget(myQListWidgetItem, myQCustomQWidget)
            self.myQListWidget.setGeometry(1108, 45, 310, 636)

        global imgEdit
        imgEdit = QLineEdit(self)
        imgEdit.setText("C:/")
        imgEdit.setGeometry(20,777,546,22)
        imgEdit.setStyleSheet("QLineEdit{background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,"
                            "stop: 0 rgb(180,180,180), stop: 1 rgb(220,220,220));"
                            "border-style: solid;"
                            "border-color: rgb(15,15,15);"
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
                           "border-color: rgb(15,15,15);"
                           "border-width: 1px;"
                           "border-radius: 2px;"
                           "color: rgb(35,35,35);"
                           "font: bold;"
                           "font-size: 10px;"
                           "font-family: Arial;}")

        ####here####
        ## 콤보박스에 넣을 이름, addItem에 넣으면 됨 모델명으로 하는 거 추천
        ## 초기값이 안들어가 있어서 vangogh로 되어있어도 맨처음에 하나 클릭 해야됨
         
        combo2 = QComboBox(self)
        combo2.addItem("vangogh")
        combo2.addItem("ukiyoe")
        combo2.addItem("C")
        combo2.setGeometry(225, 681, 100, 22)
        combo2.activated[str].connect(self.on_format2)
        btn7 = QPushButton("적용", self)
        btn7.setGeometry(325, 681, 35, 20)
        btn7.clicked.connect(self.on_save2)


        btn1 = QPushButton("●", self)
        btn1.setGeometry(1035, 692,35,20)
        btn1.clicked.connect(self.on_scale)
        btn1.setStyleSheet("QPushButton{background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,"
                           "stop: 0 rgb(60,60,60), stop: 1 rgb(45,45,45));"
                           "border-style: solid;"
                           "border-color: rgb(15,15,15);"
                           "border-width: 1px;"
                           "border-radius: 3px;"
                           "color: rgb(180,180,180);"
                           "font: bold;"
                           "font-size: 10px;"
                           "font-family: Arial;}")

        btn2 = QPushButton("More..", self)
        btn2.setGeometry(1185, 702,160,40)
        btn2.clicked.connect(self.on_effect)
        btn2.setStyleSheet("QPushButton{background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,"
                           "stop: 0 rgb(85,85,85), stop: 1 rgb(70,70,70));"
                           "border-style: solid;"
                           "border-color: rgb(55,55,55);"
                           "border-width: 1px;"
                           "border-radius: 4px;"
                           "color: rgb(190,190,190);"
                           "font: bold;"
                           "font-size: 12px;"
                           "font-family: Arial;}")
        self.shadow4 = QGraphicsDropShadowEffect()
        self.shadow4.setBlurRadius(6)
        self.shadow4.setColor(QtGui.QColor(0, 0, 0, 150))
        self.shadow4.setXOffset(1)
        self.shadow4.setYOffset(1)
        btn2.setGraphicsEffect(self.shadow4)

        btn3 = QPushButton("Open", self)
        btn3.setGeometry(587, 777,35,21)
        btn3.clicked.connect(self.on_open)
        btn3.setStyleSheet("QPushButton{background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,"
                           "stop: 0 rgb(35,35,35), stop: 1 rgb(20,20,20));"
                           "border-style: solid;"
                           "border-color: rgb(15,15,15);"
                           "border-width: 1px;"
                           "border-radius: 2px;"
                           "color: rgb(150,150,150);"
                           "font: bold;"
                           "font-size: 10px;"
                           "font-family: Arial;}")

        btn4 = QPushButton("Save", self)
        btn4.setGeometry(1327, 777,35,21)
        btn4.clicked.connect(self.on_save)
        btn4.setStyleSheet("QPushButton{background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,"
                           "stop: 0 rgb(35,35,35), stop: 1 rgb(20,20,20));"
                           "border-style: solid;"
                           "border-color: rgb(15,15,15);"
                           "border-width: 1px;"
                           "border-radius: 2px;"
                           "color: rgb(150,150,150);"
                           "font: bold;"
                           "font-size: 10px;"
                           "font-family: Arial;}")

        btn5 = QPushButton("◀", self)
        btn5.setGeometry(484, 690,60,22)
        btn5.clicked.connect(self.on_bef)
        btn5.setStyleSheet("QPushButton{background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,"
                           "stop: 0 rgb(60,60,60), stop: 1 rgb(45,45,45));"
                           "border-style: solid;"
                           "border-color: rgb(15,15,15);"
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
                           "stop: 0 rgb(60,60,60), stop: 1 rgb(45,45,45));"
                           "border-style: solid;"
                           "border-color: rgb(15,15,15);"
                           "border-width: 1px;"
                           "border-radius: 3px;"
                           "color: rgb(180,180,180);"
                           "font: bold;"
                           "font-size: 10px;"
                           "font-family: Arial;}")

    def on_about(self):
        self.next = AboutWindow()

    def on_exit(self):
        message = QMessageBox.question(self, 'Exit', "Are you sure to quit?", QMessageBox.Yes|QMessageBox.No, QMessageBox.No)
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
        pixmap = QPixmap(dstPath)#others
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
            root.filename = filedialog.askopenfilename(initialdir="C:/", title="Open", filetypes=(("jpeg files", "*.jpg"), ("all files", "*.*")))
            root.withdraw()
            global srcPath
            pixmap = QPixmap(root.filename)
            img=pixmap
            img.save(srcPath)
            smaller_pixmap = pixmap.scaled(1050, 636, Qt.KeepAspectRatio, Qt.FastTransformation)
            global imgLabel
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

    ###here
    ### 내경우는 item이랑 imgstyle이랑 통일함
    def on_format2(self, item):
        global imgStyle
        if item == "vangogh":
            imgStyle = "vangogh"
        elif item == "ukiyoe":
            imgStyle = "ukiyoe"
        elif item == "C":
            imgStyle = "C"

    def on_save2(self):
        global imgStyle
        global dstPath
        global srcPath
        sp=os.path.split(srcPath)
        #sp[0] directory
        #sp[1] filename
        if imgStyle == "vangogh":
            modelname='vangogh2photo'
            tfconfig=tf.ConfigProto(allow_soft_placement=True)
            tfconfig.gpu_options.allow_growth=True
            with tf.Session(config=tfconfig) as sess:
                model=cyclegan(sess,modelname,srcPath)
                dstPath=model.test()
            tf.contrib.keras.backend.clear_session()
            pixmap = QPixmap(dstPath)
            imgLabel.setPixmap(pixmap)
            imgLabel.setGeometry(20, 45, 1050, 636)

        if imgStyle == "ukiyoe":
            modelname='ukiyoe2photo'
            tfconfig=tf.ConfigProto(allow_soft_placement=True)
            tfconfig.gpu_options.allow_growth=True
            with tf.Session(config=tfconfig) as sess:
                model=cyclegan(sess,modelname,srcPath)
                dstPath=model.test()
            ### tensorflow 쓰는 경우 아래처럼 session초기화 해줘야 에러 안남
            tf.contrib.keras.backend.clear_session()
            ### 밑에 주석 참고
            pixmap = QPixmap(dstPath)
            imgLabel.setPixmap(pixmap)
            imgLabel.setGeometry(20, 45, 1050, 636)

        '''
        나는 애초에 모델에서 저장을 해줌,
        img=QPixmap(srcPath)
        img.save(dstPath)
        그대로 두면 원본이 덮어 씌워서
        pixmap = QPixmap(dstPath)
        imgLabel.setPixmap(pixmap)
        imgLabel.setGeometry(20, 45, 1050, 636)
        로 바꿨음
        '''
        if imgStyle == "C":
            dstPath='./dst/C/img.jpg'
            img=QPixmap(srcPath)
            img.save(dstPath)
    ###

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

class QCustomQWidget(QtWidgets.QWidget):

    def __init__(self, parent=None):
        super(QCustomQWidget, self).__init__(parent)
        self.textQVBoxLayout = QtWidgets.QVBoxLayout()
        self.textUpQLabel = QtWidgets.QLabel()
        self.textDownQLabel = QtWidgets.QLabel()
        self.textQVBoxLayout.addWidget(self.textUpQLabel)
        self.textQVBoxLayout.addWidget(self.textDownQLabel)
        self.allQHBoxLayout = QtWidgets.QHBoxLayout()
        self.iconQLabel = QtWidgets.QLabel()
        self.allQHBoxLayout.addWidget(self.iconQLabel, 0)
        self.allQHBoxLayout.addLayout(self.textQVBoxLayout, 1)
        self.setLayout(self.allQHBoxLayout)

    def setTextUp(self, text):
        self.textUpQLabel.setText(text)

    def setTextDown(self, text):
        self.textDownQLabel.setText(text)

    def setIcon(self, imagePath):
        self.iconQLabel.setPixmap(QtGui.QPixmap(imagePath))

class AboutWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setGeometry(571, 205, 778, 670)
        self.setWindowFlags(Qt.FramelessWindowHint)
        label = QLabel(self)
        pixmap = QPixmap("about.jpg") #splash
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
        label.setPixmap(bigger_pixmap)
        label.setGeometry(0, 0, 1920, 1080)
        self.show()

class EffectWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Effect")
        self.setGeometry(1218, 338, 171, 404)
        cv2.namedWindow('img', cv2.WINDOW_NORMAL)
        cv2.setWindowProperty('img', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.moveWindow('img', 531, 307)
        cv2.resizeWindow('img', 667, 435)
        self.init_window()

    def init_window(self):
        list = QListWidget(self)
        label = ('--Image Processing--',
               '└hue',
               '└filter',
               '　└warm',
               '　└cool',
               '　└sepia',
               '　└vintage',
               '└brightness',
               '└contrast',
               '└saturation',
               '└sharp',
               '└blur',
               '　',
               '--Photo Decoration--',
               '└frame',
               '└canvas')
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
        btn = QPushButton("apply", self)
        btn.setGeometry(20, 354, 131, 30)
        btn.clicked.connect(self.on_effectBtn)
        self.show()

    def on_effectList(self, item):
        global imgEffect
        if item.text() == "└hue":
            imgEffect = "hue"
        elif item.text() == "　└warm":
            imgEffect = "warm"
        elif item.text() == "　└cool":
            imgEffect = "cool"
        elif item.text() == "　└sepia":
            imgEffect = "sepia"
        elif item.text() == "　└vintage":
            imgEffect = "vintage"
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
        elif item.text() == "└frame":
            imgEffect = "frame"
        elif item.text() == "└canvas":
            imgEffect = "canvas"
        else:
            imgEffect=''

    def on_effectSld(self,value):
        global imgValue
        imgValue = value
        global winLcd
        winLcd.display(imgValue)
        global dstPath
        global img
        img = cv2.imread(dstPath, cv2.IMREAD_COLOR)
        global imgEffect
        if imgEffect=="blur":
            img = cv2.blur(img, (imgValue, imgValue), anchor=(-1, -1), borderType=cv2.BORDER_DEFAULT)
        cv2.imshow('img', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def on_effectBtn(self):
        global dstPath
        global img
        cv2.imwrite(dstPath,img)
        global imgLabel
        pixmap = QPixmap(dstPath)
        smaller_pixmap = pixmap.scaled(1050, 636, Qt.KeepAspectRatio, Qt.FastTransformation)
        imgLabel.setPixmap(smaller_pixmap)

class StartWindow(QMainWindow):

    def __init__(self, *args, **kwargs):
        super(StartWindow, self).__init__(*args, **kwargs)
        self.setGeometry(560, 240, 900, 600)
        self.setWindowFlags(Qt.FramelessWindowHint)
        label = QLabel(self)
        pixmap = QPixmap("start.jpg") #splash
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
        self.timer.setInterval(1000)
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
            time.sleep(3)
            self.close()
            window = MainWindow()
            window.exec_()

app = QApplication([])
window = StartWindow()
app.exec_()