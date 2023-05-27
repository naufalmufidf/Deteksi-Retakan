# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'mainhObGLz.ui'
##
## Created by: Qt User Interface Compiler version 5.15.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.setEnabled(True)
        MainWindow.resize(1920, 888)
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.gridLayout = QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName(u"gridLayout")
        self.shadow = QFrame(self.centralwidget)
        self.shadow.setObjectName(u"shadow")
        self.shadow.setStyleSheet(u"QFrame{\n"
"	background-color: rgb(45, 48, 55);\n"
"	color: rgb(220, 220, 220);\n"
"	border-radius: 10px;\n"
"}")
        self.shadow.setFrameShape(QFrame.StyledPanel)
        self.shadow.setFrameShadow(QFrame.Raised)
        self.widget_2 = QWidget(self.shadow)
        self.widget_2.setObjectName(u"widget_2")
        self.widget_2.setGeometry(QRect(0, 0, 1901, 141))
        self.widget_2.setStyleSheet(u"border-top-left-radius : 10px;\n"
"border-top-right-radius : 10px;\n"
"background-color: #E7B18E;")
        self.label_3 = QLabel(self.widget_2)
        self.label_3.setObjectName(u"label_3")
        self.label_3.setGeometry(QRect(470, 30, 971, 101))
        font = QFont()
        font.setFamily(u"Casta Thin")
        font.setPointSize(68)
        self.label_3.setFont(font)
        self.label_3.setLayoutDirection(Qt.LeftToRight)
        self.label_3.setStyleSheet(u"color: E7B18E;")
        self.label_3.setAlignment(Qt.AlignCenter)
        self.btn_close = QPushButton(self.widget_2)
        self.btn_close.setObjectName(u"btn_close")
        self.btn_close.setGeometry(QRect(1800, 20, 18, 18))
        self.btn_close.setCursor(QCursor(Qt.PointingHandCursor))
        self.btn_close.setStyleSheet(u"background: #FF605C;\n"
"border-radius: 9%;")
        self.btn_minimize = QPushButton(self.widget_2)
        self.btn_minimize.setObjectName(u"btn_minimize")
        self.btn_minimize.setGeometry(QRect(1830, 20, 18, 18))
        self.btn_minimize.setCursor(QCursor(Qt.PointingHandCursor))
        self.btn_minimize.setStyleSheet(u"background: #FFBD44;\n"
"border-radius: 9%;")
        self.btn_gtw = QPushButton(self.widget_2)
        self.btn_gtw.setObjectName(u"btn_gtw")
        self.btn_gtw.setGeometry(QRect(1860, 20, 18, 18))
        self.btn_gtw.setCursor(QCursor(Qt.PointingHandCursor))
        self.btn_gtw.setStyleSheet(u"background: #00CA4E;\n"
"border-radius: 9%;")
        self.label_back = QLabel(self.widget_2)
        self.label_back.setObjectName(u"label_back")
        self.label_back.setGeometry(QRect(10, 30, 151, 71))
        self.label_back.setCursor(QCursor(Qt.PointingHandCursor))
        self.label_back.setPixmap(QPixmap(u"Ihome.png"))
        self.label = QLabel(self.shadow)
        self.label.setObjectName(u"label")
        self.label.setGeometry(QRect(30, 310, 300, 300))
        self.label.setStyleSheet(u"border: 2px solid;\n"
"border-color: rgb(231, 177, 142);")
        self.label.setFrameShape(QFrame.Box)
        self.label_2 = QLabel(self.shadow)
        self.label_2.setObjectName(u"label_2")
        self.label_2.setGeometry(QRect(340, 310, 300, 300))
        self.label_2.setStyleSheet(u"border: 2px solid;\n"
"border-color: rgb(231, 177, 142);")
        self.label_2.setFrameShape(QFrame.Box)
        self.label_4 = QLabel(self.shadow)
        self.label_4.setObjectName(u"label_4")
        self.label_4.setGeometry(QRect(130, 240, 101, 61))
        font1 = QFont()
        font1.setFamily(u"Gramatika-Medium")
        font1.setPointSize(24)
        self.label_4.setFont(font1)
        self.label_4.setStyleSheet(u"color: #E7B18E;")
        self.label_4.setAlignment(Qt.AlignCenter)
        self.label_5 = QLabel(self.shadow)
        self.label_5.setObjectName(u"label_5")
        self.label_5.setGeometry(QRect(1660, 240, 121, 71))
        self.label_5.setFont(font1)
        self.label_5.setStyleSheet(u"color: #E7B18E;")
        self.label_5.setAlignment(Qt.AlignCenter)
        self.btn_loadImage = QPushButton(self.shadow)
        self.btn_loadImage.setObjectName(u"btn_loadImage")
        self.btn_loadImage.setGeometry(QRect(90, 640, 181, 51))
        font2 = QFont()
        font2.setFamily(u"Gramatika-Medium")
        font2.setPointSize(10)
        self.btn_loadImage.setFont(font2)
        self.btn_loadImage.setCursor(QCursor(Qt.PointingHandCursor))
        self.btn_loadImage.setStyleSheet(u"background-color: #E7B18E;\n"
"border: 1px solid #E7B18E; \n"
"border-radius: 20px;")
        self.btn_saveImage = QPushButton(self.shadow)
        self.btn_saveImage.setObjectName(u"btn_saveImage")
        self.btn_saveImage.setGeometry(QRect(1640, 640, 181, 51))
        self.btn_saveImage.setFont(font2)
        self.btn_saveImage.setCursor(QCursor(Qt.PointingHandCursor))
        self.btn_saveImage.setStyleSheet(u"background-color: #E7B18E;\n"
"border: 1px solid #E7B18E; \n"
"border-radius: 20px;")
        self.btn_process = QPushButton(self.shadow)
        self.btn_process.setObjectName(u"btn_process")
        self.btn_process.setGeometry(QRect(400, 640, 181, 51))
        self.btn_process.setFont(font2)
        self.btn_process.setCursor(QCursor(Qt.PointingHandCursor))
        self.btn_process.setStyleSheet(u"background-color: #E7B18E;\n"
"border: 1px solid #E7B18E; \n"
"border-radius: 20px;")
        self.label_credits = QLabel(self.shadow)
        self.label_credits.setObjectName(u"label_credits")
        self.label_credits.setGeometry(QRect(1130, 830, 761, 31))
        self.label_credits.setFont(font2)
        self.label_credits.setStyleSheet(u"color: rgb(110, 117, 134);")
        self.label_credits.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)
        self.label_7 = QLabel(self.shadow)
        self.label_7.setObjectName(u"label_7")
        self.label_7.setGeometry(QRect(650, 310, 300, 300))
        self.label_7.setStyleSheet(u"border: 2px solid;\n"
"border-color: rgb(231, 177, 142);")
        self.label_7.setFrameShape(QFrame.Box)
        self.label_8 = QLabel(self.shadow)
        self.label_8.setObjectName(u"label_8")
        self.label_8.setGeometry(QRect(960, 310, 300, 300))
        self.label_8.setStyleSheet(u"border: 2px solid;\n"
"border-color: rgb(231, 177, 142);")
        self.label_8.setFrameShape(QFrame.Box)
        self.label_9 = QLabel(self.shadow)
        self.label_9.setObjectName(u"label_9")
        self.label_9.setGeometry(QRect(1270, 310, 300, 300))
        self.label_9.setStyleSheet(u"border: 2px solid;\n"
"border-color: rgb(231, 177, 142);")
        self.label_9.setFrameShape(QFrame.Box)
        self.label_10 = QLabel(self.shadow)
        self.label_10.setObjectName(u"label_10")
        self.label_10.setGeometry(QRect(1580, 310, 300, 300))
        self.label_10.setStyleSheet(u"border: 2px solid;\n"
"border-color: rgb(231, 177, 142);")
        self.label_10.setFrameShape(QFrame.Box)
        self.label_6 = QLabel(self.shadow)
        self.label_6.setObjectName(u"label_6")
        self.label_6.setGeometry(QRect(390, 240, 201, 61))
        self.label_6.setFont(font1)
        self.label_6.setStyleSheet(u"color: #E7B18E;")
        self.label_6.setAlignment(Qt.AlignCenter)
        self.label_11 = QLabel(self.shadow)
        self.label_11.setObjectName(u"label_11")
        self.label_11.setGeometry(QRect(690, 240, 221, 61))
        self.label_11.setFont(font1)
        self.label_11.setStyleSheet(u"color: #E7B18E;")
        self.label_11.setAlignment(Qt.AlignCenter)
        self.label_12 = QLabel(self.shadow)
        self.label_12.setObjectName(u"label_12")
        self.label_12.setGeometry(QRect(1000, 240, 221, 61))
        self.label_12.setFont(font1)
        self.label_12.setStyleSheet(u"color: #E7B18E;")
        self.label_12.setAlignment(Qt.AlignCenter)
        self.label_13 = QLabel(self.shadow)
        self.label_13.setObjectName(u"label_13")
        self.label_13.setGeometry(QRect(1310, 240, 221, 61))
        self.label_13.setFont(font1)
        self.label_13.setStyleSheet(u"color: #E7B18E;")
        self.label_13.setAlignment(Qt.AlignCenter)
        self.label_14 = QLabel(self.shadow)
        self.label_14.setObjectName(u"label_14")
        self.label_14.setGeometry(QRect(1590, 580, 151, 16))
        font3 = QFont()
        font3.setFamily(u"Gramatika-Medium")
        font3.setPointSize(12)
        self.label_14.setFont(font3)
        self.label_14.setStyleSheet(u"background-color:rgb(45, 48, 55);\n"
"color: #E7B18E;")

        self.gridLayout.addWidget(self.shadow, 0, 0, 1, 1)

        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"MainWindow", None))
        self.label_3.setText(QCoreApplication.translate("MainWindow", u"Deteksi Retakan", None))
        self.btn_close.setText("")
        self.btn_minimize.setText("")
        self.btn_gtw.setText("")
        self.label_back.setText("")
        self.label.setText("")
        self.label_2.setText("")
        self.label_4.setText(QCoreApplication.translate("MainWindow", u"Input", None))
        self.label_5.setText(QCoreApplication.translate("MainWindow", u"Output", None))
        self.btn_loadImage.setText(QCoreApplication.translate("MainWindow", u"Load Image", None))
        self.btn_saveImage.setText(QCoreApplication.translate("MainWindow", u"Save Image", None))
        self.btn_process.setText(QCoreApplication.translate("MainWindow", u"Process", None))
        self.label_credits.setText(QCoreApplication.translate("MainWindow", u"Final project by : C2", None))
        self.label_7.setText("")
        self.label_8.setText("")
        self.label_9.setText("")
        self.label_10.setText("")
        self.label_6.setText(QCoreApplication.translate("MainWindow", u"Gray + mean", None))
        self.label_11.setText(QCoreApplication.translate("MainWindow", u"Bilateral filter", None))
        self.label_12.setText(QCoreApplication.translate("MainWindow", u"Canny", None))
        self.label_13.setText(QCoreApplication.translate("MainWindow", u"Closing", None))
        self.label_14.setText("")
    # retranslateUi

