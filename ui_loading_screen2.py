# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'loading_screen2uArXZr.ui'
##
## Created by: Qt User Interface Compiler version 5.15.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *


class Ui_Loading2(object):
    def setupUi(self, Loading2):
        if not Loading2.objectName():
            Loading2.setObjectName(u"Loading2")
        Loading2.resize(679, 400)
        self.centralwidget = QWidget(Loading2)
        self.centralwidget.setObjectName(u"centralwidget")
        self.verticalLayout = QVBoxLayout(self.centralwidget)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.shadow = QFrame(self.centralwidget)
        self.shadow.setObjectName(u"shadow")
        self.shadow.setStyleSheet(u"QFrame{\n"
"	background-color: rgb(45, 48, 55);\n"
"	color: rgb(220, 220, 220);\n"
"	border-radius: 10px;\n"
"}")
        self.shadow.setFrameShape(QFrame.StyledPanel)
        self.shadow.setFrameShadow(QFrame.Raised)
        self.label_title = QLabel(self.shadow)
        self.label_title.setObjectName(u"label_title")
        self.label_title.setGeometry(QRect(0, 10, 661, 181))
        font = QFont()
        font.setFamily(u"Casta")
        font.setPointSize(50)
        self.label_title.setFont(font)
        self.label_title.setStyleSheet(u"color: #E7B18E;")
        self.label_title.setAlignment(Qt.AlignBottom|Qt.AlignHCenter)
        self.label_desc = QLabel(self.shadow)
        self.label_desc.setObjectName(u"label_desc")
        self.label_desc.setGeometry(QRect(0, 190, 661, 31))
        font1 = QFont()
        font1.setFamily(u"Gramatika-Medium")
        font1.setPointSize(14)
        self.label_desc.setFont(font1)
        self.label_desc.setStyleSheet(u"color: rgb(110, 117, 134);")
        self.label_desc.setAlignment(Qt.AlignCenter)
        self.progressBar = QProgressBar(self.shadow)
        self.progressBar.setObjectName(u"progressBar")
        self.progressBar.setGeometry(QRect(30, 280, 601, 23))
        self.progressBar.setStyleSheet(u"QProgressBar{\n"
"	background-color: rgb(110, 117, 134);\n"
"	color: rgb(200, 200, 200);\n"
"	border-style: none;\n"
"	border-radius:10px;\n"
"	text-align: center;\n"
"}\n"
"\n"
"QProgressBar::chunk{\n"
"	background-color: qlineargradient(spread:pad, x1:0, y1:0.46, x2:1, y2:0.46, stop:0 rgba(231, 177, 142, 255), stop:1 rgba(138, 113, 99, 255));\n"
"	border-radius: 10px;\n"
"}")
        self.progressBar.setValue(24)
        self.label_loading = QLabel(self.shadow)
        self.label_loading.setObjectName(u"label_loading")
        self.label_loading.setGeometry(QRect(0, 310, 661, 31))
        font2 = QFont()
        font2.setFamily(u"Gramatika-Medium")
        font2.setPointSize(12)
        self.label_loading.setFont(font2)
        self.label_loading.setStyleSheet(u"color: rgb(110, 117, 134);")
        self.label_loading.setAlignment(Qt.AlignCenter)
        self.label_credits = QLabel(self.shadow)
        self.label_credits.setObjectName(u"label_credits")
        self.label_credits.setGeometry(QRect(20, 340, 621, 31))
        font3 = QFont()
        font3.setFamily(u"Gramatika-Medium")
        font3.setPointSize(10)
        self.label_credits.setFont(font3)
        self.label_credits.setStyleSheet(u"color: rgb(110, 117, 134);")
        self.label_credits.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.verticalLayout.addWidget(self.shadow)

        Loading2.setCentralWidget(self.centralwidget)

        self.retranslateUi(Loading2)

        QMetaObject.connectSlotsByName(Loading2)
    # setupUi

    def retranslateUi(self, Loading2):
        Loading2.setWindowTitle(QCoreApplication.translate("Loading2", u"MainWindow", None))
        self.label_title.setText(QCoreApplication.translate("Loading2", u"<html><head/><body><p>Deteksi Retakan</p></body></html>", None))
        self.label_desc.setText(QCoreApplication.translate("Loading2", u"Mendeteksi retakan pada dinding bangunan", None))
        self.label_loading.setText(QCoreApplication.translate("Loading2", u"loading...", None))
        self.label_credits.setText(QCoreApplication.translate("Loading2", u"Final project by : C2", None))
    # retranslateUi

