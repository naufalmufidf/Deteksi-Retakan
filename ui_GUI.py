# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'GUIpQqPcn.ui'
##
## Created by: Qt User Interface Compiler version 5.15.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *


class Ui_Preprocessing(object):
    def setupUi(self, Preprocessing):
        if not Preprocessing.objectName():
            Preprocessing.setObjectName(u"Preprocessing")
        Preprocessing.resize(1230, 869)
        self.actionGrayscale = QAction(Preprocessing)
        self.actionGrayscale.setObjectName(u"actionGrayscale")
        self.actionBiner = QAction(Preprocessing)
        self.actionBiner.setObjectName(u"actionBiner")
        self.actionNegative = QAction(Preprocessing)
        self.actionNegative.setObjectName(u"actionNegative")
        self.actionCoba = QAction(Preprocessing)
        self.actionCoba.setObjectName(u"actionCoba")
        self.actionHistogram_Grayscale = QAction(Preprocessing)
        self.actionHistogram_Grayscale.setObjectName(u"actionHistogram_Grayscale")
        self.actionHistogram_RGB = QAction(Preprocessing)
        self.actionHistogram_RGB.setObjectName(u"actionHistogram_RGB")
        self.actionHistogram_Equalization = QAction(Preprocessing)
        self.actionHistogram_Equalization.setObjectName(u"actionHistogram_Equalization")
        self.actionSimple_Contrast = QAction(Preprocessing)
        self.actionSimple_Contrast.setObjectName(u"actionSimple_Contrast")
        self.actionContrast_Stretching = QAction(Preprocessing)
        self.actionContrast_Stretching.setObjectName(u"actionContrast_Stretching")
        self.actionGamma_Correction = QAction(Preprocessing)
        self.actionGamma_Correction.setObjectName(u"actionGamma_Correction")
        self.actionKonvolusi = QAction(Preprocessing)
        self.actionKonvolusi.setObjectName(u"actionKonvolusi")
        self.actionMean = QAction(Preprocessing)
        self.actionMean.setObjectName(u"actionMean")
        self.actionGaussian = QAction(Preprocessing)
        self.actionGaussian.setObjectName(u"actionGaussian")
        self.actionMean_Filter = QAction(Preprocessing)
        self.actionMean_Filter.setObjectName(u"actionMean_Filter")
        self.actionGaussian_Filter = QAction(Preprocessing)
        self.actionGaussian_Filter.setObjectName(u"actionGaussian_Filter")
        self.actionMedian_Filter = QAction(Preprocessing)
        self.actionMedian_Filter.setObjectName(u"actionMedian_Filter")
        self.actionDFT_Smoothing = QAction(Preprocessing)
        self.actionDFT_Smoothing.setObjectName(u"actionDFT_Smoothing")
        self.actionBilateral_Filter = QAction(Preprocessing)
        self.actionBilateral_Filter.setObjectName(u"actionBilateral_Filter")
        self.actionTranslasi = QAction(Preprocessing)
        self.actionTranslasi.setObjectName(u"actionTranslasi")
        self.actionTranspose = QAction(Preprocessing)
        self.actionTranspose.setObjectName(u"actionTranspose")
        self.actionCrop_Image = QAction(Preprocessing)
        self.actionCrop_Image.setObjectName(u"actionCrop_Image")
        self.action_min90d = QAction(Preprocessing)
        self.action_min90d.setObjectName(u"action_min90d")
        self.action_min45d = QAction(Preprocessing)
        self.action_min45d.setObjectName(u"action_min45d")
        self.action45d = QAction(Preprocessing)
        self.action45d.setObjectName(u"action45d")
        self.action90d = QAction(Preprocessing)
        self.action90d.setObjectName(u"action90d")
        self.action180d = QAction(Preprocessing)
        self.action180d.setObjectName(u"action180d")
        self.action2X = QAction(Preprocessing)
        self.action2X.setObjectName(u"action2X")
        self.action3X = QAction(Preprocessing)
        self.action3X.setObjectName(u"action3X")
        self.action4X = QAction(Preprocessing)
        self.action4X.setObjectName(u"action4X")
        self.actionQuarter = QAction(Preprocessing)
        self.actionQuarter.setObjectName(u"actionQuarter")
        self.actionHalf = QAction(Preprocessing)
        self.actionHalf.setObjectName(u"actionHalf")
        self.actionThree_Quarter = QAction(Preprocessing)
        self.actionThree_Quarter.setObjectName(u"actionThree_Quarter")
        self.action360p = QAction(Preprocessing)
        self.action360p.setObjectName(u"action360p")
        self.action480p = QAction(Preprocessing)
        self.action480p.setObjectName(u"action480p")
        self.action720p = QAction(Preprocessing)
        self.action720p.setObjectName(u"action720p")
        self.action1080p = QAction(Preprocessing)
        self.action1080p.setObjectName(u"action1080p")
        self.actionplusmin = QAction(Preprocessing)
        self.actionplusmin.setObjectName(u"actionplusmin")
        self.actionkalibagi = QAction(Preprocessing)
        self.actionkalibagi.setObjectName(u"actionkalibagi")
        self.actionAND = QAction(Preprocessing)
        self.actionAND.setObjectName(u"actionAND")
        self.actionOR = QAction(Preprocessing)
        self.actionOR.setObjectName(u"actionOR")
        self.actionXOR = QAction(Preprocessing)
        self.actionXOR.setObjectName(u"actionXOR")
        self.actionPre_processing = QAction(Preprocessing)
        self.actionPre_processing.setObjectName(u"actionPre_processing")
        self.actionOperasi_Pencerahan = QAction(Preprocessing)
        self.actionOperasi_Pencerahan.setObjectName(u"actionOperasi_Pencerahan")
        self.centralwidget = QWidget(Preprocessing)
        self.centralwidget.setObjectName(u"centralwidget")
        self.widget = QWidget(self.centralwidget)
        self.widget.setObjectName(u"widget")
        self.widget.setGeometry(QRect(0, 0, 1231, 851))
        self.widget.setStyleSheet(u"background-color: rgb(45, 48, 55);")
        self.label_2 = QLabel(self.widget)
        self.label_2.setObjectName(u"label_2")
        self.label_2.setGeometry(QRect(760, 260, 381, 381))
        self.label_2.setStyleSheet(u"border: 2px solid;\n"
"border-color: rgb(231, 177, 142);\n"
"border-radius: 5px;")
        self.label_2.setFrameShape(QFrame.Box)
        self.label = QLabel(self.widget)
        self.label.setObjectName(u"label")
        self.label.setGeometry(QRect(70, 260, 381, 381))
        self.label.setStyleSheet(u"border: 2px solid;\n"
"border-color: rgb(231, 177, 142);\n"
"border-radius: 5px;")
        self.label.setFrameShape(QFrame.Box)
        self.btn_loadImage = QPushButton(self.widget)
        self.btn_loadImage.setObjectName(u"btn_loadImage")
        self.btn_loadImage.setGeometry(QRect(170, 690, 181, 51))
        font = QFont()
        font.setFamily(u"Gramatika-Medium")
        font.setPointSize(10)
        self.btn_loadImage.setFont(font)
        self.btn_loadImage.setCursor(QCursor(Qt.PointingHandCursor))
        self.btn_loadImage.setStyleSheet(u"background-color: #E7B18E;\n"
"border: 1px solid #E7B18E; \n"
"border-radius: 20px;")
        self.btn_saveImage = QPushButton(self.widget)
        self.btn_saveImage.setObjectName(u"btn_saveImage")
        self.btn_saveImage.setGeometry(QRect(860, 690, 181, 51))
        self.btn_saveImage.setFont(font)
        self.btn_saveImage.setCursor(QCursor(Qt.PointingHandCursor))
        self.btn_saveImage.setStyleSheet(u"background-color: #E7B18E;\n"
"border: 1px solid #E7B18E; \n"
"border-radius: 20px;")
        self.widget_2 = QWidget(self.widget)
        self.widget_2.setObjectName(u"widget_2")
        self.widget_2.setGeometry(QRect(0, 0, 1231, 111))
        self.widget_2.setStyleSheet(u"background-color: #E7B18E;")
        self.label_3 = QLabel(self.widget_2)
        self.label_3.setObjectName(u"label_3")
        self.label_3.setGeometry(QRect(310, 10, 611, 101))
        font1 = QFont()
        font1.setFamily(u"Casta Thin")
        font1.setPointSize(72)
        self.label_3.setFont(font1)
        self.label_3.setLayoutDirection(Qt.LeftToRight)
        self.label_3.setStyleSheet(u"color: E7B18E;")
        self.label_3.setAlignment(Qt.AlignCenter)
        self.btn_close = QPushButton(self.widget_2)
        self.btn_close.setObjectName(u"btn_close")
        self.btn_close.setGeometry(QRect(1130, 20, 18, 18))
        self.btn_close.setCursor(QCursor(Qt.PointingHandCursor))
        self.btn_close.setStyleSheet(u"background: #FF605C;\n"
"border-radius: 9%;")
        self.btn_minimize = QPushButton(self.widget_2)
        self.btn_minimize.setObjectName(u"btn_minimize")
        self.btn_minimize.setGeometry(QRect(1160, 20, 18, 18))
        self.btn_minimize.setCursor(QCursor(Qt.PointingHandCursor))
        self.btn_minimize.setStyleSheet(u"background: #FFBD44;\n"
"border-radius: 9%;")
        self.btn_gtw = QPushButton(self.widget_2)
        self.btn_gtw.setObjectName(u"btn_gtw")
        self.btn_gtw.setGeometry(QRect(1190, 20, 18, 18))
        self.btn_gtw.setCursor(QCursor(Qt.PointingHandCursor))
        self.btn_gtw.setStyleSheet(u"background: #00CA4E;\n"
"border-radius: 9%;")
        self.label_back = QLabel(self.widget_2)
        self.label_back.setObjectName(u"label_back")
        self.label_back.setGeometry(QRect(0, 20, 151, 71))
        self.label_back.setCursor(QCursor(Qt.PointingHandCursor))
        self.label_back.setPixmap(QPixmap(u"Ihome.png"))
        self.label_4 = QLabel(self.widget)
        self.label_4.setObjectName(u"label_4")
        self.label_4.setGeometry(QRect(210, 180, 101, 61))
        font2 = QFont()
        font2.setFamily(u"Gramatika-Medium")
        font2.setPointSize(24)
        self.label_4.setFont(font2)
        self.label_4.setStyleSheet(u"color: #E7B18E;")
        self.label_4.setAlignment(Qt.AlignCenter)
        self.label_5 = QLabel(self.widget)
        self.label_5.setObjectName(u"label_5")
        self.label_5.setGeometry(QRect(890, 170, 121, 71))
        self.label_5.setFont(font2)
        self.label_5.setStyleSheet(u"color: #E7B18E;")
        self.label_5.setAlignment(Qt.AlignCenter)
        self.label_6 = QLabel(self.widget)
        self.label_6.setObjectName(u"label_6")
        self.label_6.setGeometry(QRect(480, 690, 31, 16))
        font3 = QFont()
        font3.setPointSize(10)
        self.label_6.setFont(font3)
        self.label_6.setStyleSheet(u"color: #E7B18E;")
        self.slider = QSlider(self.widget)
        self.slider.setObjectName(u"slider")
        self.slider.setGeometry(QRect(480, 710, 251, 21))
        self.slider.setCursor(QCursor(Qt.PointingHandCursor))
        self.slider.setStyleSheet(u"QSlider::groove:horizontal {\n"
"border: 1px solid #bbb;\n"
"background: white;\n"
"height: 10px;\n"
"border-radius: 4px;\n"
"}\n"
"\n"
"QSlider::sub-page:horizontal {\n"
"background: qlineargradient(x1: 0, y1: 0,    x2: 0, y2: 1,\n"
"    stop: 0 #E7B18E, stop: 1 #D0A184);\n"
"background: qlineargradient(x1: 0, y1: 0.2, x2: 1, y2: 1,\n"
"    stop: 0 #D0A184, stop: 1 #8A7163);\n"
"border: 1px solid #777;\n"
"height: 10px;\n"
"border-radius: 4px;\n"
"}\n"
"\n"
"QSlider::add-page:horizontal {\n"
"background: #fff;\n"
"border: 1px solid #777;\n"
"height: 10px;\n"
"border-radius: 4px;\n"
"}\n"
"\n"
"QSlider::handle:horizontal {\n"
"background: qlineargradient(x1:0, y1:0, x2:1, y2:1,\n"
"    stop:0 #eee, stop:1 #ccc);\n"
"border: 1px solid #777;\n"
"width: 13px;\n"
"margin-top: -2px;\n"
"margin-bottom: -2px;\n"
"border-radius: 4px;\n"
"}\n"
"\n"
"QSlider::handle:horizontal:hover {\n"
"background: qlineargradient(x1:0, y1:0, x2:1, y2:1,\n"
"    stop:0 #fff, stop:1 #ddd);\n"
"border: 1px solid #444;\n"
"border-radius:"
                        " 4px;\n"
"}\n"
"\n"
"QSlider::sub-page:horizontal:disabled {\n"
"background: #bbb;\n"
"border-color: #999;\n"
"}\n"
"\n"
"QSlider::add-page:horizontal:disabled {\n"
"background: #eee;\n"
"border-color: #999;\n"
"}\n"
"\n"
"QSlider::handle:horizontal:disabled {\n"
"background: #eee;\n"
"border: 1px solid #aaa;\n"
"border-radius: 4px;\n"
"}")
        self.slider.setOrientation(Qt.Horizontal)
        self.slider.setTickInterval(1)
        self.btn_process = QPushButton(self.widget)
        self.btn_process.setObjectName(u"btn_process")
        self.btn_process.setGeometry(QRect(510, 430, 181, 51))
        self.btn_process.setFont(font)
        self.btn_process.setCursor(QCursor(Qt.PointingHandCursor))
        self.btn_process.setStyleSheet(u"background-color: #E7B18E;\n"
"border: 1px solid #E7B18E; \n"
"border-radius: 20px;")
        self.label_credits = QLabel(self.widget)
        self.label_credits.setObjectName(u"label_credits")
        self.label_credits.setGeometry(QRect(460, 810, 761, 31))
        self.label_credits.setFont(font)
        self.label_credits.setStyleSheet(u"color: rgb(110, 117, 134);")
        self.label_credits.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)
        Preprocessing.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(Preprocessing)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 1230, 21))
        self.menuColor = QMenu(self.menubar)
        self.menuColor.setObjectName(u"menuColor")
        self.menucoba = QMenu(self.menubar)
        self.menucoba.setObjectName(u"menucoba")
        self.menuNoise_Reduction = QMenu(self.menubar)
        self.menuNoise_Reduction.setObjectName(u"menuNoise_Reduction")
        self.menuHistogram = QMenu(self.menubar)
        self.menuHistogram.setObjectName(u"menuHistogram")
        self.menuContrast_n_Brightness = QMenu(self.menubar)
        self.menuContrast_n_Brightness.setObjectName(u"menuContrast_n_Brightness")
        self.menuOperasi_Geometri = QMenu(self.menubar)
        self.menuOperasi_Geometri.setObjectName(u"menuOperasi_Geometri")
        self.menuRotasi = QMenu(self.menuOperasi_Geometri)
        self.menuRotasi.setObjectName(u"menuRotasi")
        self.menuResize = QMenu(self.menuOperasi_Geometri)
        self.menuResize.setObjectName(u"menuResize")
        self.menuZoom_In = QMenu(self.menuResize)
        self.menuZoom_In.setObjectName(u"menuZoom_In")
        self.menuZoom_Out = QMenu(self.menuResize)
        self.menuZoom_Out.setObjectName(u"menuZoom_Out")
        self.menuSkewed = QMenu(self.menuResize)
        self.menuSkewed.setObjectName(u"menuSkewed")
        self.menuOperasi_Aritmatika = QMenu(self.menubar)
        self.menuOperasi_Aritmatika.setObjectName(u"menuOperasi_Aritmatika")
        self.menuOperasi_Boolean = QMenu(self.menubar)
        self.menuOperasi_Boolean.setObjectName(u"menuOperasi_Boolean")
        Preprocessing.setMenuBar(self.menubar)

        self.menubar.addAction(self.menuColor.menuAction())
        self.menubar.addAction(self.menuContrast_n_Brightness.menuAction())
        self.menubar.addAction(self.menuNoise_Reduction.menuAction())
        self.menubar.addAction(self.menuHistogram.menuAction())
        self.menubar.addAction(self.menuOperasi_Geometri.menuAction())
        self.menubar.addAction(self.menuOperasi_Aritmatika.menuAction())
        self.menubar.addAction(self.menuOperasi_Boolean.menuAction())
        self.menubar.addAction(self.menucoba.menuAction())
        self.menuColor.addAction(self.actionGrayscale)
        self.menuColor.addAction(self.actionBiner)
        self.menuColor.addAction(self.actionNegative)
        self.menucoba.addAction(self.actionCoba)
        self.menuNoise_Reduction.addAction(self.actionKonvolusi)
        self.menuNoise_Reduction.addAction(self.actionMean_Filter)
        self.menuNoise_Reduction.addAction(self.actionGaussian_Filter)
        self.menuNoise_Reduction.addAction(self.actionMedian_Filter)
        self.menuNoise_Reduction.addAction(self.actionDFT_Smoothing)
        self.menuNoise_Reduction.addSeparator()
        self.menuNoise_Reduction.addAction(self.actionBilateral_Filter)
        self.menuHistogram.addAction(self.actionHistogram_Grayscale)
        self.menuHistogram.addAction(self.actionHistogram_RGB)
        self.menuHistogram.addAction(self.actionHistogram_Equalization)
        self.menuContrast_n_Brightness.addAction(self.actionOperasi_Pencerahan)
        self.menuContrast_n_Brightness.addAction(self.actionSimple_Contrast)
        self.menuContrast_n_Brightness.addAction(self.actionContrast_Stretching)
        self.menuContrast_n_Brightness.addSeparator()
        self.menuContrast_n_Brightness.addAction(self.actionGamma_Correction)
        self.menuOperasi_Geometri.addAction(self.actionTranslasi)
        self.menuOperasi_Geometri.addAction(self.menuRotasi.menuAction())
        self.menuOperasi_Geometri.addAction(self.actionTranspose)
        self.menuOperasi_Geometri.addAction(self.menuResize.menuAction())
        self.menuOperasi_Geometri.addAction(self.actionCrop_Image)
        self.menuRotasi.addAction(self.action_min90d)
        self.menuRotasi.addAction(self.action_min45d)
        self.menuRotasi.addAction(self.action45d)
        self.menuRotasi.addAction(self.action90d)
        self.menuRotasi.addAction(self.action180d)
        self.menuResize.addAction(self.menuZoom_In.menuAction())
        self.menuResize.addAction(self.menuZoom_Out.menuAction())
        self.menuResize.addAction(self.menuSkewed.menuAction())
        self.menuZoom_In.addAction(self.action2X)
        self.menuZoom_In.addAction(self.action3X)
        self.menuZoom_In.addAction(self.action4X)
        self.menuZoom_Out.addAction(self.actionQuarter)
        self.menuZoom_Out.addAction(self.actionHalf)
        self.menuZoom_Out.addAction(self.actionThree_Quarter)
        self.menuSkewed.addAction(self.action360p)
        self.menuSkewed.addAction(self.action480p)
        self.menuSkewed.addAction(self.action720p)
        self.menuSkewed.addAction(self.action1080p)
        self.menuOperasi_Aritmatika.addAction(self.actionplusmin)
        self.menuOperasi_Aritmatika.addAction(self.actionkalibagi)
        self.menuOperasi_Boolean.addAction(self.actionAND)
        self.menuOperasi_Boolean.addAction(self.actionOR)
        self.menuOperasi_Boolean.addAction(self.actionXOR)

        self.retranslateUi(Preprocessing)

        QMetaObject.connectSlotsByName(Preprocessing)
    # setupUi

    def retranslateUi(self, Preprocessing):
        Preprocessing.setWindowTitle(QCoreApplication.translate("Preprocessing", u"MainWindow", None))
        self.actionGrayscale.setText(QCoreApplication.translate("Preprocessing", u"Grayscale", None))
        self.actionBiner.setText(QCoreApplication.translate("Preprocessing", u"Biner", None))
        self.actionNegative.setText(QCoreApplication.translate("Preprocessing", u"Negative", None))
        self.actionCoba.setText(QCoreApplication.translate("Preprocessing", u"Coba", None))
        self.actionHistogram_Grayscale.setText(QCoreApplication.translate("Preprocessing", u"Histogram Grayscale", None))
        self.actionHistogram_RGB.setText(QCoreApplication.translate("Preprocessing", u"Histogram RGB", None))
        self.actionHistogram_Equalization.setText(QCoreApplication.translate("Preprocessing", u"Histogram Equalization", None))
        self.actionSimple_Contrast.setText(QCoreApplication.translate("Preprocessing", u"Simple Contrast", None))
        self.actionContrast_Stretching.setText(QCoreApplication.translate("Preprocessing", u"Contrast Stretching", None))
        self.actionGamma_Correction.setText(QCoreApplication.translate("Preprocessing", u"Gamma Correction", None))
        self.actionKonvolusi.setText(QCoreApplication.translate("Preprocessing", u"Konvolusi", None))
        self.actionMean.setText(QCoreApplication.translate("Preprocessing", u"Mean", None))
        self.actionGaussian.setText(QCoreApplication.translate("Preprocessing", u"Gaussian", None))
        self.actionMean_Filter.setText(QCoreApplication.translate("Preprocessing", u"Mean Filter", None))
        self.actionGaussian_Filter.setText(QCoreApplication.translate("Preprocessing", u"Gaussian Filter", None))
        self.actionMedian_Filter.setText(QCoreApplication.translate("Preprocessing", u"Median Filter", None))
        self.actionDFT_Smoothing.setText(QCoreApplication.translate("Preprocessing", u"DFT Smoothing", None))
        self.actionBilateral_Filter.setText(QCoreApplication.translate("Preprocessing", u"Bilateral Filter", None))
        self.actionTranslasi.setText(QCoreApplication.translate("Preprocessing", u"Translasi", None))
        self.actionTranspose.setText(QCoreApplication.translate("Preprocessing", u"Transpose", None))
        self.actionCrop_Image.setText(QCoreApplication.translate("Preprocessing", u"Crop Image", None))
        self.action_min90d.setText(QCoreApplication.translate("Preprocessing", u"-90 Derajat", None))
        self.action_min45d.setText(QCoreApplication.translate("Preprocessing", u"-45 Derajat", None))
        self.action45d.setText(QCoreApplication.translate("Preprocessing", u"45 Derajat", None))
        self.action90d.setText(QCoreApplication.translate("Preprocessing", u"90 Derajat", None))
        self.action180d.setText(QCoreApplication.translate("Preprocessing", u"180 Derajat", None))
        self.action2X.setText(QCoreApplication.translate("Preprocessing", u"2X", None))
        self.action3X.setText(QCoreApplication.translate("Preprocessing", u"3X", None))
        self.action4X.setText(QCoreApplication.translate("Preprocessing", u"4X", None))
        self.actionQuarter.setText(QCoreApplication.translate("Preprocessing", u"1/4", None))
        self.actionHalf.setText(QCoreApplication.translate("Preprocessing", u"1/2", None))
        self.actionThree_Quarter.setText(QCoreApplication.translate("Preprocessing", u"3/4", None))
        self.action360p.setText(QCoreApplication.translate("Preprocessing", u"360p", None))
        self.action480p.setText(QCoreApplication.translate("Preprocessing", u"480p", None))
        self.action720p.setText(QCoreApplication.translate("Preprocessing", u"720p", None))
        self.action1080p.setText(QCoreApplication.translate("Preprocessing", u"1080p", None))
        self.actionplusmin.setText(QCoreApplication.translate("Preprocessing", u"Tambah dan Kurang", None))
        self.actionkalibagi.setText(QCoreApplication.translate("Preprocessing", u"Kali dan Bagi", None))
        self.actionAND.setText(QCoreApplication.translate("Preprocessing", u"AND", None))
        self.actionOR.setText(QCoreApplication.translate("Preprocessing", u"OR", None))
        self.actionXOR.setText(QCoreApplication.translate("Preprocessing", u"XOR", None))
        self.actionPre_processing.setText(QCoreApplication.translate("Preprocessing", u"Pre-processing", None))
        self.actionOperasi_Pencerahan.setText(QCoreApplication.translate("Preprocessing", u"Operasi Pencerahan", None))
        self.label_2.setText("")
        self.label.setText("")
        self.btn_loadImage.setText(QCoreApplication.translate("Preprocessing", u"Load Image", None))
        self.btn_saveImage.setText(QCoreApplication.translate("Preprocessing", u"Save Image", None))
        self.label_3.setText(QCoreApplication.translate("Preprocessing", u"Pre-Processing", None))
        self.btn_close.setText("")
        self.btn_minimize.setText("")
        self.btn_gtw.setText("")
        self.label_back.setText("")
        self.label_4.setText(QCoreApplication.translate("Preprocessing", u"Input", None))
        self.label_5.setText(QCoreApplication.translate("Preprocessing", u"Output", None))
        self.label_6.setText(QCoreApplication.translate("Preprocessing", u"0", None))
        self.btn_process.setText(QCoreApplication.translate("Preprocessing", u"Process", None))
        self.label_credits.setText(QCoreApplication.translate("Preprocessing", u"Final project by : C2", None))
        self.menuColor.setTitle(QCoreApplication.translate("Preprocessing", u"Color", None))
        self.menucoba.setTitle(QCoreApplication.translate("Preprocessing", u"coba", None))
        self.menuNoise_Reduction.setTitle(QCoreApplication.translate("Preprocessing", u"Noise Reduction", None))
        self.menuHistogram.setTitle(QCoreApplication.translate("Preprocessing", u"Histogram", None))
        self.menuContrast_n_Brightness.setTitle(QCoreApplication.translate("Preprocessing", u"Contrast n Brightness", None))
        self.menuOperasi_Geometri.setTitle(QCoreApplication.translate("Preprocessing", u"Operasi Geometri", None))
        self.menuRotasi.setTitle(QCoreApplication.translate("Preprocessing", u"Rotasi", None))
        self.menuResize.setTitle(QCoreApplication.translate("Preprocessing", u"Resize", None))
        self.menuZoom_In.setTitle(QCoreApplication.translate("Preprocessing", u"Zoom In", None))
        self.menuZoom_Out.setTitle(QCoreApplication.translate("Preprocessing", u"Zoom Out", None))
        self.menuSkewed.setTitle(QCoreApplication.translate("Preprocessing", u"Skewed", None))
        self.menuOperasi_Aritmatika.setTitle(QCoreApplication.translate("Preprocessing", u"Operasi Aritmatika", None))
        self.menuOperasi_Boolean.setTitle(QCoreApplication.translate("Preprocessing", u"Operasi Boolean", None))
    # retranslateUi

