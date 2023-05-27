import imp
import sys
import functools
import platform
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *   
from PyQt5.uic import loadUi
from PySide2 import QtCore, QtGui, QtWidgets
from PySide2.QtCore import *
from PySide2.QtGui import (QBrush, QColor, QConicalGradient, QCursor, QFont, QFontDatabase,
                           QIcon, QKeySequence, QLinearGradient, QPalette, QPainter, QPixmap, QRadialGradient)
from PySide2.QtWidgets import *

import cv2
import numpy as np
from matplotlib import pyplot as plt, widgets
import math
import pandas

from ui_loading_screen import Ui_Loading
from ui_main import Ui_MainWindow
from ui_home import Ui_Home
from ui_GUI import Ui_Preprocessing
from ui_loading_screen2 import Ui_Loading2



counter = 0

class Home(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        self.ui = Ui_Home()
        self.ui.setupUi(self)
        self.ui.btn_pre.clicked.connect(self.pre)
        self.ui.btn_detect.clicked.connect(self.detect)
        self.ui.btn_close.clicked.connect(self.quit)
        self.ui.btn_minimize.clicked.connect(self.showMinimized)

        
        self.setWindowFlag(QtCore.Qt.FramelessWindowHint)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)

        self.shadow = QGraphicsDropShadowEffect(self)
        self.shadow.setBlurRadius(20)
        self.shadow.setXOffset(0)
        self.shadow.setYOffset(0)
        self.shadow.setColor(QColor(0, 0, 0, 60))
        self.ui.shadow.setGraphicsEffect(self.shadow)

        self.show()


    @QtCore.Slot()
    def quit(self):
        app.quit()

    def detect(self):
        det=Loading2()
        widget.addWidget(det)
        widget.setCurrentIndex(widget.currentIndex()+1)
    
    def pre(self):
        det=Loading()
        widget.addWidget(det)
        widget.setCurrentIndex(widget.currentIndex()+1)




class Preprocessing(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        self.ui = Ui_Preprocessing()
        self.ui.setupUi(self)

        self.Image = None

        self.ui.btn_loadImage.clicked.connect(self.open)
        self.ui.btn_saveImage.clicked.connect(self.save)
        self.ui.btn_process.clicked.connect(self.gas)
        self.ui.btn_close.clicked.connect(self.quit)
        self.ui.btn_minimize.clicked.connect(self.showMinimized)
        self.ui.label_back.mousePressEvent = self.balik

        self.ui.btn_process.clicked.connect(self.gas)
        self.ui.slider.valueChanged.connect(self.slide)
        self.ui.actionGrayscale.triggered.connect(self.grayscale)
        self.ui.actionBiner.triggered.connect(self.biner)
        self.ui.actionNegative.triggered.connect(self.negative)
        self.ui.actionOperasi_Pencerahan.triggered.connect(self.brightness)
        self.ui.actionSimple_Contrast.triggered.connect(self.contrast)
        self.ui.actionContrast_Stretching.triggered.connect(
            self.ContrastStretching)
        self.ui.actionGamma_Correction.triggered.connect(self.gamma)
        self.ui.actionKonvolusi.triggered.connect(self.konvolusi)
        self.ui.actionMean_Filter.triggered.connect(self.mean)
        self.ui.actionGaussian_Filter.triggered.connect(self.gauss)
        self.ui.actionMedian_Filter.triggered.connect(self.med)
        self.ui.actionDFT_Smoothing.triggered.connect(self.DFTlp)
        self.ui.actionBilateral_Filter.triggered.connect(self.bilateral)
        self.ui.actionHistogram_Grayscale.triggered.connect(self.grayhist)
        self.ui.actionHistogram_RGB.triggered.connect(self.RGBhist)
        self.ui.actionHistogram_Equalization.triggered.connect(self.equalhist)
        self.ui.actionTranslasi.triggered.connect(self.translasi)
        self.ui.action_min90d.triggered.connect(self.rmin90d)
        self.ui.action_min45d.triggered.connect(self.rmin45d)
        self.ui.action45d.triggered.connect(self.r45d)
        self.ui.action90d.triggered.connect(self.r90d)
        self.ui.action180d.triggered.connect(self.r180d)
        self.ui.actionTranspose.triggered.connect(self.transpose)
        self.ui.action2X.triggered.connect(self.duax)
        self.ui.action3X.triggered.connect(self.tigax)
        self.ui.action4X.triggered.connect(self.empatx)
        self.ui.actionQuarter.triggered.connect(self.quarter)
        self.ui.actionHalf.triggered.connect(self.half)
        self.ui.actionThree_Quarter.triggered.connect(self.three_quarter)
        self.ui.action360p.triggered.connect(self.sk360)
        self.ui.action480p.triggered.connect(self.sk480)
        self.ui.action720p.triggered.connect(self.sk720)
        self.ui.action1080p.triggered.connect(self.sk1080)
        self.ui.actionCrop_Image.triggered.connect(self.crop)
        self.ui.actionplusmin.triggered.connect(self.plusmin)
        self.ui.actionkalibagi.triggered.connect(self.kalbag)
        self.ui.actionAND.triggered.connect(self.AND)
        self.ui.actionOR.triggered.connect(self.OR)
        self.ui.actionXOR.triggered.connect(self.XOR)
        self.ui.actionCoba.triggered.connect(self.ini)

        self.setWindowFlag(QtCore.Qt.FramelessWindowHint)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)

        # self.shadow = QGraphicsDropShadowEffect(self)
        # self.shadow.setBlurRadius(20)
        # self.shadow.setXOffset(0)
        # self.shadow.setYOffset(0)
        # self.shadow.setColor(QColor(0, 0, 0, 60))
        # self.ui.shadow.setGraphicsEffect(self.shadow)

        self.show()

    @QtCore.Slot()
    def quit(self):
        app.quit()
    
    def open(self):
        imagePath, _ = QFileDialog.getOpenFileName()
        self.Image = cv2.imread(imagePath)
        pixmap = QPixmap(imagePath)
        self.ui.label.setPixmap(pixmap)
        self.ui.label.setAlignment(QtCore.Qt.AlignHCenter |
                                QtCore.Qt.AlignVCenter)
        self.ui.label.setScaledContents(True)

    def balik(self, event):
        det=Home()
        widget.addWidget(det)
        widget.setCurrentIndex(widget.currentIndex()+1)

    def save(self):
        fname, filter = QFileDialog.getSaveFileName(
            self, 'Save As', 'D:\\', "Image Files (*.jpg)")
        if fname:
            cv2.imwrite(fname, self.Image)
        else:
            print('Error')

    def slide(self, value):
        self.ui.label_6.setText(str(value))

    def grayscale(self):
        H, W = self.Image.shape[:2]
        pr = np.zeros((H, W), np.uint8)
        for i in range(H):
            for j in range(W):
                pr[i, j] = np.clip(0.299 * self.Image[i, j, 0] +
                                     0.587 * self.Image[i, j, 1] +
                                     0.114 * self.Image[i, j, 2], 0, 255)
        # self.Image = gray
        filename = 'pre-processing/pra.jpg'
        cv2.imwrite(filename, pr)
        self.displayImage(2)

    def biner(self):
        # error handling
        try:
            self.Image = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)
            pr = self.Image
        except:
            pass

        H, W = pr.shape[:2]
        for i in range(H):
            for j in range(W):
                a = self.Image.item(i, j)
                if a == 0:
                    b = 0
                elif a < 180:
                    b = 1
                elif a > 180:
                    b = 255
                self.Image.itemset((i, j), b)

        filename = 'pre-processing/pra.jpg'
        cv2.imwrite(filename, pr)
        self.displayImage(2)

    def negative(self):
        # error handling
        try:
            self.Image = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)
            pr = self.Image
        except:
            pass

        H, W = pr.shape[:2]
        for i in np.arange(H):
            for j in np.arange(W):
                a = self.Image.item(i, j)
                b = math.ceil(255 - a)
                self.Image.itemset((i, j), b)

        filename = 'pre-processing/pra.jpg'
        cv2.imwrite(filename, pr)
        self.displayImage(2)

    def brightness(self):
        # error handling
        try:
            self.Image = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)
            pr = self.Image
        except:
            pass

        H, W = pr.shape[:2]
        self.ui.slider.setMinimum(0)
        self.ui.slider.setMaximum(100)

        brightness = self.ui.slider.value()
        for i in range(H):
            for j in range(W):
                a = self.Image.item(i, j)
                b = np.clip(a + brightness, 0, 255)

                self.Image.itemset((i, j), b)

        filename = 'pre-processing/pra.jpg'
        cv2.imwrite(filename, pr)
        self.displayImage(2)

    def contrast(self):
        # error handling
        try:
            self.Image = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)
            pr = self.Image
        except:
            pass

        H, W = pr.shape[:2]
        self.ui.slider.setMinimum(0)
        self.ui.slider.setMaximum(100)
        contrast = self.ui.slider.value()
        for i in range(H):
            for j in range(W):
                a = self.Image.item(i, j)
                b = np.clip(a * contrast, 0, 255)

                self.Image.itemset((i, j), b)

        filename = 'pre-processing/pra.jpg'
        cv2.imwrite(filename, pr)
        self.displayImage(2)

    def ContrastStretching(self):
        # error handling
        try:
            self.Image = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)
            pr = self.Image
        except:
            pass

        H, W = pr.shape[:2]
        minV = np.min(self.Image)
        maxV = np.max(self.Image)
        for i in range(H):
            for j in range(W):
                a = self.Image.item(i, j)
                b = float(a - minV) / (maxV - minV) * 255

                self.Image.itemset((i, j), b)

        filename = 'pre-processing/pra.jpg'
        cv2.imwrite(filename, pr)
        self.displayImage(2)

    def gamma(self):
        def adjust_gamma(image, gamma=1.0):
            invGamma = 1.0/gamma
            table = np.array([((i / 255.0) ** invGamma) *
                             255 for i in np.arange(0, 256)]).astype("uint8")
            return cv2.LUT(image, table)
        original = self.Image
        for gamma in np.arange(0.0, 3.5, 0.5):
            if gamma == 1:
                continue

            gamma = gamma if gamma > 0 else 0.1
            adjusted = adjust_gamma(original, gamma)
            cv2.putText(adjusted, f"g={gamma}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            cv2.imshow("images", np.hstack([original, adjusted]))
            cv2.waitKey(0)


    def conv(self, X, F):
        X_height = X.shape[0]
        X_width = X.shape[1]
        F_height = F.shape[0]
        F_width = F.shape[1]
        H = (F_height) // 2
        W = (F_width) // 2
        out = np.zeros((X_height, X_width))
        for i in np.arange(H + 1, X_height - H):
            for j in np.arange(W + 1, X_width - W):
                sum = 0
                for k in np.arange(-H, H + 1):
                    for l in np.arange(-W, W + 1):
                        a = X[i + k, j + l]
                        w = F[H + k, W + l]
                        sum += (w * a)
                out[i, j] = sum
        return out


    def konvolusi(self):
        img = self.Image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kernel = np.array(
            [
                [1, 1, 1],
                [1, 1, 1],
                [1, 1, 1]
            ]
        )
        pr = self.conv(img, kernel)
        plt.imshow(pr, cmap='gray', interpolation='bicubic')
        filename = 'pre-processing/pra.jpg'
        cv2.imwrite(filename, img)
        self.displayImage(2)

    def mean(self):
        img = self.Image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kernel = (1.0/9) * np.array(
            [
                [1, 1, 1],
                [1, 1, 1],
                [1, 1, 1]
            ]
        )
        pr = self.conv(img, kernel)
        plt.imshow(pr, cmap='gray', interpolation='bicubic')
        filename = 'pre-processing/pra.jpg'
        cv2.imwrite(filename, pr)
        self.displayImage(2)


    def gauss(self):
        img = self.Image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kernel = (1.0/354) * np.array(
            [
                [1, 5, 7, 5, 1],
                [5, 20, 33, 20, 5],
                [7, 33, 55, 33, 7],
                [5, 20, 33, 20, 5],
                [1, 5, 7, 5, 1]
            ]
        )
        img_out = self.conv(img, kernel)
        plt.imshow(img_out, cmap='gray', interpolation='bicubic')
        filename = 'pre-processing/pra.jpg'
        cv2.imwrite(filename, img_out)
        self.displayImage(2)

    def med(self):
        img = self.Image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_out = img.copy()
        h, w = img.shape[:2]
        for i in np.arange(3, h-3):
            for j in np.arange(3, w-3):
                neighbors = []
                for k in np.arange(-3, 4):
                    for l in np.arange(-3, 4):
                        a = img.item(i+k, j+l)
                        neighbors.append(a)
                neighbors.sort()
                median = neighbors[24]
                b = median
                img_out.itemset((i, j), b)
        plt.imshow(img_out, cmap='gray', interpolation='bicubic')
        filename = 'pre-processing/pra.jpg'
        cv2.imwrite(filename, img_out)
        self.displayImage(2)

    def DFTlp(self):
        img = cv2.imread('Input-Set/coba.jfif', 0)

        dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)

        magnitude_spectrum = 20 * \
            np.log((cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])))

        rows, cols = img.shape
        crow, ccol = int(rows / 2), int(cols / 2)
        mask = np.zeros((rows, cols, 2), np.uint8)
        r = 80
        center = [crow, ccol]
        x, y = np.ogrid[:rows, :cols]
        mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r*r
        mask[mask_area] = 1

        fshift = dft_shift * mask
        fshift_mask_mag = 20 * \
            np.log((cv2.magnitude(fshift[:, :, 0], fshift[:, :, 1])))
        f_ishift = np.fft.ifftshift(fshift)

        img_back = cv2.idft(f_ishift)
        img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

        fig = plt.figure(figsize=(12, 12))
        ax1 = fig.add_subplot(2, 2, 1)
        ax1.imshow(img, cmap='gray')
        ax1.title.set_text('Input Image')

        ax2 = fig.add_subplot(2, 2, 2)
        ax2.imshow(magnitude_spectrum, cmap='gray')
        ax2.title.set_text('FFT of Image')

        ax3 = fig.add_subplot(2, 2, 3)
        ax3.imshow(fshift_mask_mag, cmap='gray')
        ax3.title.set_text('FFT + Mask')

        ax4 = fig.add_subplot(2, 2, 4)
        ax4.imshow(img_back, cmap='gray')
        ax4.title.set_text('Inverse Fourier')

        self.displayImage(2)

    def bilateral(self):
        img = self.Image

        def bilateral_filter_image(image_matrix, window_length=7, sigma_color=25, sigma_space=9, mask_image_matrix=None):
            mask_image_matrix = np.zeros(
                (image_matrix.shape[0], image_matrix.shape[1])) if mask_image_matrix is None else mask_image_matrix  # default: filtering the entire image
            # transfer the image_matrix to type int32，for uint cann't represent the negative number afterward
            image_matrix = image_matrix.astype(np.int32)

            def limit(x):
                x = 0 if x < 0 else x
                x = 255 if x > 255 else x
                return x
            limit_ufun = np.vectorize(limit, otypes=[np.uint8])

            def look_for_gaussion_table(delta):
                return delta_gaussion_dict[delta]

            def generate_bilateral_filter_distance_matrix(window_length, sigma):
                distance_matrix = np.zeros((window_length, window_length, 3))
                left_bias = int(math.floor(-(window_length - 1) / 2))
                right_bias = int(math.floor((window_length - 1) / 2))
                for i in range(left_bias, right_bias+1):
                    for j in range(left_bias, right_bias+1):
                        distance_matrix[i-left_bias][j -
                                                     left_bias] = math.exp(-(i**2+j**2)/(2*(sigma**2)))
                return distance_matrix
            delta_gaussion_dict = {
                i: math.exp(-i ** 2 / (2 * (sigma_color**2))) for i in range(256)}
            # to accelerate the process of get the gaussion matrix about color.key:color difference，value:gaussion weight
            look_for_gaussion_table_ufun = np.vectorize(
                look_for_gaussion_table, otypes=[np.float64])
            bilateral_filter_distance_matrix = generate_bilateral_filter_distance_matrix(
                window_length, sigma_space)  # get the gaussion weight about distance directly

            margin = int(window_length / 2)
            left_bias = math.floor(-(window_length - 1) / 2)
            right_bias = math.floor((window_length - 1) / 2)
            filter_image_matrix = image_matrix.astype(np.float64)

            for i in range(0 + margin, image_matrix.shape[0] - margin):
                for j in range(0 + margin, image_matrix.shape[1] - margin):
                    if mask_image_matrix[i][j] == 0:
                        filter_input = image_matrix[i + left_bias:i + right_bias + 1,
                                                    j + left_bias:j + right_bias + 1]  # get the input window
                        bilateral_filter_value_matrix = look_for_gaussion_table_ufun(
                            np.abs(filter_input-image_matrix[i][j]))  # get the gaussion weight about color
                        # multiply color gaussion weight  by distane gaussion weight to get the no-norm weigth matrix
                        bilateral_filter_matrix = np.multiply(
                            bilateral_filter_value_matrix, bilateral_filter_distance_matrix)
                        bilateral_filter_matrix = bilateral_filter_matrix / \
                            np.sum(bilateral_filter_matrix, keepdims=False, axis=(
                                0, 1))  # normalize the weigth matrix
                        # multiply the input window by the weigth matrix，then get the sum of channels seperately
                        filter_output = np.sum(np.multiply(
                            bilateral_filter_matrix, filter_input), axis=(0, 1))
                        filter_image_matrix[i][j] = filter_output
            filter_image_matrix = limit_ufun(
                filter_image_matrix)  # limit the range
            return filter_image_matrix

        image_matrix = img
        bilateral_filtered = bilateral_filter_image(image_matrix)
        filename = 'pre-processing/pra.jpg'
        cv2.imwrite(filename, bilateral_filtered)
        self.displayImage(2)

    def grayhist(self):
        H, W = self.Image.shape[:2]
        gray = np.zeros((H, W), np.uint8)
        for i in range(H):
            for j in range(W):
                gray[i, j] = np.clip(0.299 * self.Image[i, j, 0] +
                                     0.587 * self.Image[i, j, 1] +
                                     0.114 * self.Image[i, j, 2],
                                     0, 255
                                     )

        self.Image = gray
        filename = 'pre-processing/pra.jpg'
        cv2.imwrite(filename, gray)
        self.displayImage(2)
        plt.hist(self.Image.ravel(), 255, [0, 255])
        plt.show()



    def RGBhist(self):
        color = ('b', 'g', 'r')  # data disimpan pada tuple
        for i, col in enumerate(color):  # loop berdasarkan warna
            histo = cv2.calcHist([self.Image], [i], None, [256], [
                0, 256])  # mengitung histogram dari array
            plt.plot(histo, color=col)  # plotting untuk histogram
            plt.xlim([0, 256])  # set batas sb. x
            filename = 'pre-processing/pra.jpg'
            cv2.imwrite(filename, self.Image)
            self.displayImage(2)
            plt.show()  # visualisasi histogram pada windows baru


    def equalhist(self):
        # menghitung histogram dari kumpulan data (img array--> 1D)
        hist, bins = np.histogram(self.Image.flatten(), 256, [0, 256])
        cdf = hist.cumsum()  # menghitung jumlah kumulatif elemen array
        cdf_normalized = cdf * hist.max() / cdf.max()  # rumus untuk normalisasi
        # menutupi (masking) array di mana sama dengan nilai yang diberikan
        cdf_m = np.ma.masked_equal(cdf, 0)
        cdf_m = (cdf_m - cdf_m.min()) * 255 / \
                (cdf_m.max() - cdf_m.min())  # proses perhitungan
        # mengisi nilai array dengan nilai skalar
        cdf = np.ma.filled(cdf_m, 0).astype("uint8")
        self.Image = cdf[self.Image]  # mengaplikasikan ke citra
        filename = 'pre-processing/pra.jpg'
        cv2.imwrite(filename, cdf[self.Image])
        self.displayImage(2)  # menampilkan gambar pada label 2

        plt.plot(cdf_normalized, color="b")  # plotting sesuai normalisasi
        plt.hist(self.Image.flatten(), 256, [
            0, 256], color="r")  # membuat histogram
        plt.xlim([0, 256])  # mengatur sumbu x
        # menampilkan teks pada histogram (atas kiri)
        plt.legend(("cdf", "histogram"), loc="upper left")
        plt.show()  # visualisasi hasil histogram

    def translasi(self):
        H, W = self.Image.shape[:2]
        quarter_H, quarter_W = H/2, W/2
        T = np.float32([[1, 0, quarter_W], [0, 1, quarter_H]])
        img = cv2.warpAffine(self.Image, T, (W, H))
        self.Image = img
        filename = 'pre-processing/pra.jpg'
        cv2.imwrite(filename, img)
        self.displayImage(2)


    def rotasi(self, degree):
        h, w = self.Image.shape[:2]

        rotationMatrix = cv2.getRotationMatrix2D((w/2, h/2), degree, 0.7)

        cos = np.abs(rotationMatrix[0, 0])
        sin = np.abs(rotationMatrix[0, 1])

        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))

        rotationMatrix[0, 2] += (nW / 2) - w / 2
        rotationMatrix[1, 2] += (nH / 2) - h / 2
        rot_image = cv2.warpAffine(self.Image, rotationMatrix, (h, w))
        self.Image = rot_image
        filename = 'pre-processing/pra.jpg'
        cv2.imwrite(filename, rot_image)
        self.displayImage(2)

    def rmin90d(self):
        self.rotasi(-90)

    def rmin45d(self):
        self.rotasi(-45)

    def r45d(self):
        self.rotasi(45)

    def r90d(self):
        self.rotasi(90)

    def r180d(self):
        self.rotasi(180)

    def transpose(self):
        transp = cv2.transpose(self.Image)
        self.Image = transp
        filename = 'pre-processing/pra.jpg'
        cv2.imwrite(filename, transp)
        self.displayImage(2)

    # here
    def zoom(self, size):
        scale = size
        resize_img = cv2.resize(
            self.Image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        cv2.imshow('Before', self.Image)
        cv2.imshow('After', resize_img)
        filename = 'pre-processing/pra.jpg'
        cv2.imwrite(filename, resize_img)
        self.displayImage(2)

    def duax(self):
        self.zoom(2)

    def tigax(self):
        self.zoom(3)

    def empatx(self):
        self.zoom(4)

    def quarter(self):
        self.zoom(1/4)

    def half(self):
        self.zoom(1/2)

    def three_quarter(self):
        self.zoom(3/4)

    def skew(self, size):
        scale = size
        img = cv2.resize(self.Image, scale, interpolation=cv2.INTER_AREA)
        cv2.imshow('Before', self.Image)
        cv2.imshow('After', img)
        filename = 'pre-processing/pra.jpg'
        cv2.imwrite(filename, img)
        self.displayImage(2)

    def sk360(self):
        self.skew((640, 360))

    def sk480(self):
        self.skew((852, 480))

    def sk720(self):
        self.skew((1280, 720))

    def sk1080(self):
        self.skew((1920, 1080))

    def crop(self):
        x1 = 100
        y1 = 100
        x2 = 300
        y2 = 400
        crop_img = self.Image[x1:x2, y1:y2]
        cv2.imshow('Before', self.Image)
        cv2.imshow('After', crop_img)
        filename = 'pre-processing/pra.jpg'
        cv2.imwrite(filename, crop_img)
        self.displayImage(2)

    def plusmin(self):
        img1 = cv2.imread('Input-Set/coba.jfif', 0)
        img2 = cv2.imread('Input-Set/coba1.jpg', 0)
        tbh = img1 + img2
        krg = img1 - img2
        print('Image 1:')
        print(img1)
        print('Image 2:')
        print(img2)
        print('Penjumlahan:')
        print(tbh)
        print('Pengurangan:')
        print(krg)
        cv2.imshow('Image 1', img1)
        cv2.imshow('Image 2', img2)
        cv2.imshow('Penjumlahan', tbh)
        cv2.imshow('Pengurangan', krg)
        cv2.waitKey()
        cv2.destroyAllWindows()

    def kalbag(self):
        img1 = cv2.imread('Input-Set/coba.jfif', 0)
        img2 = cv2.imread('Input-Set/coba1.jpg', 0)
        kl = img1 * img2
        bg = img1 / img2
        print('Image 1:')
        print(img1)
        print('Image 2:')
        print(img2)
        print('Perkalian:')
        print(kl)
        print('Pembagian:')
        print(bg)
        cv2.imshow('Image 1', img1)
        cv2.imshow('Image 2', img2)
        cv2.imshow('Perkalian', kl)
        cv2.imshow('Pembagian', bg)
        cv2.waitKey()
        cv2.destroyAllWindows()

    def AND(self):
        img1 = cv2.imread('Input-Set/coba.jfif', 1)
        img2 = cv2.imread('Input-Set/coba1.jpg', 1)
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        op = cv2.bitwise_and(img1, img2)
        print('Image 1:')
        print(img1)
        print('Image 2:')
        print(img2)
        print('AND:')
        print(op)
        cv2.imshow('Image 1', img1)
        cv2.imshow('Image 2', img2)
        cv2.imshow('Operasi AND', op)
        cv2.waitKey()
        cv2.destroyAllWindows()

    def OR(self):
        img1 = cv2.imread('Input-Set/coba.jfif', 1)
        img2 = cv2.imread('Input-Set/coba1a.jpg', 1)
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        op = cv2.bitwise_or(img1, img2)
        print('Image 1:')
        print(img1)
        print('Image 2:')
        print(img2)
        print('OR:')
        print(op)
        cv2.imshow('Image 1', img1)
        cv2.imshow('Image 2', img2)
        cv2.imshow('Operasi OR', op)
        cv2.waitKey()
        cv2.destroyAllWindows()

    def XOR(self):
        img1 = cv2.imread('Input-Set/coba.jfif', 1)
        img2 = cv2.imread('Input-Set/coba2a.jpg', 1)
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        op = cv2.bitwise_xor(img1, img2)
        print('Image 1:')
        print(img1)
        print('Image 2:')
        print(img2)
        print('XOR:')
        print(op)
        cv2.imshow('Image 1', img1)
        cv2.imshow('Image 2', img2)
        cv2.imshow('Operasi XOR', op)
        cv2.waitKey()
        cv2.destroyAllWindows()

    def ini(self):
        gray = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)
        blur = cv2.blur(gray, (3, 3))
        img_log = (np.log(blur+1)/(np.log(1+np.max(blur))))*255
        img_log = np.array(img_log, dtype=np.uint8)
        img_log = blur
        bilateral = cv2.bilateralFilter(img_log, 5, 75, 75)
        edges = cv2.Canny(bilateral, 100, 200)
        kernel = np.ones((5, 5), np.uint8)
        closing = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        orb = cv2.ORB_create(nfeatures=1500)
        keypoints, descriptors = orb.detectAndCompute(closing, None)
        featuredImg = cv2.drawKeypoints(closing, keypoints, None)
        cv2.imwrite('Output-Set/CrackDetected-7.jpg', featuredImg)
        self.Image = featuredImg
        filename = 'pre-processing/pra.jpg'
        cv2.imwrite(filename, featuredImg)
        self.displayImage(2)

    def grab_buffer(fig):
        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        return data

    def gas(self):
        img = self.Image

        def grayS(img):
            H, W = img.shape[:2]
            gray = np.zeros((H, W), np.uint8)
            for i in range(H):
                for j in range(W):
                    gray[i, j] = np.clip(0.299 * img[i, j, 0] +
                                         0.587 * img[i, j, 1] +
                                         0.114 * img[i, j, 2], 0, 255)
            return gray

        def meanS(image):
            img = image
            kernel = (1/9) * np.array(
                [
                    [1, 1, 1],
                    [1, 1, 1],
                    [1, 1, 1]
                ]
            )
            blur = self.conv(img, kernel)
            return blur

        # def bilateralS(image):
        #     def bilateral_filter_image(image_matrix, window_length=5, sigma_color=75, sigma_space=75, mask_image_matrix=None):
        #         mask_image_matrix = np.zeros(
        #             (image_matrix.shape[0], image_matrix.shape[1])) if mask_image_matrix is None else mask_image_matrix
        #         image_matrix = image_matrix.astype(np.int32)

        #         def limit(x):
        #             x = 0 if x < 0 else x
        #             x = 255 if x > 255 else x
        #             return x
        #         limit_ufun = np.vectorize(limit, otypes=[np.uint8])

        #         def look_for_gaussion_table(delta):
        #             return delta_gaussion_dict[delta]

            #     def generate_bilateral_filter_distance_matrix(window_length, sigma):
            #         distance_matrix = np.zeros(
            #             (window_length, window_length, 3))
            #         left_bias = int(math.floor(-(window_length - 1) / 2))
            #         right_bias = int(math.floor((window_length - 1) / 2))
            #         for i in range(left_bias, right_bias+1):
            #             for j in range(left_bias, right_bias+1):
            #                 distance_matrix[i-left_bias][j -
            #                                              left_bias] = math.exp(-(i**2+j**2)/(2*(sigma**2)))
            #         return distance_matrix
            #     delta_gaussion_dict = {
            #         i: math.exp(-i ** 2 / (2 * (sigma_color**2))) for i in range(256)}
            #     look_for_gaussion_table_ufun = np.vectorize(
            #         look_for_gaussion_table, otypes=[np.float64])
            #     bilateral_filter_distance_matrix = generate_bilateral_filter_distance_matrix(
            #         window_length, sigma_space)

            #     margin = int(window_length / 2)
            #     left_bias = math.floor(-(window_length - 1) / 2)
            #     right_bias = math.floor((window_length - 1) / 2)
            #     filter_image_matrix = image_matrix.astype(np.float64)

            #     for i in range(0 + margin, image_matrix.shape[0] - margin):
            #         for j in range(0 + margin, image_matrix.shape[1] - margin):
            #             if mask_image_matrix[i][j] == 0:
            #                 filter_input = image_matrix[i + left_bias:i + right_bias + 1,
            #                                             j + left_bias:j + right_bias + 1]
            #                 bilateral_filter_value_matrix = look_for_gaussion_table_ufun(
            #                     np.abs(filter_input-image_matrix[i][j]))
            #                 bilateral_filter_matrix = np.multiply(
            #                     bilateral_filter_value_matrix, bilateral_filter_distance_matrix)
            #                 bilateral_filter_matrix = bilateral_filter_matrix / \
            #                     np.sum(bilateral_filter_matrix,
            #                            keepdims=False, axis=(0, 1))
            #                 filter_output = np.sum(np.multiply(
            #                     bilateral_filter_matrix, filter_input), axis=(0, 1))
            #                 filter_image_matrix[i][j] = filter_output
            #     filter_image_matrix = limit_ufun(filter_image_matrix)
            #     return filter_image_matrix

            # img = image
            # image_matrix = img
            # bilateral_filtered = bilateral_filter_image(image_matrix)

        gray = grayS(img)
        blur = meanS(gray)
        blur = np.array(blur, dtype=np.uint8)
        img_log = (np.log(blur+1)/(np.log(1+np.max(blur))))*255
        img_log = np.array(img_log, dtype=np.uint8)
        img_log = blur
        # bilateral = bilateralS(img_log)
        bilateral = cv2.bilateralFilter(img_log, 5, 75, 75)
        self.Image = bilateral
        df = pandas.DataFrame(bilateral)
        df.to_excel(str("Pradeteksi-pixel") + ".xlsx")
        filename = 'pre-processing/pra.jpg'
        cv2.imwrite(filename, bilateral)
        self.displayImage(2)

        # belum dijabarkan karena belum digunakan
        # edges = cv2.Canny(bilateral, 100, 200)
        # kernel = np.ones((5, 5), np.uint8)
        # closing = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        # orb = cv2.ORB_create(nfeatures=1500)
        # keypoints, descriptors = orb.detectAndCompute(closing, None)
        # featuredImg = cv2.drawKeypoints(closing, keypoints, None)
        # cv2.imwrite('Output-Set/CrackDetected-7.jpg', featuredImg)

    def displayImage(self, windows=1):
        qformat = QImage.Format_Indexed8

        if len(self.Image.shape) == 3:
            if (self.Image.shape[2]) == 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888
        img = QImage(
            self.Image, self.Image.shape[1], self.Image.shape[0], self.Image.strides[0], qformat)
        img = img.rgbSwapped()
        if windows == 1:
            self.ui.label.setPixmap(QPixmap.fromImage(img))
            self.label.setAlignment(
                QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
            self.ui.label.setScaledContents(True)
        if windows == 2:
            pixmap = QPixmap('pre-processing/pra.jpg')
            self.ui.label_2.setPixmap(pixmap)
            self.ui.label_2.setAlignment(
                QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
            self.ui.label_2.setScaledContents(True)
        print(self.Image)






class MainWindow(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.Image = None

        self.ui.btn_loadImage.clicked.connect(self.open)
        self.ui.btn_saveImage.clicked.connect(self.save)
        self.ui.btn_process.clicked.connect(self.gas)
        self.ui.btn_close.clicked.connect(self.quit)
        self.ui.btn_minimize.clicked.connect(self.showMinimized)

        self.ui.label_back.mousePressEvent = self.balik     

        self.setWindowFlag(QtCore.Qt.FramelessWindowHint)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)

        self.shadow = QGraphicsDropShadowEffect(self)
        self.shadow.setBlurRadius(20)
        self.shadow.setXOffset(0)
        self.shadow.setYOffset(0)
        self.shadow.setColor(QColor(0, 0, 0, 60))
        self.ui.shadow.setGraphicsEffect(self.shadow)

    def balik(self, event):
        det=Home()
        widget.addWidget(det)
        widget.setCurrentIndex(widget.currentIndex()+1)
    
    def eventFilter(self, obj, event):
        if obj is self.w and event.type() == QtCore.QEvent.Close:
            self.quit_app()
            event.ignore()
            return True
        return super(Manager, self).eventFilter(obj, event)

    @QtCore.Slot()
    def quit(self):
        print('CLEAN EXIT')
        self.removeEventFilter(self)
        app.quit()
    
    def open(self):
        imagePath, _ = QFileDialog.getOpenFileName()
        self.Image = cv2.imread(imagePath)
        pixmap = QPixmap(imagePath)
        self.ui.label.setPixmap(pixmap)
        self.ui.label.setAlignment(QtCore.Qt.AlignHCenter |
                                QtCore.Qt.AlignVCenter)
        self.ui.label.setScaledContents(True)

    def save(self):
        fname, filter = QFileDialog.getSaveFileName(
            self, 'Save As', 'D:\\', "Image Files (*.jpg)")
        if fname:
            cv2.imwrite(fname, self.Image)
        else:
            print('Error')

    def gas(self):
        img = self.Image

        def conv(X, F):
            X_height = X.shape[0]
            X_width = X.shape[1]
            F_height = F.shape[0]
            F_width = F.shape[1]
            H = (F_height) // 2
            W = (F_width) // 2
            out = np.zeros((X_height, X_width))
            for i in np.arange(H + 1, X_height - H):
                for j in np.arange(W + 1, X_width - W):
                    sum = 0
                    for k in np.arange(-H, H + 1):
                        for l in np.arange(-W, W + 1):
                            a = X[i + k, j + l]
                            w = F[H + k, W + l]
                            sum += (w * a)
                    out[i, j] = sum
            return out

        def grayS(img):
            H, W = img.shape[:2]
            gray = np.zeros((H, W), np.uint8)
            for i in range(H):
                for j in range(W):
                    gray[i, j] = np.clip(0.299 * img[i, j, 0] +
                                         0.587 * img[i, j, 1] +
                                         0.114 * img[i, j, 2], 0, 255)
            return gray

        def meanS(image):
            img = image
            kernel = (1/9) * np.array(
                [
                    [1, 1, 1],
                    [1, 1, 1],
                    [1, 1, 1]
                ]
            )
            blur = conv(img, kernel)
            return blur


        # def bilateralS(image):
        #     def bilateral_filter_image(image_matrix, window_length=5, sigma_color=75, sigma_space=75, mask_image_matrix=None):
        #         mask_image_matrix = np.zeros(
        #             (image_matrix.shape[0], image_matrix.shape[1])) if mask_image_matrix is None else mask_image_matrix
        #         image_matrix = image_matrix.astype(np.int32)

        #         def limit(x):
        #             x = 0 if x < 0 else x
        #             x = 255 if x > 255 else x
        #             return x
        #         limit_ufun = np.vectorize(limit, otypes=[np.uint8])

        #         def look_for_gaussion_table(delta):
        #             return delta_gaussion_dict[delta]

            #     def generate_bilateral_filter_distance_matrix(window_length, sigma):
            #         distance_matrix = np.zeros(
            #             (window_length, window_length, 3))
            #         left_bias = int(math.floor(-(window_length - 1) / 2))
            #         right_bias = int(math.floor((window_length - 1) / 2))
            #         for i in range(left_bias, right_bias+1):
            #             for j in range(left_bias, right_bias+1):
            #                 distance_matrix[i-left_bias][j -
            #                                              left_bias] = math.exp(-(i**2+j**2)/(2*(sigma**2)))
            #         return distance_matrix
            #     delta_gaussion_dict = {
            #         i: math.exp(-i ** 2 / (2 * (sigma_color**2))) for i in range(256)}
            #     look_for_gaussion_table_ufun = np.vectorize(
            #         look_for_gaussion_table, otypes=[np.float64])
            #     bilateral_filter_distance_matrix = generate_bilateral_filter_distance_matrix(
            #         window_length, sigma_space)

            #     margin = int(window_length / 2)
            #     left_bias = math.floor(-(window_length - 1) / 2)
            #     right_bias = math.floor((window_length - 1) / 2)
            #     filter_image_matrix = image_matrix.astype(np.float64)

            #     for i in range(0 + margin, image_matrix.shape[0] - margin):
            #         for j in range(0 + margin, image_matrix.shape[1] - margin):
            #             if mask_image_matrix[i][j] == 0:
            #                 filter_input = image_matrix[i + left_bias:i + right_bias + 1,
            #                                             j + left_bias:j + right_bias + 1]
            #                 bilateral_filter_value_matrix = look_for_gaussion_table_ufun(
            #                     np.abs(filter_input-image_matrix[i][j]))
            #                 bilateral_filter_matrix = np.multiply(
            #                     bilateral_filter_value_matrix, bilateral_filter_distance_matrix)
            #                 bilateral_filter_matrix = bilateral_filter_matrix / \
            #                     np.sum(bilateral_filter_matrix,
            #                            keepdims=False, axis=(0, 1))
            #                 filter_output = np.sum(np.multiply(
            #                     bilateral_filter_matrix, filter_input), axis=(0, 1))
            #                 filter_image_matrix[i][j] = filter_output
            #     filter_image_matrix = limit_ufun(filter_image_matrix)
            #     return filter_image_matrix

            # img = image
            # image_matrix = img
            # bilateral_filtered = bilateral_filter_image(image_matrix)

        gray = grayS(img)
        
        blur = meanS(gray)
        cv2.imwrite('steps/mean.jpg', gray)
        pixmap = QPixmap('steps/mean.jpg')
        self.ui.label_2.setPixmap(pixmap)
        self.ui.label_2.setAlignment(QtCore.Qt.AlignHCenter |
                                QtCore.Qt.AlignVCenter)
        self.ui.label_2.setScaledContents(True)
        blur = np.array(blur, dtype=np.uint8)
        img_log = (np.log(blur+1)/(np.log(1+np.max(blur))))*255
        img_log = np.array(img_log, dtype=np.uint8)
        img_log = blur

        # bilateral = bilateralS(img_log)
        bilateral = cv2.bilateralFilter(img_log, 5, 75, 75)
        cv2.imwrite('steps/bilateral.jpg', bilateral)
        pixmap = QPixmap('steps/bilateral.jpg')
        self.ui.label_7.setPixmap(pixmap)
        self.ui.label_7.setAlignment(QtCore.Qt.AlignHCenter |
                                QtCore.Qt.AlignVCenter)
        self.ui.label_7.setScaledContents(True)
        
        edges = cv2.Canny(bilateral, 100, 200)
        cv2.imwrite('steps/canny.jpg', edges)
        pixmap = QPixmap('steps/canny.jpg')
        self.ui.label_8.setPixmap(pixmap)
        self.ui.label_8.setAlignment(QtCore.Qt.AlignHCenter |
                                QtCore.Qt.AlignVCenter)
        self.ui.label_8.setScaledContents(True)

        kernel = np.ones((5, 5), np.uint8)
        closing = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        cv2.imwrite('steps/closing.jpg', closing)
        pixmap = QPixmap('steps/closing.jpg')
        self.ui.label_9.setPixmap(pixmap)
        self.ui.label_9.setAlignment(QtCore.Qt.AlignHCenter |
                                QtCore.Qt.AlignVCenter)
        self.ui.label_9.setScaledContents(True)

        orb = cv2.ORB_create(nfeatures=1500)
        keypoints, descriptors = orb.detectAndCompute(closing, None)
        featuredImg = cv2.drawKeypoints(closing, keypoints, None)
        cv2.imwrite('steps/output.jpg', featuredImg)
        pixmap = QPixmap('steps/output.jpg')
        self.ui.label_10.setPixmap(pixmap)
        self.ui.label_10.setAlignment(QtCore.Qt.AlignHCenter |
                                QtCore.Qt.AlignVCenter)
        self.ui.label_10.setScaledContents(True)
        number_of_edges = np.count_nonzero(closing)

        print(number_of_edges)


        if number_of_edges > 1200:
            print("Wrinkle Found ")
            self.ui.label_14.setText("Wrinkle Found")
        else:
            print("No Wrinklle Found ")
            self.ui.label_14.setText("No Wrinkle Found")


        df = pandas.DataFrame(closing)
        df.to_excel(str("deteksi-pixel") + ".xlsx")
        




class Loading(QMainWindow):

    def __init__(self):
        QMainWindow.__init__(self)
        self.ui = Ui_Loading()
        self.ui.setupUi(self)

        self.setWindowFlag(QtCore.Qt.FramelessWindowHint)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)

        self.shadow = QGraphicsDropShadowEffect(self)
        self.shadow.setBlurRadius(20)
        self.shadow.setXOffset(0)
        self.shadow.setYOffset(0)
        self.shadow.setColor(QColor(0, 0, 0, 60))
        self.ui.shadow.setGraphicsEffect(self.shadow)

        # QTIMER ==> START
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.progress)
        # TIMER IN MILLISECONDS
        self.timer.start(35)

        # CHANGE DESCRIPTION

        self.ui.label_desc.setText("A step to prepare image \nfor next process")
        QtCore.QTimer.singleShot(2000, lambda: self.ui.label_desc.setText("Preparing..."))
        QtCore.QTimer.singleShot(3500, lambda: self.ui.label_desc.setText("Nah beres, here you go!"))

        # SHOW ==> MAIN WINDOW
        ########################################################################
        self.show()
        ## ==> END ##

    # ==> APP FUNCTIONS
    ########################################################################
    def progress(self):

        global counter

        # SET VALUE TO PROGRESS BAR
        self.ui.progressBar.setValue(counter)

        # CLOSE SPLASH SCREE AND OPEN APP
        if counter > 100:
            # STOP TIMER
            self.timer.stop()

            # SHOW MAIN WINDOW
            self.main = Preprocessing()
            self.main.show()
            # self.main.show() del later

            # CLOSE SPLASH SCREEN
            self.close()

        # INCREASE COUNTER
        counter += 1






class Loading2(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        self.ui = Ui_Loading2()
        self.ui.setupUi(self)

        self.setWindowFlag(QtCore.Qt.FramelessWindowHint)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)

        self.shadow = QGraphicsDropShadowEffect(self)
        self.shadow.setBlurRadius(20)
        self.shadow.setXOffset(0)
        self.shadow.setYOffset(0)
        self.shadow.setColor(QColor(0, 0, 0, 60))
        self.ui.shadow.setGraphicsEffect(self.shadow)

        # QTIMER ==> START
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.progress)
        # TIMER IN MILLISECONDS
        self.timer.start(35)

        # CHANGE DESCRIPTION

        self.ui.label_desc.setText("Mendeteksi retakan pada dinding bangunan")
        QtCore.QTimer.singleShot(2000, lambda: self.ui.label_desc.setText("Preparing..."))
        QtCore.QTimer.singleShot(3500, lambda: self.ui.label_desc.setText("Nah beres, here you go!"))

        # SHOW ==> MAIN WINDOW
        ########################################################################
        self.show()
        ## ==> END ##

    # ==> APP FUNCTIONS
    ########################################################################
    def progress(self):

        global counter

        # SET VALUE TO PROGRESS BAR
        self.ui.progressBar.setValue(counter)

        # CLOSE SPLASH SCREE AND OPEN APP
        if counter > 100:
            # STOP TIMER
            self.timer.stop()

            # SHOW MAIN WINDOW
            self.main = MainWindow()
            self.main.show()
            # self.main.show() del later

            # CLOSE SPLASH SCREEN
            self.close()

        # INCREASE COUNTER
        counter += 1




if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = Home()
    sys.exit(app.exec_())
window.show()
