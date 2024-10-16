# -*- coding: utf-8 -*-

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QMainWindow, QApplication, QPushButton, QLabel, QFileDialog
from PyQt5.QtGui import QImage,QPixmap, QColor
import sys
import cv2

from PyQt5.QtCore import Qt
import numpy as np
import tensorflow as tf

import pytesseract
from PIL import Image
import os

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1300, 950)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
#---------------------------------------------------
        self.background_label = QtWidgets.QLabel(self.centralwidget)
        self.background_label.setGeometry(QtCore.QRect(0, 0, 1300, 950))  # Set the size to match the window
        self.background_label.setObjectName("background_label")
        self.background_label.setPixmap(QtGui.QPixmap('D:/bg4.jpg')) 
        self.background_label.setScaledContents(True)

        
        self.frame = QtWidgets.QFrame(self.centralwidget)
        self.frame.setGeometry(QtCore.QRect(0, 0, 600, 400))
        self.frame.setObjectName("frame")

        MainWindow.setCentralWidget(self.centralwidget)

       

        self.pushButton_13 = QtWidgets.QPushButton(MainWindow)
        self.pushButton_13.setGeometry(QtCore.QRect(850, 750, 200, 50)) 

        self.pushButton_13.setObjectName("pushButton_13")
        self.pushButton_13.setText("extract")

        self.pushButton_12 = QtWidgets.QPushButton(MainWindow)
        self.pushButton_12.setGeometry(QtCore.QRect(100, 850, 200, 50)) 
        self.pushButton_12.setObjectName("pushButton_12")
        self.pushButton_12.setText("impulse")

        self.pushButton_11 = QtWidgets.QPushButton(MainWindow)
        self.pushButton_11.setGeometry(QtCore.QRect(600, 800, 200, 50)) 
        self.pushButton_11.setObjectName("pushButton_11")
        self.pushButton_11.setText("edge")

        self.pushButton_10 = QtWidgets.QPushButton(MainWindow)
        self.pushButton_10.setGeometry(QtCore.QRect(850, 850, 200, 50)) 
        self.pushButton_10.setObjectName("pushButton_10")
        self.pushButton_10.setText("seg-knn")

        self.pushButton_9 = QtWidgets.QPushButton(MainWindow)
        self.pushButton_9.setGeometry(QtCore.QRect(850, 800, 200, 50)) 
        self.pushButton_9.setObjectName("pushButton_8")
        self.pushButton_9.setText("threshold")

        self.pushButton_8 = QtWidgets.QPushButton(MainWindow)
        self.pushButton_8.setGeometry(QtCore.QRect(1080, 800, 200, 100)) 
        self.pushButton_8.setObjectName("pushButton_8")
        self.pushButton_8.setText("save")

        self.pushButton_7 = QtWidgets.QPushButton(MainWindow)
        self.pushButton_7.setGeometry(QtCore.QRect(100, 800, 200, 50)) 
        self.pushButton_7.setObjectName("pushButton_7")
        self.pushButton_7.setText("gauss noise")

        self.pushButton_6 = QtWidgets.QPushButton(MainWindow)
        self.pushButton_6.setGeometry(QtCore.QRect(600, 750, 200, 50)) 
        self.pushButton_6.setObjectName("pushButton_6")
        self.pushButton_6.setText("stretching")

        self.pushButton_5 = QtWidgets.QPushButton(MainWindow)
        self.pushButton_5.setGeometry(QtCore.QRect(600, 850, 200, 50)) 
        self.pushButton_5.setObjectName("pushButton_5")
        self.pushButton_5.setText("eqialization")

         
        self.pushButton_4 = QtWidgets.QPushButton(MainWindow)
        self.pushButton_4.setGeometry(QtCore.QRect(350, 750, 200, 50)) 
        self.pushButton_4.setObjectName("pushButton_4")
        self.pushButton_4.setText("gaussian Filter")
        

        self.pushButton_3 = QtWidgets.QPushButton(MainWindow)
        self.pushButton_3.setGeometry(QtCore.QRect(350, 800, 200, 50)) 
        self.pushButton_3.setObjectName("pushButton_3")
        self.pushButton_3.setText("Median colored")

        # Create median filter button and set its properties
        self.pushButton_2 = QtWidgets.QPushButton(MainWindow)
        self.pushButton_2.setGeometry(QtCore.QRect(350, 850, 200, 50)) 
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_2.setText("Median grey")

        self.pushButton = QtWidgets.QPushButton(MainWindow)
        self.pushButton.setGeometry(QtCore.QRect(100, 750, 200, 50))
        self.pushButton.setObjectName("pushButton")
        self.pushButton.setText("Open Image")
        self.pushButton.setStyleSheet("QPushButton { border-radius: 12px; }")


        # Label for displaying original image
        self.label = QtWidgets.QLabel(MainWindow)
        self.label.setGeometry(QtCore.QRect(420, 100, 500, 600))
        self.label.setText("")
        self.label.setObjectName("label")
        self.label.setStyleSheet("border: 2px solid #DFC4B9; background-color: white;")

       # Label for displaying filtered image
        self.label_2 = QtWidgets.QLabel(self.frame)
        self.label_2.setGeometry(QtCore.QRect(20, 20, 361, 221))  # Adjusted geometry
        self.label_2.setObjectName("label_2")
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)  # Align the label content to the center

        self.titleLabel = QtWidgets.QLabel(self.centralwidget)
        self.titleLabel.setGeometry(QtCore.QRect(420, 20, 500, 50))
        self.titleLabel.setObjectName("titleLabel")
        self.titleLabel.setText("Let's Enhance Our Images")
        self.titleLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.titleLabel.setStyleSheet("font-size: 24px; font-weight: bold; color: #4F97A2;")

        #----------------------------------
        self.setStyleSheet("""
            QPushButton {
               background-color: #76B2BC; /* Pastel faded orange */
                border: 2px solid #4F97A2; /* Pastel orange */
                border-radius: 12px;
                color: white; /* Pastel orange */
                text-align: center;
                text-decoration: none;
                font-size: 16px;
                margin: 4px 2px;
            }
            
            QPushButton:hover {
                background-color: #AECFD4; /* Pastel orange */
                color: white;
            }
        """)


        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Image Filter"))
        self.pushButton.setText(_translate("MainWindow", "Open Image"))

class UI(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(UI, self).__init__()

        self.setupUi(self)

        self.pushButton.clicked.connect(self.clicker)
        self.pushButton_2.clicked.connect(self.filter_image)
        self.pushButton_3.clicked.connect(self.apply_colored_median_filter)
        self.pushButton_4.clicked.connect(self.gaussian_filter)
        self.pushButton_5.clicked.connect(self.histogram_equalization)
        self.pushButton_6.clicked.connect(self.contrast_stretching)
        self.pushButton_7.clicked.connect(self.add_gaussian_noise_and_display)
        self.pushButton_8.clicked.connect(self.function_save)
        self.pushButton_9.clicked.connect(self.add_thresholding)
        self.pushButton_10.clicked.connect(self.add_kmeans_segmentation)
        self.pushButton_11.clicked.connect(self.add_edge_detection)
        self.pushButton_12.clicked.connect(self.add_impulse_noise_and_display)
        self.pushButton_13.clicked.connect(self.extract_text_from_image)














        self.a = None

    def clicker(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Open File", "", "Image Files (*.png *.jpg *.bmp);;All Files (*)")
        if fname:
            self.a = fname
            pixmap = QPixmap(fname)
            self.label.setPixmap(pixmap.scaled(self.label.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))
            self.label.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)
    
    def filter_image(self):
        if self.a:
            img = cv2.imread(self.a)
            grey = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            median_image = cv2.medianBlur(grey, 5)
            pixmap = QtGui.QPixmap.fromImage(QtGui.QImage(median_image.data, median_image.shape[1], median_image.shape[0],
                                                          median_image.strides[0], QtGui.QImage.Format_Grayscale8))
            pixmap = pixmap.scaled(self.label.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
            self.label.setPixmap(pixmap)

    def apply_colored_median_filter(self):
        if self.a:
            # Load the original colored image
            original_image = cv2.imread(self.a)

            # Apply median filter to each channel separately
            median_filtered_r = cv2.medianBlur(original_image[:, :, 0], 5)  # Apply median filter to Red channel
            median_filtered_g = cv2.medianBlur(original_image[:, :, 1], 5)  # Apply median filter to Green channel
            median_filtered_b = cv2.medianBlur(original_image[:, :, 2], 5)  # Apply median filter to Blue channel

            # Combine the filtered channels to form the filtered colored image
            median_filtered_image = cv2.merge((median_filtered_b, median_filtered_g, median_filtered_r))  # Merge in the order: B, G, R

            # Convert the filtered image to QPixmap
            pixmap = QtGui.QPixmap.fromImage(QtGui.QImage(median_filtered_image.data, median_filtered_image.shape[1], median_filtered_image.shape[0],
                                                      median_filtered_image.strides[0], QtGui.QImage.Format_RGB888))
            pixmap = pixmap.scaled(self.label.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
            self.label.setPixmap(pixmap)

    def gaussian_filter(self):
        if self.a:
            # Load the original colored image
             
            img = cv2.imread(self.a)

            # Apply Gaussian blur to each color channel separately
            gaussian_filtered_r = cv2.GaussianBlur(img[:, :, 0], (5, 5), 0)  # Apply Gaussian blur to Red channel
            gaussian_filtered_g = cv2.GaussianBlur(img[:, :, 1], (5, 5), 0)  # Apply Gaussian blur to Green channel
            gaussian_filtered_b = cv2.GaussianBlur(img[:, :, 2], (5, 5), 0)  # Apply Gaussian blur to Blue channel

            # Combine the filtered channels to form the filtered colored image
            gaussian_filtered_image = cv2.merge((gaussian_filtered_r, gaussian_filtered_g, gaussian_filtered_b))

            # Convert BGR to RGB color space
            gaussian_filtered_image_rgb = cv2.cvtColor(gaussian_filtered_image, cv2.COLOR_BGR2RGB)

            # Convert the filtered image to QPixmap
            height, width, channels = gaussian_filtered_image_rgb.shape
            bytes_per_line = channels * width
            q_img = QtGui.QImage(gaussian_filtered_image_rgb.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888)

            pixmap = QtGui.QPixmap.fromImage(q_img)
            pixmap = pixmap.scaled(self.label.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
            self.label.setPixmap(pixmap)

    def histogram_equalization(self):
        if self.a:
        # Load the original image
            image = cv2.imread(self.a)

      # Convert the image to YUV color space
            
# Convert the image to YUV color space
            image_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)

# Equalize the histogram of all channels (Y, U, V)
            for i in range(3):
                image_yuv[:,:,i] = cv2.equalizeHist(image_yuv[:,:,i])

# Convert the image back to BGR color space
            equalized_image = cv2.cvtColor(image_yuv, cv2.COLOR_YUV2BGR)

        

        # Convert the BGR image to RGB for compatibility with QImage
            equalized_image_resized_rgb = cv2.cvtColor(equalized_image, cv2.COLOR_BGR2RGB)

        # Convert numpy array to QImage
            height, width, channel = equalized_image_resized_rgb.shape
            bytesPerLine = channel*width
            qimage_equalized = QImage(equalized_image_resized_rgb.data, width, height, bytesPerLine, QImage.Format_RGB888)
        #-----------------------
            scaled_image = qimage_equalized.scaled(500, 600, Qt.KeepAspectRatio)

        # Convert QImage to QPixmap
            pixmap = QPixmap.fromImage(scaled_image)

        # Display QPixmap in your label
            self.label.setPixmap(pixmap)
            


    
    def contrast_stretching(self):
        if self.a:
        # Load the original image
            img = cv2.imread(self.a)

        # Split the image into its color channels
            b, g, r = cv2.split(img)

        # Apply contrast stretching to each color channel separately
            stretched_b = cv2.equalizeHist(b)
            stretched_g = cv2.equalizeHist(g)
            stretched_r = cv2.equalizeHist(r)

        # Merge the stretched color channels back into a single image
            stretched_img = cv2.merge((stretched_r, stretched_g, stretched_b))

        # Convert the image to QPixmap and display it
            height, width, channels = stretched_img.shape
            bytes_per_line = channels * width
            q_img = QtGui.QImage(stretched_img.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888)
            pixmap = QtGui.QPixmap.fromImage(q_img)
            pixmap = pixmap.scaled(self.label.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
            self.label.setPixmap(pixmap)



    def add_gaussian_noise(self, image, mean=0, sigma=25):
        """
        Adds Gaussian noise to the loaded image.

        Parameters:
            image: numpy.ndarray
                The input image.
            mean: float, optional
                Mean of the Gaussian noise. Default is 0.
            sigma: float, optional
                Standard deviation of the Gaussian noise. Default is 25.

        Returns:
            numpy.ndarray
                The noisy image.
        """
        row, col, ch = image.shape
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        noisy_image = np.clip(image + gauss, 0, 255).astype(np.uint8)
        return noisy_image

    def display_image(self, image):
        """
        Display the image.

        Parameters:
            image: numpy.ndarray
                The image to display.
        """
        if self.label and image is not None:
            height, width, channels = image.shape
            bytes_per_line = channels * width
            q_img = QtGui.QImage(image.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888)
            pixmap = QtGui.QPixmap.fromImage(q_img)
            pixmap = pixmap.scaled(self.label.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
            self.label.setPixmap(pixmap)
        else:
            print("Error: QLabel widget not set or image is None.")

    def add_noise_and_display(self):
        """
        Load an image, add Gaussian noise, and display the noisy image.
        """
        if self.a:
            # Load the original image
            original_image = cv2.imread(self.a)

            # Check if image is loaded successfully
            if original_image is None:
                print("Error: Unable to load image.")
                return None

            # Add Gaussian noise to the loaded image
            noisy_image_gaussian = self.add_gaussian_noise(original_image, mean=0, sigma=25)

            # Display noisy image with Gaussian noise
            self.display_image(noisy_image_gaussian)
        else:
            print("Error: Image path is not provided.")

    
    def add_gaussian_noise_and_display(self, mean=0, sigma=25):
        if self.a:
            image = cv2.imread(self.a)
            row, col, ch = image.shape
            gauss = np.random.normal(mean, sigma, (row, col, ch))
            noisy_image = np.clip(image + gauss, 0, 255).astype(np.uint8)
            
          
            rgb_image = cv2.cvtColor(noisy_image, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            q_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)

        # Calculate the scaled size to fit the label
            scaled_image = q_image.scaled(500, 600, Qt.KeepAspectRatio)

        # Convert QImage to QPixmap
            pixmap = QPixmap.fromImage(scaled_image)

        # Display QPixmap in your label
            self.label.setPixmap(pixmap)
        # Convert the noisy image to QImage
            
    
    def add_impulse_noise_and_display(self, amount=0.04):
        if self.a:
            image = cv2.imread(self.a)
            row, col, ch = image.shape
            s_vs_p = 0.5
            noisy_image = np.copy(image)

    # Salt mode
            num_salt = np.ceil(amount * image.size * s_vs_p)
            coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
            noisy_image[coords] = 255

    # Pepper mode
            num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
            coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
            noisy_image[coords] = 0

            rgb_image = cv2.cvtColor(noisy_image, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            q_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)

        # Calculate the scaled size to fit the label
        #-----------
            scaled_image = q_image.scaled(500, 600, Qt.KeepAspectRatio)

        # Convert QImage to QPixmap
            pixmap = QPixmap.fromImage(scaled_image)

        # Display QPixmap in your label
            self.label.setPixmap(pixmap)

        # Convert the noisy image to QImage
            
            
    
    def function_save(self):
    # Get the currently displayed pixmap from the UI label
        pixmap = self.label.pixmap()

        if pixmap.isNull():
            print("Error: No image displayed.")
            return

    # Convert QPixmap to numpy array
        image = pixmap.toImage()
        image = image.convertToFormat(QImage.Format_RGB888)
        width = image.width()
        height = image.height()

    # Create a buffer to hold the image data
        buffer = np.zeros((height, width, 3), dtype=np.uint8)

    # Copy image data from the pixmap to the buffer
        for y in range(height):
            for x in range(width):
                pixel_value = image.pixel(x, y)
                rgb_value = QColor(pixel_value).getRgb()[:3]
                buffer[y, x] = rgb_value


    # Save the image to the specified output path
        output_path = 'D:/processed_image.jpg'  # Specify full path including drive letter
        cv2.imwrite(output_path, cv2.cvtColor(buffer, cv2.COLOR_RGB2BGR))
    


    def add_thresholding(self):
        if self.a:
        # Load the original image
            img = cv2.imread(self.a)

            if len(img.shape) == 2:  # Grayscale image
            # Calculate histogram
                hist = cv2.calcHist([img], [0], None, [256], [0, 256])

            # Find optimal threshold value using Otsu's method
                _, threshold = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Apply thresholding
                _, thresholded_img = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)

            else:  # Colored image
            # Set default threshold value
                threshold = 127

            # Apply thresholding to each color channel separately
                _, thresholded_r = cv2.threshold(img[:, :, 2], float(threshold), 255, cv2.THRESH_BINARY)
                _, thresholded_g = cv2.threshold(img[:, :, 1], float(threshold), 255, cv2.THRESH_BINARY)
                _, thresholded_b = cv2.threshold(img[:, :, 0], float(threshold), 255, cv2.THRESH_BINARY)

            # Merge the thresholded color channels back into a single image
                thresholded_img = cv2.merge((thresholded_b, thresholded_g, thresholded_r))

        # Convert the image to QPixmap and display it
            height, width, channels = thresholded_img.shape
            bytes_per_line = channels * width
            q_img = QtGui.QImage(thresholded_img.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888)
            pixmap = QtGui.QPixmap.fromImage(q_img)
            pixmap = pixmap.scaled(self.label.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
            self.label.setPixmap(pixmap)
    


    def add_kmeans_segmentation(self):
        if self.a:
        # Load the original image
            img = cv2.imread(self.a)

            
            # Reshape the image to 2D array of pixels (rows x cols, channels)
            pixel_values = img.reshape((-1, 3))

            # Convert to float32
            pixel_values = np.float32(pixel_values)

            # Define criteria and apply kmeans()
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
            k = 3  # You can adjust the number of clusters (k) as needed
            _, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

            # Convert back to 8 bit values
            centers = np.uint8(centers)

            # Map the labels to their respective centers
            segmented_image = centers[labels.flatten()]

            # Reshape back to the original image shape
            segmented_image = segmented_image.reshape(img.shape)

            # Convert the image to QPixmap and display it
            height, width, channels = segmented_image.shape
            bytes_per_line = channels * width
            q_img = QtGui.QImage(segmented_image.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888)
            pixmap = QtGui.QPixmap.fromImage(q_img)
            pixmap = pixmap.scaled(self.label.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
            self.label.setPixmap(pixmap)
    

    def add_edge_detection(self):
        if self.a:
        # Load the original image
            img = cv2.imread(self.a)

            if len(img.shape) == 3:  # Colored image
                img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:  # Already grayscale
                img_gray = img.copy()

        # Perform edge detection
            edges = cv2.Canny(img_gray, 100, 200)  # You can adjust the thresholds as needed
        
            kernel_size = (3, 3)  # Adjust the kernel size as needed
            iterations = 1  # Adjust the number of iterations as needed
            kernel = np.ones(kernel_size, np.uint8)
            edges = cv2.dilate(edges, kernel, iterations=iterations)
        # Convert the image to QPixmap and display it
            
            height, width = edges.shape
            bytes_per_line = width
            q_img = QtGui.QImage(edges.data, width, height, bytes_per_line, QtGui.QImage.Format_Grayscale8)
            pixmap = QtGui.QPixmap.fromImage(q_img)
            pixmap = pixmap.scaled(self.label.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
            self.label.setPixmap(pixmap)

    def extract_text_from_image(self):
        if self.a:
            # Open the image using OpenCV
            img = cv2.imread(self.a)

            # Convert the image to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Define the temporary image file path
            temp_img_path = 'temp_img.jpg'

            # Save the grayscale image to the temporary file
            cv2.imwrite(temp_img_path, gray)

            # Perform OCR to extract text from the temporary image file
            text = pytesseract.image_to_string(Image.open(temp_img_path), config=r"--psm 11 --oem 3")
            
             # Save the extracted text into a text file
            text_file_path = 'extracted_text.txt'
            with open(text_file_path, 'w') as text_file:
                text_file.write(text)

            # Display a message indicating that text has been extracted and saved
            QtWidgets.QMessageBox.information(self, "Text Extracted", f"Text has been extracted and saved to {text_file_path}")

            # Optionally, you can delete the temporary image file after OCR
            os.remove(temp_img_path)
       
            # Display the extracted text
            print(text)

            # Optionally, you can delete the temporary image file after OCR
        else:
            print("Error: No image loaded.")

   

if __name__ == "__main__":
    app = QApplication(sys.argv)
    ui = UI()
    ui.show()
    sys.exit(app.exec_())
