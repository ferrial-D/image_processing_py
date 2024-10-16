# Image Processing Application
This application allows you to load, display, and apply various image processing techniques to images using a graphical user interface (GUI) built with PyQt5 and OpenCV.

## Features
Load Image: Load and display an image from your file system.
## Image Filters:
- Median Filter (Grayscale)
- Colored Median Filter
- Gaussian Filter
- Histogram Equalization: Improve contrast by equalizing the histogram in YUV color space.
- Contrast Stretching: Apply contrast stretching on each channel to improve the image's dynamic range.
- Add Noise:
- Add Gaussian noise to an image.
- Add Impulse (Salt-and-Pepper) noise.
- Thresholding: Apply binary thresholding using Otsu's method for grayscale images and manual thresholding for colored images.
- K-means Segmentation: Perform color quantization using K-means clustering.
- Edge Detection: Detect edges using the Canny edge detection algorithm, followed by dilation.
- Text Extraction: Extract text from an image using Optical Character Recognition (OCR) with Tesseract.
## Requirements
Python 3
PyQt5
OpenCV (cv2)
NumPy
Tesseract-OCR
pytesseract (Python wrapper for Tesseract)
PIL (for image processing related to Tesseract)