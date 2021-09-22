# Facial Recognition Project
The projects in this directory involve the use of deep learning for the following:
```buildoutcfg
- Detecting faces in images
- Identification of facial landmarks eg. nose, eye, lips etc
- Facial recognition system for identifying a face from face encodings
- Facial recognition system for applying makeup on facial landmarks
- Facial recognition system for predicting the lookalike of an image from a database of collection of images
```

## Requirements
1. Install <i class="icon-cog"></i> **[dlib](http://dlib.net)**
2. Install <i class="icon-cog"></i> **[pillow](https://pillow.readthedocs.io/en/stable/installation.html)**
3. Install <i class="icon-cog"></i> **[face_recognition](https://face-recognition.readthedocs.io/en/latest/installation.html)**:
a pretrained HOG detector on faces


## Face Detection
The facial detection script (</i> **[`detect_faces.py`](https://github.com/emmanuelaboah/Machine-Learning-and-Data-Science/tree/master/facial%20recognition/face_detection)**)
takes an image as input and detects all the faces in the image by drawing
a bounding box around each face.

An example of the output from the facial detection script is below:

![alt text](https://github.com/emmanuelaboah/Machine-Learning-and-Data-Science/blob/master/facial%20recognition/face_detection/output/detected_diverse_faces1.jpg)


## Facial Feature Detection
