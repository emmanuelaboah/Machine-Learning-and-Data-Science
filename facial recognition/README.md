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
The facial feature detection (</i> **[`facial_feature_detection.py`](https://github.com/emmanuelaboah/Machine-Learning-and-Data-Science/tree/master/facial%20recognition/facial_feature_detection)**)
identifies key facial features or landmarks of an image.\
It is able to identify facial features such as the chin, eye, eyebrow, nose
lips etc.\
It takes an image as an input.

A typical output of detected facial landmarks of people in an image is 
shown in below:\
![alt text](https://github.com/emmanuelaboah/Machine-Learning-and-Data-Science/blob/master/facial%20recognition/facial_feature_detection/output/facial_feat_diverse_faces1.jpg)

## Facial Recognition System
The facial recognition system encodes images (faces) of known people
and then tries to identify whether an unknown image or face is found in
the database of known image encodings or not.

It takes an unknown image as input checks to see if there is a match between
the unknown face encoding and the known face encoding in the database.
It prints the name of the image (person) if there is a match, otherwise 
it prints out "unknown".


## Digital Makeup
The digital makeup script (</i> **[`digital_makeup.py`](https://github.com/emmanuelaboah/Machine-Learning-and-Data-Science/tree/master/facial%20recognition/facial_recognition_system)**)
applies a digital makeup of choice to key facial landmarks such as
lips, eyebrow etc. It takes an image of faces as input and loops through
the faces in the image to apply a digital makeup on facial landmark of choice.

Example of the output of a digital makeup on faces is shown below:

![alt text](https://github.com/emmanuelaboah/Machine-Learning-and-Data-Science/blob/master/facial%20recognition/facial_recognition_system/output/face_makeup1.jpg)


## Finding Lookalikes


