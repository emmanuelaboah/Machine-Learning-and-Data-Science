# Facial recognition system for identifying a face from face encodings
# The detection system is based on images (faces) of known people, and it
# tries to identify whether an unknown image (face) is found in the database
# of known images or not.

# Import relevant libraries
import PIL.Image  # python image library
import PIL.ImageDraw
import face_recognition
import os


# load image file for detection
def load_image(name, path="images"):
    """
        loads image file into numpy array for detection
    """
    image_path = os.path.join(path, name)
    image = face_recognition.load_image_file(image_path)

    return image


# Save output images1
def save_image(image, name, path="output"):
    """" Use PIL for saving output images"""

    output_path = os.path.join(path, name)
    return image.save(output_path)


# Change image format into Python image Library object for rendering
def img_to_pil(image):
    """ Load image into python image Library """

    return PIL.Image.fromarray(image)


# get the facial encodings in an image of known persons
def facial_encodings_known(known_image):
    """ extract face encoding of a known person in an image """

    known_face_encodings = face_recognition.face_encodings(known_image)[0]
    return known_face_encodings


# get the facial encodings in an image of unknown persons
def facial_encodings_unknown(unknown_image, upsample=2):
    """ extract face encoding of an unknown person in an image """

    # get face locations of the people in the image
    face_locations = face_recognition.face_locations(unknown_image,
                                                     number_of_times_to_upsample=upsample)

    unknown_face_encode = face_recognition.face_encodings(unknown_image, known_face_locations=face_locations)
    return unknown_face_encode


# _______manin__________
# Load all known images
# This could be any number of images
img1 = load_image("image_1.jpg")
img2 = load_image("image_2.jpg")
img3 = load_image("image_3.jpg")

# Extract the face encodings of the known faces
img1_face_encodings = facial_encodings_known(img1)
img2_face_encodings = facial_encodings_known(img2)
img3_face_encodings = facial_encodings_known(img3)

# List of known faces in database
known_face_encoding_list = [img1_face_encodings, img2_face_encodings, img3_face_encodings]

# Load unknown image to be verified or recognized
unknown_img = load_image("unknown_4.jpg")

# Extract face encodings of unknown person(s) in the image
unknown_face_encodings = facial_encodings_unknown(unknown_img)

# Loop through the faces in the photo and check for matches
for unknown_face_encoding in unknown_face_encodings:

    # Check to see if there is a match between the unknown face encoding and
    # the known face encodings in our database
    outputs = face_recognition.compare_faces(known_face_encoding_list, unknown_face_encoding, tolerance=0.6)

    person = "Unknown"

    if outputs[0]:
        person = "Person 1"

    elif outputs[1]:
        person = "Person 2"

    elif outputs[2]:
        person = "Person 3"

    # Print the name of the person
    print("Recognizes {} in the image".format(person))
