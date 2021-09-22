# Facial recognition system for predicting your lookalike
# from a database of known images
# For example: this script can be applied to checking
# your celebrity lookalike from a database of celebrities :)

# Import relevant libraries
import PIL.Image  # python image library
import PIL.ImageDraw
from pathlib import Path
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

    face_encodings = face_recognition.face_encodings(known_image)[0]
    return face_encodings


# get the facial encodings in an image of unknown persons in the database
def facial_encodings_unknown(unknown_image, upsample=1):
    """ extract face encoding of an unknown person in an image """

    # get face locations of the people in the image
    face_locations = face_recognition.face_locations(unknown_image,
                                                     number_of_times_to_upsample=upsample)

    unknown_face_encode = face_recognition.face_encodings(unknown_image, known_face_locations=face_locations)
    return unknown_face_encode


# Load the image of the person you want to find similar people for
image_ = load_image("image_2.jpg")

# Encode the image of the person you want to find similar people for
image_encoding = facial_encodings_known(image_)

# Declare variables to keep track of the most similar face match so far
best_match = None
best_match_distance = 1


# Loop through the database of images you want to check for similar people
for im_path in Path("database").glob("*.png"):
    # Load each image to check for match
    unknown_im = face_recognition.load_image_file(im_path)

    # Extract the location of faces and the face encodings of each image
    database_face_encodings = facial_encodings_unknown(unknown_im)

    # Get the face encoding distance between the person you want to find
    # and all the faces in this image
    face_distance = face_recognition.face_distance(database_face_encodings, image_encoding)

    # Save the image to memory if it is the most similar face so far
    if face_distance < best_match_distance:
        best_match_distance = face_distance
        best_match = unknown_im


# Convert the image from array to PIL format and ...
# Display the best match (i.e. the most similar face image)
image_to_pil = img_to_pil(best_match)
print("Displaying the best match...")
image_to_pil.show()

# Save the best match to the output file
save_image(image_to_pil, "lookalike.jpg")
