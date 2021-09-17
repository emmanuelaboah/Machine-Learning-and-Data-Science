# Uses a pretrained HOG detector to detect all the faces in an image
# Follow the instructions in the README in this repository to install DLIB.
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
    """ Load image into python image Library"""
    return PIL.Image.fromarray(image)


# Extract all faces in the image
def detect_faces_locations(image):
    """ finds all the different faces in the image"""
    face_locations = face_recognition.face_locations(image)
    print("The image or photograph has {} face(s)".format(len(face_locations)))

    return face_locations


# Load the image or photograph and change the format to PIL
image_ = load_image("diverse_people1.jpg")
image_to_pil = img_to_pil(image_)

# Get the face locations
face_positions = detect_faces_locations(image_)


# Draw bounding boxes around detected faces and
# print the locations of the faces in the image
for face_loc in face_positions:

    # Extract the locations of each face:
    top, right, bottom, left = face_loc

    # Print the locations of the each face
    print("There is a face located at the pixel location top: {}, right: {}, "
          "bottom: {}, left: {}".format(top, right, bottom, left))

    # draw a bounding box around the face
    face_draw = PIL.ImageDraw.Draw(image_to_pil)
    face_draw.rectangle([left, top, right, bottom], outline="red")


#  Display the image on the screen
image_to_pil.show()

#  Save image to output file
save_image(image_to_pil, "detected_diverse_faces1.jpg")