# This script identifies facial landmarks of an image
# eg: Nose, eye, lips, mouth etc.
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


# Function for detecting all facial features in the image
def detect_facial_landmarks(image):
    """ finds all facial landmarks in the image"""
    facial_landmarks_list = face_recognition.face_landmarks(image)
    print("The image or photograph has {} face(s)".format(len(facial_landmarks_list)))

    return facial_landmarks_list


# Load the image or photograph and change the format to PIL
image_ = load_image("diverse_people3.jpg")
image_to_pil = img_to_pil(image_)

# Get the facial landmarks
facial_features = detect_facial_landmarks(image_)

# Loop through all the faces in the image
for facial_feature in facial_features:

    # Extract facial landmarks (such as nose, mouth, eye, etc.) from each face in the image
    for feature, list_of_points in facial_feature.items():
        # Print the locations of the each facial feature
        # e.g. The location of the nose or eye
        print("The points of the location of the {} in this face are {}".format(feature, list_of_points))

        # Trace the facial features in the image
        # Create a PIL drawing object for drawing lines
        face_draw = PIL.ImageDraw.Draw(image_to_pil)
        face_draw.line(list_of_points, fill="blue", width=2)

#  Display the image on the screen
image_to_pil.show()

#  Save image to output file
save_image(image_to_pil, "facial_feat_diverse_faces3.jpg")
