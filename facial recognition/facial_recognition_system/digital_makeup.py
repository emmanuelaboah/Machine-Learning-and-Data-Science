# Facial recognition system for applying digital makeup to
# key facial landmarks of the face such as lips, eyebrow, nose etc.

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


# Function for detecting all facial features in the image
def detect_facial_landmarks(image):
    """ finds all facial landmarks in the image"""
    facial_landmarks_list = face_recognition.face_landmarks(image)
    print("The image or photograph has {} face(s)".format(len(facial_landmarks_list)))

    return facial_landmarks_list


# Load the image or photograph and change the format to PIL
image_ = load_image("image_4.jpg")
image_to_pil = img_to_pil(image_)

# Get the facial landmarks
facial_features = detect_facial_landmarks(image_)

# Loop through all the facial features in the faces
for facial_feature in facial_features:
    '''
    The following facial features are returned by the face landmark detection model:
    1. chin
    2. left and right eyebrows
    3. nose bridge
    4. nose tip
    5. left and right eye
    '''

    # Create a PIL drawing object for drawing lines
    face_draw = PIL.ImageDraw.Draw(image_to_pil, "RGBA")

    # Makeup over the lips using the polygon method of the python image library
    face_draw.polygon(facial_feature["top_lip"], fill=(128, 0, 128, 100))
    face_draw.polygon(facial_feature["bottom_lip"], fill=(128, 0, 128, 100))

    # Makeup over the eyebrows using the line method of the python image library
    face_draw.line(facial_feature["left_eyebrow"], fill=(128, 0, 128, 100), width=3)
    face_draw.line(facial_feature["right_eyebrow"], fill=(128, 0, 128, 100), width=3)


# Display the image on the screen
image_to_pil.show()

# Save the digital makeup to output file
save_image(image_to_pil, "face_makeup3.jpg")





