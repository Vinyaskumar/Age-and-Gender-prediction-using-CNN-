import streamlit as st
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from PIL import Image
import tempfile
import os

# Load the face detection Haar cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the model
model = load_model('model_k.h5')

def detect_and_crop_faces(image_path):
    # Load the image
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # If faces are detected, crop and save the first one
    if len(faces) > 0:
        x, y, w, h = faces[0]
        face_img = img[y:y + h, x:x + w]

        # Save the cropped face to a temporary file
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, "cropped_face.jpg")
        cv2.imwrite(temp_path, face_img)

        return temp_path
    else:
        return None

def predict_age_gender(image_path):
    img = load_img(image_path, target_size=(128, 128), grayscale=True)
    img_array = img_to_array(img)
    img_array = img_array / 255.0

    gender_dict = {0: 'Male', 1: 'Female'}

    pred = model.predict(img_array.reshape(1, 128, 128, 1))
    pred_gender = gender_dict[round(pred[0][0][0])]
    pred_age = round(pred[1][0][0])

    return pred_gender, pred_age

# Streamlit App
def main():
    st.title("Age and Gender Prediction App")

    # Browse Button to Select Image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the selected image
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

        # Predict Button
        if st.button("Predict"):
            # Save the uploaded file to a temporary location
            temp_dir = tempfile.gettempdir()
            temp_path = os.path.join(temp_dir, "uploaded_image.jpg")
            uploaded_file.seek(0)  # Ensure the file is read from the beginning
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.read())

            # Perform face detection and crop faces
            cropped_face_path = detect_and_crop_faces(temp_path)

            # If faces are detected, proceed with prediction
            if cropped_face_path:
                #st.image(cropped_face_path, caption="Cropped Face", use_column_width=True)

                # Perform Age and Gender Prediction
                pred_gender, pred_age = predict_age_gender(cropped_face_path)

                # Display Prediction Results
                st.success(f"Predicted Gender: {pred_gender}, Predicted Age: {pred_age}")
            else:
                st.warning("No faces detected in the image.")

if __name__ == "__main__":
    main()
