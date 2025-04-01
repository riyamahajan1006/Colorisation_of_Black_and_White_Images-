import streamlit as st
import cv2
import numpy as np
import os
from PIL import Image

# Set up model paths
DIR = os.path.dirname(os.path.abspath(__file__))
PROTOTXT = os.path.join(DIR, "model", "colorization_deploy_v2.prototxt")
POINTS = os.path.join(DIR, "model", "pts_in_hull.npy")
MODEL = os.path.join(DIR, "model", "colorization_release_v2.caffemodel")

# Load the model
net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)
pts = np.load(POINTS)
class8 = net.getLayerId("class8_ab")
conv8 = net.getLayerId("conv8_313_rh")
pts = pts.transpose().reshape(2, 313, 1, 1)
net.getLayer(class8).blobs = [pts.astype("float32")]
net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

# Streamlit app title
st.title("Image Colorization")

# Image upload
uploaded_file = st.file_uploader("Choose a black and white image...", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    # Read and process the image
    image = Image.open(uploaded_file)
    image = np.array(image)
    scaled = image.astype("float32") / 255.0
    lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)
    resized = cv2.resize(lab, (224, 224))
    L = cv2.split(resized)[0]
    L -= 50

    # Colorize the image
    net.setInput(cv2.dnn.blobFromImage(L))
    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))
    ab = cv2.resize(ab, (image.shape[1], image.shape[0]))
    L = cv2.split(lab)[0]
    colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)
    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
    colorized = np.clip(colorized, 0, 1)
    colorized = (255 * colorized).astype("uint8")

    # Display images
    st.image(image, caption="Original Image", use_container_width=True)
    st.image(colorized, caption="Colorized Image", use_container_width=True)
