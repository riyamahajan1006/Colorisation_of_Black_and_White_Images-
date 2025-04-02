import streamlit as st
import cv2
import numpy as np
import os
from PIL import Image

# Hide Streamlit default menu and footer
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Custom CSS for additional styling
custom_css = """
<style>
body {
    background-color: #f0f2f6;
}
h1 {
    color: #4a4a4a;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}
.stButton>button {
    background-color: #4CAF50;
    color: white;
    border: none;
    border-radius: 5px;
    padding: 10px 24px;
    cursor: pointer;
}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# App Title and Description
st.title("Beautiful Image Colorization")
st.write("Upload a black and white image and watch it come to life with vibrant colors!")

# Sidebar with app info
st.sidebar.header("About This App")
st.sidebar.info(
    "This app uses a pre-trained deep learning model to colorize black and white images. "
    "Simply upload your image, and let the magic happen!"
)

# Set up model paths
DIR = os.path.dirname(os.path.abspath(__file__))
PROTOTXT = os.path.join(DIR, "model", "colorization_deploy_v2.prototxt")
POINTS = os.path.join(DIR, "model", "pts_in_hull.npy")
MODEL = os.path.join(DIR, "model", "colorization_release_v2.caffemodel")

# Use st.cache_resource to cache the model loading
@st.cache_resource
def load_model():
    net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)
    pts = np.load(POINTS)
    class8 = net.getLayerId("class8_ab")
    conv8 = net.getLayerId("conv8_313_rh")
    pts = pts.transpose().reshape(2, 313, 1, 1)
    net.getLayer(class8).blobs = [pts.astype("float32")]
    net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]
    return net

net = load_model()

# File uploader widget
uploaded_file = st.file_uploader("Choose a black and white image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open the image and display the original
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)
    st.image(image_np, caption="Original Image", use_container_width=True)
    
    with st.spinner("Colorizing..."):
        # Preprocess the image for the model
        scaled = image_np.astype("float32") / 255.0
        lab = cv2.cvtColor(scaled, cv2.COLOR_RGB2LAB)
        resized = cv2.resize(lab, (224, 224))
        L = cv2.split(resized)[0]
        L -= 50

        net.setInput(cv2.dnn.blobFromImage(L))
        ab = net.forward()[0, :, :, :].transpose((1, 2, 0))
        ab = cv2.resize(ab, (image_np.shape[1], image_np.shape[0]))
        
        # Combine with original L channel and convert back to RGB
        L_channel = cv2.split(lab)[0]
        colorized = np.concatenate((L_channel[:, :, np.newaxis], ab), axis=2)
        colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2RGB)
        colorized = np.clip(colorized, 0, 1)
        colorized = (255 * colorized).astype("uint8")
    
    # Display the colorized image
    st.image(colorized, caption="Colorized Image", use_container_width=True)
    st.success("Colorization complete!")
