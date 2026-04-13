import streamlit as st
import cv2
import numpy as np
import os
from PIL import Image
import io

st.set_page_config(
    page_title="ChromaRevive · Image Colorization",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,700;1,400&family=DM+Sans:wght@300;400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"] {
    background-color: #0c0c0f !important;
    color: #e8e0d4 !important;
    font-family: 'DM Sans', sans-serif !important;
}

[data-testid="stAppViewContainer"] {
    background-image:
        radial-gradient(ellipse 80% 60% at 50% -10%, rgba(180,140,90,0.12) 0%, transparent 70%),
        radial-gradient(ellipse 40% 40% at 90% 80%, rgba(120,80,40,0.08) 0%, transparent 60%);
}

#MainMenu, footer, header, [data-testid="stToolbar"] { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }
.block-container { padding: 2.5rem 3rem 4rem !important; max-width: 1100px !important; }

.hero {
    text-align: center;
    padding: 3.5rem 0 2rem;
    position: relative;
}
.hero-eyebrow {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.72rem;
    font-weight: 500;
    letter-spacing: 0.28em;
    text-transform: uppercase;
    color: #b8924a;
    margin-bottom: 1rem;
    text-align: center;
}
.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: clamp(2.8rem, 6vw, 4.5rem);
    font-weight: 700;
    color: #f0e6d3;
    line-height: 1.1;
    margin: 0 0 0.5rem;
    letter-spacing: -0.02em;
    text-align: center;
}
.hero-title em {
    font-style: italic;
    color: #c8a06a;
}
.hero-subtitle {
    font-family: 'DM Sans', sans-serif;
    font-size: 1.05rem;
    font-weight: 300;
    color: #8a7d6e;
    max-width: 480px;
    margin: 0.8rem auto 0;
    line-height: 1.6;
    text-align: center;
}
.hero-rule {
    width: 60px;
    height: 1px;
    background: linear-gradient(90deg, transparent, #b8924a, transparent);
    margin: 2rem auto 0;
}

[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.025) !important;
    border: 1.5px dashed rgba(184,146,74,0.35) !important;
    border-radius: 16px !important;
    padding: 2.5rem !important;
    transition: border-color 0.3s, background 0.3s !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(184,146,74,0.7) !important;
    background: rgba(184,146,74,0.04) !important;
}
[data-testid="stFileUploaderDropzone"] p {
    color: #8a7d6e !important;
    font-family: 'DM Sans', sans-serif !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] {
    color: #b8924a !important;
}

.section-label {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.68rem;
    font-weight: 500;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: #b8924a;
    margin-bottom: 0.6rem;
    display: block;
}

[data-testid="stSpinner"] {
    color: #b8924a !important;
}

.success-banner {
    background: linear-gradient(135deg, rgba(184,146,74,0.12), rgba(140,100,50,0.08));
    border: 1px solid rgba(184,146,74,0.3);
    border-radius: 12px;
    padding: 1rem 1.4rem;
    display: flex;
    align-items: center;
    gap: 0.8rem;
    margin: 1.5rem 0 0;
}
.success-banner span {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.9rem;
    color: #d4aa72;
}

.stDownloadButton > button {
    background: linear-gradient(135deg, #b8924a, #8a6830) !important;
    color: #0c0c0f !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.88rem !important;
    letter-spacing: 0.04em !important;
    padding: 0.6rem 1.6rem !important;
    transition: opacity 0.2s, transform 0.2s !important;
    width: 100% !important;
}
.stDownloadButton > button:hover {
    opacity: 0.88 !important;
    transform: translateY(-1px) !important;
}

.stat-row {
    display: flex;
    gap: 0.8rem;
    margin-top: 1.2rem;
    flex-wrap: wrap;
}
.stat-pill {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 8px;
    padding: 0.5rem 0.9rem;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.78rem;
    color: #8a7d6e;
}
.stat-pill strong {
    color: #c8a06a;
    display: block;
    font-size: 0.95rem;
}

.footer {
    text-align: center;
    margin-top: 5rem;
    padding-top: 2rem;
    border-top: 1px solid rgba(255,255,255,0.06);
    font-family: 'DM Sans', sans-serif;
    font-size: 0.78rem;
    color: #4a4540;
    letter-spacing: 0.05em;
}
.footer a { color: #7a6a50; text-decoration: none; }

[data-testid="caption"] {
    color: #5a5248 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.75rem !important;
    text-align: center !important;
}

.stProgress > div > div {
    background: linear-gradient(90deg, #b8924a, #e8c07a) !important;
}
</style>
""", unsafe_allow_html=True)

DIR      = os.path.dirname(os.path.abspath(__file__))
PROTOTXT = os.path.join(DIR, "model", "colorization_deploy_v2.prototxt")
POINTS   = os.path.join(DIR, "model", "pts_in_hull.npy")
MODEL    = os.path.join(DIR, "model", "colorization_release_v2.caffemodel")

@st.cache_resource(show_spinner=False)
def load_model():
    net    = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)
    pts    = np.load(POINTS)
    class8 = net.getLayerId("class8_ab")
    conv8  = net.getLayerId("conv8_313_rh")
    pts    = pts.transpose().reshape(2, 313, 1, 1)
    net.getLayer(class8).blobs = [pts.astype("float32")]
    net.getLayer(conv8).blobs  = [np.full([1, 313], 2.606, dtype="float32")]
    return net

def colorize(image_np, net):
    scaled  = image_np.astype("float32") / 255.0
    lab     = cv2.cvtColor(scaled, cv2.COLOR_RGB2LAB)
    resized = cv2.resize(lab, (224, 224))
    L       = cv2.split(resized)[0]
    L      -= 50
    net.setInput(cv2.dnn.blobFromImage(L))
    ab        = net.forward()[0, :, :, :].transpose((1, 2, 0))
    ab        = cv2.resize(ab, (image_np.shape[1], image_np.shape[0]))
    L_full    = cv2.split(lab)[0]
    colorized = np.concatenate((L_full[:, :, np.newaxis], ab), axis=2)
    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2RGB)
    colorized = np.clip(colorized, 0, 1)
    colorized = (255 * colorized).astype("uint8")
    return colorized

def pil_to_bytes(img_array):
    img = Image.fromarray(img_array)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

st.markdown("""
<div class="hero">
    <div class="hero-eyebrow">Deep Learning · Computer Vision</div>
    <h1 class="hero-title">Chroma<em>Revive</em></h1>
    <p class="hero-subtitle">Breathe color back into black &amp; white photographs using a pretrained neural network.</p>
    <div class="hero-rule"></div>
</div>
""", unsafe_allow_html=True)

model_ok = all(os.path.exists(p) for p in [PROTOTXT, POINTS, MODEL])

if not model_ok:
    st.error("Model files not found. Place `colorization_deploy_v2.prototxt`, `colorization_release_v2.caffemodel`, and `pts_in_hull.npy` inside a `model/` folder.")
    st.stop()

with st.spinner("Loading model weights..."):
    net = load_model()

uploaded = st.file_uploader(
    "Drop a black & white image here, or click to browse",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed",
)

if uploaded:
    image    = Image.open(uploaded).convert("RGB")
    image_np = np.array(image)
    h, w     = image_np.shape[:2]

    with st.spinner("Painting with color…"):
        colorized = colorize(image_np, net)

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown('<span class="section-label">Original</span>', unsafe_allow_html=True)
        st.image(image_np, use_container_width=True)

    with col2:
        st.markdown('<span class="section-label">Colorized</span>', unsafe_allow_html=True)
        st.image(colorized, use_container_width=True)

    st.markdown(f"""
    <div class="stat-row">
        <div class="stat-pill"><strong>{w} × {h}</strong>Resolution</div>
        <div class="stat-pill"><strong>Zhang et al. 2016</strong>Model</div>
        <div class="stat-pill"><strong>L*a*b</strong>Color Space</div>
        <div class="stat-pill"><strong>313</strong>Color Bins</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    dl_col, _ = st.columns([1, 2])
    with dl_col:
        st.download_button(
            label="⬇  Download Colorized Image",
            data=pil_to_bytes(colorized),
            file_name=f"{os.path.splitext(uploaded.name)[0]}_colorized.png",
            mime="image/png",
        )

    st.markdown("""
    <div class="success-banner">
        <span>✦</span>
        <span>Colorization complete — colors predicted using 313 quantized <em>ab</em> bins in L*a*b space.</span>
    </div>
    """, unsafe_allow_html=True)

else:
    st.markdown("""
    <div style="text-align:center; padding: 3rem 0; color: #3a3530;">
        <div style="font-size:2.5rem; margin-bottom:0.8rem;">◑</div>
        <div style="font-family:'DM Sans',sans-serif; font-size:0.9rem; letter-spacing:0.06em;">
            Upload an image above to begin
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div class="footer">
    Built by <a href="https://github.com/riyamahajan1006">Riya Mahajan</a> &nbsp;·&nbsp;
    Model: <a href="https://arxiv.org/abs/1603.08511">Zhang et al., ECCV 2016</a>
</div>
""", unsafe_allow_html=True)