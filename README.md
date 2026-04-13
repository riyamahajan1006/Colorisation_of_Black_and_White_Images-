# ChromaRevive — Black & White Image Colorization

> A deep learning web app that breathes color back into grayscale photographs — instantly, in the browser.

---

## Overview

**ChromaRevive** is a Streamlit web application that uses a pretrained convolutional neural network to automatically colorize black and white images. The model predicts realistic, plausible colors with no human input — powered by the landmark paper:

> **"Colorful Image Colorization"** — Richard Zhang, Phillip Isola, Alexei A. Efros (ECCV 2016)  
> [Paper](https://arxiv.org/abs/1603.08511) · [Project Page](http://richzhang.github.io/colorization/)

---

## Features

- Drag-and-drop image upload (JPG, JPEG, PNG)
- Side-by-side before/after comparison
- One-click download of the colorized result
- Image metadata display — resolution, model, color space, bin count
- Custom dark cinematic UI with no default Streamlit chrome
- Model loaded once and cached across sessions via `@st.cache_resource`
- Graceful error handling if model files are missing

---

## Model & Architecture

| Component | Details |
|---|---|
| **Model** | Zhang et al. CNN — `colorization_release_v2.caffemodel` |
| **Framework** | OpenCV DNN module (Caffe backend) |
| **Color Space** | L\*a\*b (CIE Lab) |
| **Input to model** | Grayscale L channel, resized to 224x224 |
| **Output** | Predicted *a* and *b* channels |
| **Color bins** | 313 quantized *ab* cluster centers (`pts_in_hull.npy`) |

### Colorization Pipeline

```
B&W Image (RGB/grayscale)
        |
        v
  Convert to L*a*b color space
        |
        v
  Extract L channel -> resize to 224x224
        |
        v
  Feed into pretrained CNN
        |
        v
  Predict ab channels (313 color bins)
        |
        v
  Resize ab -> merge with original L
        |
        v
  Convert L*a*b -> RGB -> display & download
```

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.x | Core language |
| Streamlit | Web app framework |
| OpenCV (`cv2`) | Image I/O, color conversion, DNN inference |
| NumPy | Array and tensor operations |
| Pillow (PIL) | Image loading and in-memory export |

---

## Project Structure

```
Image_colorisation/
|
|-- model/
|   |-- colorization_deploy_v2.prototxt     # Network architecture definition
|   |-- colorization_release_v2.caffemodel  # Pretrained weights (~125 MB)
|   |-- pts_in_hull.npy                     # 313 quantized ab color cluster centers
|
|-- app.py          # Streamlit web application
|-- colorise.py     # CLI colorization script
|-- README.md
```

---

## Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/riyamahajan1006/Colorisation_of_Black_and_White_Images-.git
cd Colorisation_of_Black_and_White_Images-
```

### 2. Install dependencies

```bash
pip install streamlit opencv-python numpy 
```

### 3. Download model files

Place all three files inside the `model/` folder:

| File | Link |
|---|---|
| `colorization_release_v2.caffemodel` | [Download (~125 MB)](https://www.dropbox.com/s/dx0qvhhp5hbcx7z/colorization_release_v2.caffemodel) |
| `colorization_deploy_v2.prototxt` | Included in repo |
| `pts_in_hull.npy` | Included in repo |

### 4. Run the app

```bash
streamlit run app.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

---

## How to Use

1. Run the app with `streamlit run app.py`
2. Drop a black & white image into the upload zone (JPG, JPEG, or PNG)
3. The colorized result appears instantly alongside the original
4. Click **Download Colorized Image** to save the output

---

## Key Concepts

**L\*a\*b Color Space** — Lab separates lightness (L) from color (ab). Since grayscale images already encode L, the model only needs to predict the ab channels, making the problem well-defined.

**313 Quantized Color Bins** — Instead of regressing exact colors (which produces washed-out results), the model predicts a probability distribution over 313 ab bins and takes the weighted mean. This produces vivid, confident colorizations.

**Class Rebalancing** — Rare but vibrant colors (reds, yellows) are up-weighted during training so the model does not always default to desaturated grays and browns.

---

## Future Improvements

- [ ] Deploy on Streamlit Cloud / Hugging Face Spaces for public access
- [ ] Add batch upload support for multiple images
- [ ] Integrate GAN-based colorization (Pix2Pix) for richer results
- [ ] Add a slider for interactive before/after comparison
- [ ] Support video colorization frame by frame

---

## Author

**Riya Mahajan**  
[GitHub](https://github.com/riyamahajan1006) 