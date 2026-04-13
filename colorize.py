import numpy as np
import argparse
import cv2
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROTOTXT = os.path.join(BASE_DIR, "model", "colorization_deploy_v2.prototxt")
POINTS   = os.path.join(BASE_DIR, "model", "pts_in_hull.npy")
MODEL    = os.path.join(BASE_DIR, "model", "colorization_release_v2.caffemodel")

ap = argparse.ArgumentParser(description="Colorize a black-and-white image using Zhang et al. (2016)")
ap.add_argument("-i", "--image",  type=str, required=True,  help="Path to input B&W image")
ap.add_argument("-o", "--output", type=str, default="",     help="Path to save colorized image (optional)")
args = vars(ap.parse_args())

for label, path in [("Prototxt", PROTOTXT), ("Model", MODEL), ("Points", POINTS)]:
    if not os.path.exists(path):
        print(f"[ERROR] {label} file not found: {path}")
        sys.exit(1)

if not os.path.exists(args["image"]):
    print(f"[ERROR] Input image not found: {args['image']}")
    sys.exit(1)

print("[INFO] Loading colorization model...")
net  = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)
pts  = np.load(POINTS)

class8 = net.getLayerId("class8_ab")
conv8  = net.getLayerId("conv8_313_rh")
pts    = pts.transpose().reshape(2, 313, 1, 1)
net.getLayer(class8).blobs = [pts.astype("float32")]
net.getLayer(conv8).blobs  = [np.full([1, 313], 2.606, dtype="float32")]

print("[INFO] Colorizing image...")
image  = cv2.imread(args["image"])
scaled = image.astype("float32") / 255.0
lab    = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)

resized = cv2.resize(lab, (224, 224))
L = cv2.split(resized)[0]
L -= 50

net.setInput(cv2.dnn.blobFromImage(L))
ab = net.forward()[0, :, :, :].transpose((1, 2, 0))
ab = cv2.resize(ab, (image.shape[1], image.shape[0]))

L          = cv2.split(lab)[0]
colorized  = np.concatenate((L[:, :, np.newaxis], ab), axis=2)
colorized  = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
colorized  = np.clip(colorized, 0, 1)
colorized  = (255 * colorized).astype("uint8")

if args["output"]:
    out_path = args["output"]
else:
    base, ext = os.path.splitext(args["image"])
    out_path  = f"{base}_colorized{ext}"

cv2.imwrite(out_path, colorized)
print(f"[INFO] Colorized image saved to: {out_path}")

try:
    cv2.imshow("Original",  image)
    cv2.imshow("Colorized", colorized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
except Exception:
    print("[INFO] GUI display not available — image saved to disk.")