"""
╔══════════════════════════════════════════════════════╗
║         J.A.R.V.I.S. HUD — Computer Vision          ║
║              Workshop · Standalone Script            ║
╚══════════════════════════════════════════════════════╝

HOW TO RUN:
  1. Install dependencies (run this once in your terminal):
       pip install mediapipe==0.10.9 opencv-python

  2. Run the script:
       python jarvis_hud.py

  3. A live webcam window will open with the HUD overlay.
     Press Q to quit.

🔧 CUSTOMIZE: Edit the settings under "YOUR SETTINGS" below.
"""

import sys
import urllib.request
import os
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# ══════════════════════════════════════════════════════════
#   🔧 YOUR SETTINGS — change anything in this block!
# ══════════════════════════════════════════════════════════

PILOT_NAME   = "PILOT: TONY STARK"   # 🔧 your name here!
STATUS_MSG   = "JARVIS ACTIVE  |  THREAT LEVEL: LOW  |  SYSTEMS: NOMINAL"

#  Colors are (Blue, Green, Red) — each number 0–255
#  Iron Man red      → (0, 0, 220)
#  Arc reactor blue  → (255, 180, 0)
#  War Machine silver→ (200, 200, 200)
#  Rescue gold       → (0, 215, 255)
#  Default cyan-green→ (0, 255, 180)
HUD_COLOR    = (0, 255, 180)
LABEL_BG     = (0, 80, 50)
STATUS_COLOR = (0, 200, 255)

SHOW_POWER_BAR = True   # 🔧 set to False to hide the power bar
POWER_LEVEL    = 87     # 🔧 fake power level 0–100

# ══════════════════════════════════════════════════════════


# ── Step 1: Download the face detection model if needed ──────────────────────
MODEL_PATH = "face_detector.tflite"
MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_detector/blaze_face_short_range/float16/1/"
    "blaze_face_short_range.tflite"
)

if not os.path.exists(MODEL_PATH):
    print("📥 Downloading face detection model (one-time, ~1 MB)...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("✅ Model downloaded!")
else:
    print("✅ Model already present.")


# ── Step 2: Load the face detector ───────────────────────────────────────────
base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
options      = mp_vision.FaceDetectorOptions(
    base_options=base_options,
    min_detection_confidence=0.5
)
detector = mp_vision.FaceDetector.create_from_options(options)
print("✅ Face detector loaded.")


# ── Step 3: Drawing helpers ───────────────────────────────────────────────────
def draw_corners(img, x, y, w, h, color, length=22):
    """Draws 4 corner brackets for the sci-fi targeting look."""
    pts = [
        ((x,   y),   (x+length, y),     (x,   y+length)),
        ((x+w, y),   (x+w-length, y),   (x+w, y+length)),
        ((x,   y+h), (x+length, y+h),   (x,   y+h-length)),
        ((x+w, y+h), (x+w-length, y+h), (x+w, y+h-length)),
    ]
    for corner, h_pt, v_pt in pts:
        cv2.line(img, corner, h_pt, color, 2)
        cv2.line(img, corner, v_pt, color, 2)


def draw_label(img, text, x, y, color, bg):
    """Draws a filled label box with text above the bounding box."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, 0.55, 1)
    cv2.rectangle(img, (x, y - th - 10), (x + tw + 10, y), bg, -1)
    cv2.putText(img, text, (x + 5, y - 5), font, 0.55, color, 1, cv2.LINE_AA)


def draw_status(img, msg, color):
    """Draws a status bar across the bottom of the frame."""
    h, w = img.shape[:2]
    cv2.rectangle(img, (0, h - 30), (w, h), (0, 0, 0), -1)
    cv2.putText(img, msg, (10, h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def draw_power_bar(img, x, y, w, level, color):
    """Draws a horizontal power bar below the bounding box."""
    bar_y  = y + 6
    fill_w = int(w * (level / 100))
    cv2.rectangle(img, (x, bar_y), (x + w, bar_y + 8), (40, 40, 40), -1)
    cv2.rectangle(img, (x, bar_y), (x + fill_w, bar_y + 8), color, -1)
    cv2.putText(img, f"PWR {level}%", (x + w + 6, bar_y + 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)


def process_frame(frame):
    """Run detection and draw the full HUD on a single frame."""
    h, w   = frame.shape[:2]
    output = frame.copy()

    # MediaPipe needs RGB
    rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    results  = detector.detect(mp_image)

    face_count = 0

    if results.detections:
        face_count = len(results.detections)
        for det in results.detections:
            score  = det.categories[0].score
            bbox   = det.bounding_box
            x      = max(0, bbox.origin_x)
            y      = max(0, bbox.origin_y)
            bw     = bbox.width
            bh     = bbox.height

            # Clamp box to frame edges
            x2 = min(w, x + bw)
            y2 = min(h, y + bh)
            bw = x2 - x
            bh = y2 - y

            # Subtle tint inside the box
            overlay = output.copy()
            cv2.rectangle(overlay, (x, y), (x + bw, y + bh), HUD_COLOR, -1)
            cv2.addWeighted(overlay, 0.08, output, 0.92, 0, output)

            # Corner brackets + label
            draw_corners(output, x, y, bw, bh, HUD_COLOR)
            draw_label(output, f"{PILOT_NAME}  |  CONF: {score:.0%}",
                       x, y, HUD_COLOR, LABEL_BG)

            # Optional power bar
            if SHOW_POWER_BAR:
                draw_power_bar(output, x, y + bh, bw, POWER_LEVEL, HUD_COLOR)

    # Status bar — shows face count + fps hint
    status = f"{face_count} FACE(S) DETECTED  |  {STATUS_MSG}" if face_count else f"SCANNING...  |  {STATUS_MSG}"
    draw_status(output, status, STATUS_COLOR)

    return output


# ── Step 4: Open the webcam and run the live loop ────────────────────────────
print("\n🎬 Opening webcam... press Q to quit.\n")

# Try camera indexes 0, 1, 2 — on Mac the built-in camera is sometimes not index 0
cap = None
for index in range(3):
    print(f"   Trying camera index {index}...")
    test = cv2.VideoCapture(index)
    # Try to actually read a frame — isOpened() alone isn't reliable on Mac
    if test.isOpened():
        ret, _ = test.read()
        if ret:
            cap = test
            print(f"✅ Camera found at index {index}!")
            break
    test.release()

if cap is None:
    print("\n❌ Could not open any webcam. Things to try:")
    print("   1. Go to System Settings → Privacy & Security → Camera")
    print("      and make sure your Terminal app has camera access.")
    print("   2. Close any other app using the camera (Zoom, FaceTime, etc.)")
    print("   3. Try unplugging and replugging an external webcam.")
    sys.exit(1)

# Set a reasonable resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Warm up — discard the first few frames (Mac cameras take a moment to adjust)
for _ in range(5):
    cap.read()

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Lost webcam feed — exiting.")
        break

    # Flip horizontally so it feels like a mirror
    frame = cv2.flip(frame, 1)

    # Run the HUD pipeline on this frame
    hud_frame = process_frame(frame)

    # Show the result in a window
    cv2.imshow("J.A.R.V.I.S. HUD  |  Press Q to quit", hud_frame)

    # Q or Escape → quit
    key = cv2.waitKey(1) & 0xFF
    if key in (ord('q'), ord('Q'), 27):
        print("\n👋 Shutting down J.A.R.V.I.S. — goodbye.")
        break

cap.release()
cv2.destroyAllWindows()
