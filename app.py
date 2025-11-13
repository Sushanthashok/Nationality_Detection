import streamlit as st
import numpy as np
import cv2
from mtcnn import MTCNN
from PIL import Image
import os
import json

st.set_page_config(page_title="Nationality Detection App", layout="wide")
st.title("🌍 Nationality, Emotion, Age & Dress Color Detection")

# -----------------------------
#    MODEL PATHS
# -----------------------------
EMOTION_MODEL_PATH = "models/emotion_mobilenetv2.h5"
NATIONALITY_MODEL_PATH = "models/nationality_mobilenetv2.h5"
NATIONALITY_LABELS_PATH = "models/nationality_labels.json"

# -----------------------------
#    LOAD MODELS SAFELY
# -----------------------------
emotion_model = None
use_real_emotion_model = False

nationality_model = None
use_real_nationality_model = False
nationality_labels = ["Indian", "United States", "African", "Other"]


# -----------------------------
#  TRY LOADING EMOTION MODEL
# -----------------------------
def try_load_emotion_model():
    global emotion_model, use_real_emotion_model
    if not os.path.exists(EMOTION_MODEL_PATH):
        st.warning("No real emotion model found. Using placeholder emotions.")
        return False
    try:
        import tensorflow as tf  # noqa: F401
        from tensorflow.keras.models import load_model
    except Exception:
        st.warning("TensorFlow not available. Using placeholder emotions.")
        return False

    @st.cache_resource
    def load_emotion_model(p):
        return load_model(p)

    try:
        emotion_model = load_emotion_model(EMOTION_MODEL_PATH)
        use_real_emotion_model = True
        st.success("Emotion model loaded successfully.")
        return True
    except Exception as e:
        st.warning(f"Failed to load emotion model: {e}")
        use_real_emotion_model = False
        return False


# -----------------------------
#  TRY LOADING NATIONALITY MODEL
# -----------------------------
def try_load_nationality_model():
    global nationality_model, use_real_nationality_model, nationality_labels

    if not os.path.exists(NATIONALITY_MODEL_PATH):
        st.warning("No real nationality model found. Using placeholder.")
        return False

    try:
        import tensorflow as tf  # noqa: F401
        from tensorflow.keras.models import load_model
    except Exception:
        st.warning("TensorFlow not available. Using placeholder nationality.")
        return False

    @st.cache_resource
    def load_nat_model(p):
        return load_model(p)

    try:
        nationality_model = load_nat_model(NATIONALITY_MODEL_PATH)
        use_real_nationality_model = True

        # Load label order
        if os.path.exists(NATIONALITY_LABELS_PATH):
            with open(NATIONALITY_LABELS_PATH, "r") as f:
                nationality_labels = json.load(f)

        st.success("Nationality model loaded successfully.")
        return True
    except Exception as e:
        st.warning(f"Failed to load nationality model: {e}")
        use_real_nationality_model = False
        return False


# Load both models on startup
try_load_emotion_model()
try_load_nationality_model()

# -----------------------------
#   PLACEHOLDER EMOTION
# -----------------------------
EMOTION_LABELS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]


def predict_emotion_placeholder(_):
    return "neutral", 0.50


# -----------------------------
#   REAL EMOTION PREDICT
# -----------------------------
def preprocess_face_for_emotion(face_np):
    face = cv2.resize(face_np, (224, 224))
    face = face.astype("float32") / 255.0
    return np.expand_dims(face, axis=0)


def predict_emotion_real(face_np):
    x = preprocess_face_for_emotion(face_np)
    preds = emotion_model.predict(x)[0]
    idx = int(np.argmax(preds))
    return EMOTION_LABELS[idx], float(preds[idx])


# -----------------------------
#   NATIONALITY PREDICTION
# -----------------------------
def preprocess_face_for_nat(face_np):
    face = cv2.resize(face_np, (224, 224))
    face = face.astype("float32") / 255.0
    return np.expand_dims(face, axis=0)


def predict_nationality_real(face_np):
    x = preprocess_face_for_nat(face_np)
    preds = nationality_model.predict(x)[0]
    idx = int(np.argmax(preds))
    top3_idx = np.argsort(preds)[::-1][:3]
    top3 = [(nationality_labels[i], float(preds[i])) for i in top3_idx]
    return nationality_labels[idx], float(preds[idx]), top3


# -----------------------------
#   PLACEHOLDER NATIONALITY
# -----------------------------
def predict_nationality_placeholder(_):
    return "Other", 0.50, []


# -----------------------------
#   AGE PREDICTION (PLACEHOLDER)
# -----------------------------
def predict_age_placeholder(_):
    return int(np.random.randint(18, 40))


# -----------------------------
#   IMPROVED DRESS COLOR DETECTION
# -----------------------------
def detect_dress_color(image, face_box, k_clusters=3):
    """
    Estimate dress color from the region below the detected face.
    Args:
        image: HxWx3 RGB numpy array (uint8)
        face_box: (x, y, w, h) bounding box of face (ints)
        k_clusters: number of kmeans clusters to use
    Returns:
        (color_name, confidence) where color_name is one of:
            Red, Orange, Yellow, Green, Blue, Purple, Pink, Brown, Gray, Black, White, Unknown
        confidence: float in [0,1]
    """
    h_img, w_img = image.shape[:2]
    x, y, w, h = face_box
    x, y, w, h = int(x), int(y), int(w), int(h)

    # Compute torso region: a rectangle below the face
    top = y + int(h * 0.6)
    if top >= h_img:
        top = y
    bottom = min(h_img, y + int(h * 2.4))
    left = max(0, x - int(0.25 * w))
    right = min(w_img, x + w + int(0.25 * w))

    if bottom - top < 10 or right - left < 10:
        top = min(h_img - 1, y + int(h * 0.5))
        bottom = min(h_img, top + max(20, int(h * 1.2)))
        left = max(0, x - int(w * 0.3))
        right = min(w_img, x + w + int(w * 0.3))

    torso = image[top:bottom, left:right].copy()
    if torso.size == 0:
        return "Unknown", 0.0

    # Resize to manageable size
    torso_small = cv2.resize(torso, (80, 80), interpolation=cv2.INTER_AREA)

    # Convert to HSV and mask low-saturation/value pixels (likely skin/shadow)
    hsv = cv2.cvtColor(torso_small, cv2.COLOR_RGB2HSV)
    h_channel, s_channel, v_channel = cv2.split(hsv)
    mask = (s_channel >= 30) & (v_channel >= 40)
    mask = mask.astype("uint8")

    # Candidate pixels
    pixels = torso_small.reshape(-1, 3)
    mask_flat = mask.reshape(-1)
    if mask_flat.sum() < 20:
        pixels_filtered = pixels
    else:
        pixels_filtered = pixels[mask_flat == 1]

    if pixels_filtered.shape[0] == 0:
        pixels_filtered = pixels

    Z = pixels_filtered.astype("float32")
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.2)
    K = min(k_clusters, max(1, len(Z) // 10))
    if K <= 0:
        return "Unknown", 0.0

    try:
        _, labels, centers = cv2.kmeans(Z, K, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    except Exception:
        center = np.mean(Z, axis=0)
        centers = np.array([center], dtype="float32")
        labels = np.zeros((Z.shape[0], 1), dtype="int32")

    labels_flat = labels.flatten()
    counts = np.bincount(labels_flat, minlength=centers.shape[0])
    best_idx = int(np.argmax(counts))
    dominant_rgb = centers[best_idx].astype("uint8").tolist()  # in RGB order

    named_colors = {
        "Red": (220, 20, 60),
        "Orange": (255, 140, 0),
        "Yellow": (255, 215, 0),
        "Green": (34, 139, 34),
        "Blue": (30, 144, 255),
        "Purple": (148, 0, 211),
        "Pink": (255, 105, 180),
        "Brown": (150, 75, 0),
        "Gray": (128, 128, 128),
        "Black": (20, 20, 20),
        "White": (245, 245, 245)
    }

    def color_distance(c1, c2):
        return np.linalg.norm(np.array(c1, dtype=float) - np.array(c2, dtype=float))

    best_name = "Unknown"
    best_dist = float("inf")
    for name, rgb in named_colors.items():
        d = color_distance(dominant_rgb, rgb)
        if d < best_dist:
            best_dist = d
            best_name = name

    max_dist = 441.0
    confidence = max(0.0, 1.0 - (best_dist / max_dist))
    dominance = counts[best_idx] / float(np.sum(counts))
    confidence = float(confidence * 0.6 + dominance * 0.4)

    if best_name in ("Black", "Gray") and np.mean(v_channel) < 50:
        return "Unknown", 0.2

    return best_name, round(confidence, 2)


# -----------------------------
#   MTCNN FACE DETECTOR
# -----------------------------
detector = MTCNN()


# -----------------------------
#   STREAMLIT UI
# -----------------------------
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    st.image(image, caption="Uploaded Image", use_container_width=True)

    st.subheader("🔍 Detecting face...")

    try:
        results = detector.detect_faces(image_np)
        if results is None:
            results = []
    except Exception as e:
        st.error(f"Face detector error: {e}")
        results = []

    # Keep only the largest face (highest area) to avoid false positives
    if len(results) > 1:
        def area_of(f):
            bx, by, bw, bh = f.get("box", (0, 0, 0, 0))
            return max(0, abs(int(bw))) * max(0, abs(int(bh)))
        results = sorted(results, key=lambda f: area_of(f), reverse=True)
        results = [r for r in results if area_of(r) > 0][:1]

    if len(results) == 0:
        st.error("No face detected.")
    else:
        st.success(f"Detected {len(results)} face(s)")

        for i, face in enumerate(results):
            st.markdown(f"### 👤 Face #{i + 1}")

            x, y, w, h = face["box"]
            x, y, w, h = int(abs(x)), int(abs(y)), int(abs(w)), int(abs(h))
            x2 = min(image_np.shape[1], x + w)
            y2 = min(image_np.shape[0], y + h)
            x1 = max(0, x)
            y1 = max(0, y)

            if x1 >= x2 or y1 >= y2:
                face_crop = image_np.copy()
            else:
                face_crop = image_np[y1:y2, x1:x2]

            st.image(Image.fromarray(face_crop), width=200)

            # NATIONALITY
            if use_real_nationality_model:
                try:
                    nationality, nat_conf, nat_top3 = predict_nationality_real(face_crop)
                except Exception as e:
                    st.warning(f"Nationality model error: {e}. Using placeholder.")
                    nationality, nat_conf, nat_top3 = predict_nationality_placeholder(face_crop)
            else:
                nationality, nat_conf, nat_top3 = predict_nationality_placeholder(face_crop)

            st.write(f"**Nationality:** {nationality} ({nat_conf:.2f})")

            # EMOTION
            if use_real_emotion_model:
                try:
                    emotion, econf = predict_emotion_real(face_crop)
                except Exception as e:
                    st.warning(f"Emotion model error: {e}. Using placeholder.")
                    emotion, econf = predict_emotion_placeholder(face_crop)
            else:
                emotion, econf = predict_emotion_placeholder(face_crop)

            st.write(f"**Emotion:** {emotion} ({econf:.2f})")

            # AGE (placeholder)
            age = predict_age_placeholder(face_crop)

            # DRESS COLOR (improved)
            dress_color, color_conf = detect_dress_color(image_np, (x, y, w, h))

            # CONDITIONAL OUTPUTS
            if nationality == "Indian":
                st.write(f"**Age:** {age}")
                st.write(f"**Dress Color:** {dress_color} (confidence: {color_conf:.2f})")
            elif nationality == "United States":
                st.write(f"**Age:** {age}")
            elif nationality == "African":
                st.write(f"**Dress Color:** {dress_color} (confidence: {color_conf:.2f})")

            # TOP-3 NATIONALITY
            if nat_top3:
                st.write("Top-3 Nationality Predictions:")
                for label, c in nat_top3:
                    st.write(f"- {label}: {c:.2f}")
