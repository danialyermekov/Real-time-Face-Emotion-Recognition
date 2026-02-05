import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
import logging
import argparse
from datetime import datetime
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import av

from model import EmotionClassify_CNN

# --------------------------- Logging Setup ---------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# Parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description='Face Emotion Recognition App')
    parser.add_argument(
        '--model', 
        type=str, 
        default='emotion_cnn_aug.pth',
        help='Path to the model file (default: emotion_cnn.pth)'
    )
    # Streamlit adds its own arguments, so we need to handle unknown args
    args, unknown = parser.parse_known_args()
    return args

# Get model path from command line or session state
if 'model_path' not in st.session_state:
    args = parse_args()
    st.session_state.model_path = args.model


# Emotion classes
EMOTIONS = ['Angry', 'Fear', 'Happy', 'Sad', 'Surprise']
MEAN = [0.5109673, 0.5090926, 0.5081655]
STD = [0.25057644, 0.25016046, 0.25036415]

# Load model
@st.cache_resource
def load_model(model_path):
    logger.info(f"Attempting to load model from: {model_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    try:
        model = EmotionClassify_CNN(num_classes=len(EMOTIONS))
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        logger.info("Model loaded successfully")
        return model, device
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}", exc_info=True)
        raise

# Load face detector
@st.cache_resource
def load_face_detector():
    logger.info("Loading face detector (Haar Cascade)")
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    logger.info("Face detector loaded successfully")
    return face_cascade

# Image preprocessing
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])
    return transform(image).unsqueeze(0)

# Predict emotion
def predict_emotion(image, model, device):
    try:
        with torch.no_grad():
            image_tensor = preprocess_image(image).to(device)
            outputs = model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        emotion = EMOTIONS[predicted_class]
        logger.debug(f"Predicted emotion: {emotion} with confidence {confidence:.3f}")
        return emotion, confidence, probabilities[0].cpu().numpy()
    except Exception as e:
        logger.error(f"Error predicting emotion: {str(e)}", exc_info=True)
        raise

# Detect faces and predict emotions
def detect_and_predict(image, model, device, face_cascade):
    img_array = np.array(image)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    logger.info(f"Detected {len(faces)} face(s) in image")
    
    results = []
    annotated_image = img_array.copy()
    
    for idx, (x, y, w, h) in enumerate(faces):
        try:
            face_img = image.crop((x, y, x+w, y+h))
            emotion, confidence, probs = predict_emotion(face_img, model, device)
            results.append({
                'emotion': emotion,
                'confidence': confidence,
                'bbox': (x, y, w, h),
                'probabilities': probs
            })
            
            # Draw rectangle and label
            cv2.rectangle(annotated_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            label = f"{emotion}: {confidence:.2f}"
            cv2.putText(annotated_image, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            logger.info(f"Face {idx+1}: {emotion} (confidence: {confidence:.3f})")
        except Exception as e:
            logger.error(f"Error processing face {idx+1}: {str(e)}", exc_info=True)
    
    return annotated_image, results

class EmotionVideoTransformer(VideoTransformerBase):
    def __init__(self):
        logger.info("Initializing EmotionVideoTransformer for WebRTC stream")
        model_path = st.session_state.get('model_path', 'emotion_cnn.pth')
        self.model, self.device = load_model(model_path)
        self.face_cascade = load_face_detector()
        self.frame_count = 0
    
    def transform(self, frame):
        self.frame_count += 1
        if self.frame_count % 30 == 0:
            logger.debug(f"Processing frame {self.frame_count}")
        
        img = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(img_rgb)
        
        annotated_image, _ = detect_and_predict(pil_image, self.model, self.device, self.face_cascade)
        
        return cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)

# Main Streamlit app
def main():
    logger.info("Starting Streamlit WebRTC app")
    st.set_page_config(page_title="Emotion Recognition", page_icon="😊", layout="wide")
    
    st.title("🎭 Face Emotion Recognition")
    st.markdown("Upload an image or use your camera to detect emotions in real-time!")
    
    try:
        model_path = st.session_state.model_path
        model, device = load_model(model_path)
        face_cascade = load_face_detector()
    except Exception as e:
        logger.error(f"Failed to initialize app: {str(e)}")
        st.error(f"Failed to load model or face detector: {str(e)}")
        return
    
    # Sidebar
    st.sidebar.title("Settings")
    mode = st.sidebar.radio("Choose mode:", ["📸 Upload Image", "📹 Real-time Camera"])
    logger.info(f"User selected mode: {mode}")
    
    if mode == "📸 Upload Image":
        st.header("Upload an Image")

        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        
        st.markdown("---")
        col_headers = st.columns([1, 1])
        with col_headers[0]:
            st.subheader("Camera Input")
        with col_headers[1]:
            st.subheader("Emotion Detection")
        
        input_col, result_col = st.columns([1, 1])
        
        with input_col:
            camera_photo = st.camera_input("Take a picture", )
        
        with result_col:
            st.subheader("Recognition Result")
            result_placeholder = st.empty()
        
        # Determine which image to use
        image = None
        if uploaded_file is not None:
            logger.info(f"Image uploaded: {uploaded_file.name}")
            image = Image.open(uploaded_file).convert('RGB')
            
            st.markdown("---")
            col1, col2 = st.columns([1, 1])
            with col1:
                st.subheader("Original Image")
                st.image(image, use_container_width=True)
            
            with col2:
                st.subheader("Recognition Result")
                with st.spinner("Analyzing..."):
                    annotated_image, results = detect_and_predict(image, model, device, face_cascade)
                    st.image(annotated_image, use_container_width=True)
                    
        elif camera_photo is not None:
            logger.info("Photo captured from camera")
            image = Image.open(camera_photo).convert('RGB')
            
            with result_col:
                with st.spinner("Analyzing..."):
                    annotated_image, results = detect_and_predict(image, model, device, face_cascade)
                    st.image(annotated_image, use_container_width=True)
        
        if image is not None:            
            if results:
                st.success(f"Detected {len(results)} face(s)")
                
                if len(results) > 1:
                    result_cols = st.columns(min(len(results), 3))
                    for idx, result in enumerate(results):
                        with result_cols[idx % 3]:
                            st.metric(
                                label=f"Face {idx + 1}",
                                value=result['emotion'].upper(),
                                delta=f"{result['confidence']:.1%}"
                            )
                            prob_dict = {EMOTIONS[i]: result['probabilities'][i] for i in range(len(EMOTIONS))}
                            st.bar_chart(prob_dict, height=150)
                else:
                    result = results[0]
                    with st.expander(f"Face 1 - Emotion: {result['emotion'].upper()} ({result['confidence']:.2%})", expanded=True):
                        st.write("**Probability Distribution:**")
                        prob_dict = {EMOTIONS[i]: result['probabilities'][i] for i in range(len(EMOTIONS))}
                        st.bar_chart(prob_dict)
            else:
                st.warning("No faces detected. Please try another image.")
                logger.warning("No faces detected in uploaded image")
    
    else:
        st.header("Real-time Emotion Detection")
        st.markdown("**Note:** Allow camera access when prompted.")
        logger.info("Starting WebRTC camera mode")
        
        st.subheader("WebRTC Stream")
        rtc_configuration = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
        webrtc_streamer(
            key="emotion-detection",
            video_transformer_factory=EmotionVideoTransformer,
            rtc_configuration=rtc_configuration,
            media_stream_constraints={"video": True, "audio": False}
        )        
    
    st.sidebar.markdown("---")
    st.sidebar.info(
        "**About:**\n\n"
        "This app uses a CNN model to detect emotions from facial expressions.\n\n"
        f"**Emotions detected:** {', '.join(EMOTIONS)}\n\n"
        f"**Model file:** {st.session_state.model_path}\n\n"
        f"**Device:** {device}"
    )

if __name__ == "__main__":
    main()
