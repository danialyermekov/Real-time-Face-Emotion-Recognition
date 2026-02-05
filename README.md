# Face Emotion Recognition App 🎭

A Streamlit web application for detecting and classifying facial emotions using a trained CNN model. The app supports both image upload and real-time camera input with face detection and emotion classification.

## Features

- 📸 **Image Upload**: Upload photos to detect emotions from faces
- 📷 **Camera Input**: Use your webcam to capture photos and analyze emotions in real-time
- � **WebRTC Streaming**: Real-time video emotion detection
- �🎯 **Multi-Face Detection**: Detects and analyzes multiple faces in a single image
- 📊 **Confidence Scores**: Shows probability distribution for all emotion classes
- 🎨 **Visual Feedback**: Annotated images with bounding boxes and emotion labels
- 🔧 **Flexible Model Loading**: Specify custom model path via command-line argument

## Detected Emotions

The model can recognize the following 5 emotions:

- 😠 **Angry**
- 😨 **Fear**
- 😊 **Happy**
- 😢 **Sad**
- 😮 **Surprise**

## Installation

### Prerequisites

- Python 3.8+
- Virtual environment (recommended)

### Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/HolySxn/emotion-detector.git
   cd emotion-detector
   ```

2. **Create and activate virtual environment**

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Linux/Mac
   # or
   .venv\Scripts\activate  # On Windows
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Ensure the trained model is present**

   Make sure `emotion_cnn.pth` is in the project root directory. If you don't have it:

   - Train the model using `final.ipynb` notebook
   - Or download pre-trained weights (if available)

## Usage

### Basic Usage

Run the app with the default model (`emotion_cnn.pth`):

```bash
streamlit run app.py
```

### Advanced Usage

**Specify a custom model file:**

```bash
streamlit run app.py -- --model path/to/your/model.pth
```

**Use a different model in the same directory:**

```bash
streamlit run app.py -- --model emotion_cnn_v2.pth
```

**Get help:**

```bash
streamlit run app.py -- --help
```

### Command-Line Arguments

- `--model`: Path to the model file (default: `emotion_cnn.pth`)

**Note:** Use `--` to separate Streamlit arguments from app arguments.

The app will automatically open in your default web browser at `http://localhost:8501`

## How to Use the App

### Mode 1: Upload Image

1. Select **"📸 Upload Image"** mode in the sidebar
2. Click **"Browse files"** to upload an image (JPG, JPEG, or PNG)
3. The app will detect faces and display emotions with confidence scores
4. View detailed probability distributions for each detected face

### Mode 2: Camera Input

1. Select **"📸 Upload Image"** mode in the sidebar
2. Click **"Take a picture"** under Camera Input
3. Allow camera access when prompted by your browser
4. The captured photo appears in the left column
5. Emotion detection results appear in the right column
6. View confidence scores and probability distributions below

### Mode 3: Real-Time WebRTC

1. Select **"📹 Real-time Camera"** mode in the sidebar
2. Allow camera access when prompted
3. The app will process video frames in real-time
4. Faces and emotions will be detected and annotated on the video stream

**Note:** WebRTC requires additional dependencies (`streamlit-webrtc`, `av`)

## Model Information

- **Architecture**: Custom CNN (EmotionClassify_CNN)
- **Input Size**: 128×128 RGB images
- **Normalization**:
  - Mean: [0.5109673, 0.5090926, 0.5081655]
  - Std: [0.25057644, 0.25016046, 0.25036415]
- **Number of Classes**: 5 emotions
- **Framework**: PyTorch

## Technical Details

### Model Architecture (EmotionClassify_CNN)

```
Conv Block 1: Conv2d(3→32) + BatchNorm + ReLU + MaxPool + Dropout(0.15)
Conv Block 2: Conv2d(32→64) + BatchNorm + ReLU + MaxPool + Dropout(0.15)
Conv Block 3: Conv2d(64→128) + BatchNorm + ReLU + MaxPool + Dropout(0.20)
Flatten
FC Block 1: Linear(32768→256) + BatchNorm + ReLU + Dropout(0.30)
FC Block 2: Linear(256→128) + BatchNorm + ReLU + Dropout(0.30)
Output: Linear(128→5)
```

### Face Detection

The app uses **OpenCV's Haar Cascade classifier** (`haarcascade_frontalface_default.xml`) for face detection before emotion classification.

### Dependencies

Key packages:

- `streamlit`: Web interface
- `torch`: Deep learning framework
- `torchvision`: Image transformations
- `opencv-python`: Face detection
- `numpy`, `pillow`: Image processing
- `streamlit-webrtc`: Real-time video streaming (optional)

See `requirements.txt` for complete list.

## Troubleshooting

### Model not found

```
Error loading model: [Errno 2] No such file or directory: 'emotion_cnn.pth'
```

**Solution**:

- Ensure `emotion_cnn.pth` is in the project root directory
- Or specify the correct path: `streamlit run app.py -- --model /path/to/model.pth`
- Train the model using `final.ipynb` if you don't have it

### Camera not working

**Solution**:

- Make sure you've allowed camera access in your browser
- Try refreshing the page and allowing permissions again
- Check if another application is using the camera
- Use Firefox or Chrome for best compatibility

### CUDA/GPU issues

The app automatically detects if CUDA is available and falls back to CPU if needed.

```
Device: cuda  # GPU available
Device: cpu   # Using CPU
```

### WebRTC not working

If real-time video streaming doesn't work:

1. Ensure `streamlit-webrtc` and `av` are installed: `pip install streamlit-webrtc av`
2. Try using Chrome or Firefox browsers
3. Check firewall settings
4. Use the camera input mode as an alternative

## Requirements

- Python 3.8+
- See `requirements_simple.txt` or `requirements.txt` for package dependencies

## Project Structure

```
.
├── app.py                      # Main Streamlit application
├── model.py                    # CNN model definition
├── emotion_cnn.pth             # Trained model weights (not in repo)
├── final.ipynb                 # Training notebook
├── requirements.txt            # Python dependencies
├── requirements_simple.txt     # Minimal dependencies (optional)
├── README.md                   # This file
├── MODEL.md                    # Model description
├── .gitignore                  # Git ignore rules
└── .venv/                      # Virtual environment (not in repo)
```

## Features Explained

### Multi-Face Detection

When multiple faces are detected in an image:

- Results are displayed in a grid layout (up to 3 per row)
- Each face shows emotion label and confidence score
- Individual probability distributions for each face

### Single Face Detection

When one face is detected:

- Detailed expandable view with full probability distribution
- Bar chart showing confidence for all 5 emotions
- Larger, more detailed visualization

## Tips for Best Results

- **Lighting**: Use well-lit images for better face detection
- **Face Position**: Front-facing faces work best
- **Image Quality**: Higher resolution images generally perform better
- **Multiple Faces**: The app can handle multiple faces in one image
- **Distance**: Faces should be reasonably large in the frame (minimum 30×30 pixels)

## Model Training

To train your own model:

1. Open `final.ipynb` in Jupyter Notebook or JupyterLab
2. Prepare your dataset in ImageFolder format:
   ```
   Data/
   ├── Angry/
   ├── Fear/
   ├── Happy/
   ├── Sad/
   └── Surprise/
   ```
3. Run all cells in the notebook
4. The trained model will be saved as `emotion_cnn.pth`

## Acknowledgments

Built with:

- [Streamlit](https://streamlit.io/) - Web framework
- [PyTorch](https://pytorch.org/) - Deep learning framework
- [OpenCV](https://opencv.org/) - Computer vision library
- [streamlit-webrtc](https://github.com/whitphx/streamlit-webrtc) - Real-time video streaming
