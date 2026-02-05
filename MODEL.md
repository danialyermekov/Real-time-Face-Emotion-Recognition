# Model Documentation

## Overview

**EmotionClassify_CNN** is a custom Convolutional Neural Network designed for facial emotion recognition. The model classifies facial expressions into 5 distinct emotion categories.

## Model Architecture

### Network Structure

```
EmotionClassify_CNN(
  Input: 128×128×3 RGB images

  # Convolutional Blocks
  Conv Block 1:
    - Conv2d(3 → 32, kernel_size=3, padding=1)
    - BatchNorm2d(32)
    - F.relu() activation
    - MaxPool2d(kernel_size=2, stride=2)
    - Dropout2d(p=0.15)

  Conv Block 2:
    - Conv2d(32 → 64, kernel_size=3, padding=1)
    - BatchNorm2d(64)
    - F.relu() activation
    - MaxPool2d(kernel_size=2, stride=2)
    - Dropout2d(p=0.15)

  Conv Block 3:
    - Conv2d(64 → 128, kernel_size=3, padding=1)
    - BatchNorm2d(128)
    - F.relu() activation
    - MaxPool2d(kernel_size=2, stride=2)
    - Dropout2d(p=0.20)

  # Flatten Layer
  Flatten: 128 × 16 × 16 = 32,768 features

  # Fully Connected Blocks
  FC Block 1:
    - Linear(32,768 → 256)
    - BatchNorm1d(256)
    - F.relu() activation
    - Dropout(p=0.30)

  FC Block 2:
    - Linear(256 → 128)
    - BatchNorm1d(128)
    - F.relu() activation
    - Dropout(p=0.30)

  # Output Layer
  Output:
    - Linear(128 → 5)
)
```

### Total Parameters

- **Convolutional Layers**: ~74K parameters
- **Fully Connected Layers**: ~8.4M parameters
- **Total**: ~8.5M trainable parameters

## Input Specifications

### Image Preprocessing

```python
Input Size: 128 × 128 pixels
Color Space: RGB (3 channels)
Data Type: Float32
Value Range: [0, 1] (after ToTensor())
```

### Normalization

The model expects normalized inputs using the following statistics computed from the training dataset:

```python
MEAN = [0.5109673, 0.5090926, 0.5081655]  # Per-channel mean
STD = [0.25057644, 0.25016046, 0.25036415]  # Per-channel standard deviation
```

**Normalization Formula:**

```
normalized_pixel = (pixel - MEAN) / STD
```

These values were calculated from the training dataset and ensure that the input distribution matches what the model was trained on.

## Output Specifications

### Emotion Classes

The model outputs logits for 5 emotion classes:

| Index | Emotion  | Description                          |
| ----- | -------- | ------------------------------------ |
| 0     | Angry    | Angry or irritated facial expression |
| 1     | Fear     | Fearful or anxious expression        |
| 2     | Happy    | Happy or joyful expression           |
| 3     | Sad      | Sad or sorrowful expression          |
| 4     | Surprise | Surprised or shocked expression      |

### Prediction Format

```python
# Model output (before softmax)
logits = model(input_tensor)  # Shape: [batch_size, 5]

# Convert to probabilities
probabilities = F.softmax(logits, dim=1)  # Shape: [batch_size, 5]

# Get predicted class
predicted_class = torch.argmax(probabilities, dim=1)  # Shape: [batch_size]
```

## Training Dataset

### Dataset Information

**Source**: [Human Face Emotions Dataset](https://www.kaggle.com/datasets/samithsachidanandan/human-face-emotions)

**Platform**: Kaggle

**Dataset Structure**:

```
Data/
├── Angry/
│   └── [angry face images]
├── Fear/
│   └── [fearful face images]
├── Happy/
│   └── [happy face images]
├── Sad/
│   └── [sad face images]
└── Surprise/
    └── [surprised face images]
```

### Dataset Characteristics

- **Format**: RGB images of human faces
- **Classes**: 5 balanced emotion categories
- **Image Resolution**: Variable (resized to 128×128 during preprocessing)
- **Split**: Training, validation, and test sets (see `final.ipynb` for details)

### Data Augmentation

During training, the following augmentations were applied:

```python
transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(p=0.5),  # Optional
    transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Optional
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD)
])
```

## Model Performance

### Training Configuration

```python
Loss Function: CrossEntropyLoss
Optimizer: Adam
Learning Rate: 1e-3 (initial)
Batch Size: 32
Epochs: 50
Device: CUDA (GPU) / CPU fallback
```

### Regularization Techniques

1. **Dropout**: Applied after each convolutional block (15-20%) and fully connected layers (30%)
2. **Batch Normalization**: Used in all convolutional and fully connected blocks
3. **Data Augmentation**: Random flips and color jittering (if enabled)
4. **Early Stopping**: Monitored validation accuracy

### Evaluation Metrics

- **Accuracy**: Overall classification accuracy
- **Per-Class Accuracy**: Individual accuracy for each emotion
- **Confusion Matrix**: Detailed performance across all classes

(See `final.ipynb` for specific performance metrics)

## Model Files

### Saved Model Format

**File**: `emotion_cnn.pth`

**Format**: PyTorch state_dict (weights only)

**Loading Example**:

```python
from model import EmotionClassify_CNN
import torch

model = EmotionClassify_CNN(num_classes=5)
model.load_state_dict(torch.load('emotion_cnn.pth', map_location='cpu'))
model.eval()
```

## Inference Pipeline

### Complete Inference Flow

```python
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from model import EmotionClassify_CNN

# 1. Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EmotionClassify_CNN(num_classes=5)
model.load_state_dict(torch.load('emotion_cnn.pth', map_location=device))
model.to(device)
model.eval()

# 2. Define preprocessing
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5109673, 0.5090926, 0.5081655],
        std=[0.25057644, 0.25016046, 0.25036415]
    )
])

# 3. Load and preprocess image
image = Image.open('face.jpg').convert('RGB')
input_tensor = transform(image).unsqueeze(0).to(device)

# 4. Perform inference
with torch.no_grad():
    logits = model(input_tensor)
    probabilities = F.softmax(logits, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1).item()
    confidence = probabilities[0][predicted_class].item()

# 5. Get emotion label
emotions = ['Angry', 'Fear', 'Happy', 'Sad', 'Surprise']
emotion = emotions[predicted_class]

print(f"Predicted Emotion: {emotion}")
print(f"Confidence: {confidence:.2%}")
```

## Design Choices

### Why This Architecture?

1. **Three Convolutional Blocks**: Sufficient depth to learn hierarchical features from facial expressions without overfitting
2. **Progressive Channel Increase** (32→64→128): Gradually increases model capacity while maintaining computational efficiency
3. **Batch Normalization**: Stabilizes training and allows higher learning rates
4. **Dropout**: Prevents overfitting, especially important given the limited dataset size
5. **MaxPooling**: Reduces spatial dimensions while preserving important features
6. **Dense Layers**: Final feature extraction and classification with appropriate capacity

### Alternative Approaches Considered

- **Transfer Learning**: Using pre-trained models (VGG, ResNet) - would be more powerful but requires more resources
- **Deeper Networks**: More convolutional layers - risk of overfitting with limited data
- **Attention Mechanisms**: Could focus on important facial regions - adds complexity

## Limitations

1. **Face Detection Dependency**: Requires accurate face detection as preprocessing step
2. **Lighting Sensitivity**: Performance may degrade under poor lighting conditions
3. **Pose Variation**: Best performance on frontal faces; profile views may be less accurate
4. **Expression Intensity**: Subtle expressions may be harder to classify
5. **Cultural Differences**: Trained on specific dataset; may not generalize to all populations
6. **Five Classes Only**: Limited to the trained emotion categories

## Future Improvements

1. **Larger Dataset**: Train on more diverse facial expression datasets
2. **Data Augmentation**: More aggressive augmentation for better generalization
3. **Multi-Task Learning**: Joint training with facial landmark detection
4. **Ensemble Methods**: Combine multiple models for better accuracy
5. **Attention Mechanisms**: Focus on discriminative facial regions
6. **More Emotion Classes**: Expand to neutral, disgust, contempt, etc.
7. **Real-Time Optimization**: Model quantization or pruning for faster inference

## References

- **Dataset**: [Human Face Emotions - Kaggle](https://www.kaggle.com/datasets/samithsachidanandan/human-face-emotions)
- **Framework**: [PyTorch](https://pytorch.org/)
- **Face Detection**: [OpenCV Haar Cascades](https://docs.opencv.org/4.x/db/d28/tutorial_cascade_classifier.html)

## License

This model is for educational and research purposes.
