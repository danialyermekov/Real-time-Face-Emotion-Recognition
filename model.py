import torch.nn as nn
import torch.nn.functional as F

# Define the CNN model architecture
class EmotionClassify_CNN(nn.Module):
    def __init__(self, num_classes):
        super(EmotionClassify_CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.drop1 = nn.Dropout(0.15)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.drop2 = nn.Dropout(0.15)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.drop3 = nn.Dropout(0.20)
        self.feature_h = 16   
        self.feature_w = 16
        in_features = 128 * self.feature_h * self.feature_w

        self.fc1 = nn.Linear(in_features, 256)
        self.fbn1 = nn.BatchNorm1d(256)
        self.fdrop1 = nn.Dropout(0.3)

        self.fc2 = nn.Linear(256, 128)
        self.fbn2 = nn.BatchNorm1d(128)
        self.fdrop2 = nn.Dropout(0.3)

        self.fc3 = nn.Linear(128, num_classes)

        self.flatten = nn.Flatten(start_dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.drop1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)
        x = self.drop2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool3(x)
        x = self.drop3(x)

        x = self.flatten(x) 

        x = self.fc1(x)
        x = self.fbn1(x)
        x = F.relu(x)
        x = self.fdrop1(x)

        x = self.fc2(x)
        x = self.fbn2(x)
        x = F.relu(x)
        x = self.fdrop2(x)

        x = self.fc3(x)
        return x