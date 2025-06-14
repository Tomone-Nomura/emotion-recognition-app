import torch
import torch.nn as nn
import torch.nn.functional as F

class EmotionCNN(nn.Module):
    def __init__(self, num_classes=7):
        super(EmotionCNN, self).__init__()
        
        # 1つ目の畳み込みブロック
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 2つ目の畳み込みブロック
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 3つ目の畳み込みブロック
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 4つ目の畳み込みブロック
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 全結合層
        self.fc1 = nn.Linear(512 * 3 * 3, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)
        
    def forward(self, x):
        # 畳み込みブロック1
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        # 畳み込みブロック2
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        # 畳み込みブロック3
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        
        # 畳み込みブロック4
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool4(x)
        
        # 全結合層
        x = x.view(-1, 512 * 3 * 3)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x