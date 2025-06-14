import torch
from torch.utils.data import Dataset
import os
from PIL import Image
import torchvision.transforms as transforms

class FER2013Dataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        
        self.image_paths = []
        self.emotions = []
        
        # 各感情クラスのディレクトリをスキャン
        for emotion_idx, emotion_class in enumerate(self.classes):
            emotion_dir = os.path.join(self.root_dir, emotion_class)
            if not os.path.exists(emotion_dir):
                continue
                
            for img_file in os.listdir(emotion_dir):
                if img_file.endswith('.png') or img_file.endswith('.jpg'):
                    self.image_paths.append(os.path.join(emotion_dir, img_file))
                    self.emotions.append(emotion_idx)
    
    def __len__(self):
        return len(self.image_paths)
        
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        emotion = self.emotions[idx]
        
        # 画像の読み込み
        image = Image.open(img_path)
        
        # グレースケールの場合はRGBに変換
        if image.mode == 'L':
            image = image.convert('RGB')
            
        # 変換の適用
        if self.transform:
            image = self.transform(image)
            
        return image, emotion

# データ変換の定義
def get_transforms(phase):
    if phase == 'train':
        transform = transforms.Compose([
            transforms.Resize((48, 48)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((48, 48)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    return transform