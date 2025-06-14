import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import copy
import os

from emotion_dataset import FER2013Dataset, get_transforms
from emotion_model import EmotionCNN

# 設定
batch_size = 64
num_epochs = 20
learning_rate = 0.001
num_classes = 7
model_save_path = 'models/emotion_model.pt'

# デバイスの設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

# データローダーの設定
def setup_dataloaders():
    # トレーニングデータセット
    train_dataset = FER2013Dataset(
        root_dir='archive/train',
        transform=get_transforms('train')
    )
    
    # テストデータセット
    test_dataset = FER2013Dataset(
        root_dir='archive/test',
        transform=get_transforms('val')
    )
    
    # データローダー
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    dataset_sizes = {
        'train': len(train_dataset),
        'val': len(test_dataset)
    }
    
    print(f"Dataset sizes: {dataset_sizes}")
    
    return train_loader, test_loader, dataset_sizes

# トレーニング関数
def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, num_epochs):
    since = time.time()
    
    # 結果を記録
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        # 各エポックにはトレーニングフェーズと検証フェーズがある
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # モデルをトレーニングモードに設定
                dataloader = dataloaders['train']
            else:
                model.eval()   # モデルを評価モードに設定
                dataloader = dataloaders['val']
            
            running_loss = 0.0
            running_corrects = 0
            
            # データのバッチ処理
            for inputs, labels in tqdm(dataloader, desc=phase):
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # 勾配パラメータをゼロにする
                optimizer.zero_grad()
                
                # 順伝播
                # トレーニングフェーズの場合のみ勾配を記録
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    # トレーニングフェーズの場合は逆伝播+最適化
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                # 統計
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            # エポックごとの損失と精度
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            
            # 履歴に保存
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item())
                scheduler.step(epoch_loss)  # 学習率の更新
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())
            
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            # モデルのコピーを保存
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                
                # 最良のモデルを保存
                os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
                torch.save(best_model_wts, model_save_path)
                print(f'Model saved to {model_save_path}')
        
        print()
    
    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:.4f}')
    
    # 学習曲線のプロット
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Validation')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train')
    plt.plot(history['val_acc'], label='Validation')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.close()
    
    # 最良の重みをロード
    model.load_state_dict(best_model_wts)
    return model, history

# メイン関数
def main():
    # データローダーのセットアップ
    train_loader, test_loader, dataset_sizes = setup_dataloaders()
    
    dataloaders = {
        'train': train_loader,
        'val': test_loader
    }
    
    # モデルのインスタンス化
    model = EmotionCNN(num_classes=num_classes).to(device)
    
    # 損失関数と最適化手法の定義
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 学習率スケジューラー
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=5, verbose=True
    )
    
    # モデルのトレーニング
    model, history = train_model(
        model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, num_epochs
    )

if __name__ == '__main__':
    main()