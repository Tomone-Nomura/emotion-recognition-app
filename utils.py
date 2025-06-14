import cv2
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# 感情ラベル
emotion_labels = ['怒り', '嫌悪', '恐怖', '幸福', '中立', '悲しみ', '驚き']

# 顔検出器の初期化
def init_face_detector():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    return face_cascade

# 画像から顔を検出
def detect_faces(image, face_cascade):
    # グレースケールに変換
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # 顔検出
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    return faces

# 検出された顔を前処理
def preprocess_face(face_img):
    # リサイズ
    face_img = cv2.resize(face_img, (48, 48))
    
    # チャンネルの確認と調整
    if len(face_img.shape) == 2:  # グレースケールの場合
        face_img = cv2.cvtColor(face_img, cv2.COLOR_GRAY2RGB)
    elif face_img.shape[2] == 4:  # アルファチャンネルがある場合
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGRA2RGB)
    elif face_img.shape[2] == 3:  # BGRの場合
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    
    # テンソル変換
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    face_tensor = transform(Image.fromarray(face_img))
    return face_tensor.unsqueeze(0)  # バッチ次元を追加

# 顔の感情を予測
def predict_emotion(face_tensor, model, device):
    model.eval()
    with torch.no_grad():
        face_tensor = face_tensor.to(device)
        outputs = model(face_tensor)
        probabilities = F.softmax(outputs, dim=1)[0]
        prediction = outputs.argmax(dim=1, keepdim=True).item()
    
    emotion = emotion_labels[prediction]
    probs = probabilities.cpu().numpy()
    
    return emotion, probs

# 結果を画像に描画
def draw_results(image, faces, emotions):
    result_img = image.copy()
    
    for (x, y, w, h), emotion in zip(faces, emotions):
        # 顔の周りに矩形を描画
        cv2.rectangle(result_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # 感情ラベルを表示
        cv2.putText(result_img, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    return result_img

# 確率分布をプロットする関数
def plot_emotion_probabilities(probabilities):
    fig, ax = plt.subplots(figsize=(8, 4))
    y_pos = np.arange(len(emotion_labels))
    
    ax.barh(y_pos, probabilities, align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(emotion_labels)
    ax.invert_yaxis()  # ラベルを上から下に表示
    ax.set_xlabel('確率')
    ax.set_title('感情予測確率')
    
    return fig