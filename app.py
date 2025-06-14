import streamlit as st
import cv2
import numpy as np
from PIL import Image
import torch
import os
import matplotlib.pyplot as plt

from emotion_model import EmotionCNN
import utils

# ページ設定
st.set_page_config(page_title="表情感情認識アプリ", layout="wide")

# タイトルと説明
st.title("表情感情認識アプリ")
st.write("このアプリは人の顔から感情を認識します。画像をアップロードするか、カメラを使用してください。")

# 初期化
@st.cache_resource
def load_model():
    model = EmotionCNN()
    if os.path.exists("models/emotion_model.pt"):
        model.load_state_dict(torch.load("models/emotion_model.pt", map_location=torch.device('cpu')))
        model.eval()
    else:
        st.error("モデルファイルが見つかりません。先にモデルをトレーニングしてください。")
    return model

# モデルのロード
model = load_model()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 顔検出器のロード
face_cascade = utils.init_face_detector()

# 画像処理関数
def process_image(image):
    # PIL画像をnumpy配列に変換
    if isinstance(image, Image.Image):
        image_np = np.array(image)
    else:
        image_np = image
    
    # BGR形式に変換（OpenCV用）
    if len(image_np.shape) == 3 and image_np.shape[2] == 3:
        image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    else:
        image_cv = image_np
    
    # 顔検出
    faces = utils.detect_faces(image_cv, face_cascade)
    
    results = []
    for (x, y, w, h) in faces:
        # 顔部分を切り出し
        face_img = image_cv[y:y+h, x:x+w]
        
        # 前処理
        face_tensor = utils.preprocess_face(face_img)
        
        # 感情予測
        emotion, probabilities = utils.predict_emotion(face_tensor, model, device)
        
        # 結果を保存
        results.append({
            'coords': (x, y, w, h),
            'emotion': emotion,
            'probabilities': probabilities
        })
    
    # 結果を画像に描画
    emotions = [r['emotion'] for r in results]
    annotated_image = utils.draw_results(image_cv, faces, emotions)
    
    # BGR->RGB変換（表示用）
    if len(annotated_image.shape) == 3 and annotated_image.shape[2] == 3:
        annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
    
    return annotated_image, results

# サイドバー - 入力方法の選択
option = st.sidebar.selectbox(
    "画像入力方法を選択",
    ["画像をアップロード", "カメラを使用"]
)

# メイン領域を2列に分割
col1, col2 = st.columns(2)

with col1:
    if option == "画像をアップロード":
        uploaded_file = st.file_uploader("顔写真をアップロード", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="アップロードされた画像", width=400)
            
            # 処理ボタン
            if st.button("感情分析"):
                processed_img, results = process_image(image)
                
                if len(results) > 0:
                    st.image(processed_img, caption="検出結果", width=400)
                    
                    with col2:
                        for i, result in enumerate(results):
                            st.subheader(f"顔 #{i+1} - 感情: {result['emotion']}")
                            
                            # 確率のプロット
                            fig = utils.plot_emotion_probabilities(result['probabilities'])
                            st.pyplot(fig)
                else:
                    st.warning("画像から顔を検出できませんでした。別の画像を試してください。")
    
    else:  # カメラを使用
        st.write("カメラを使用して感情を分析します。")
        camera_image = st.camera_input("カメラ")
        
        if camera_image is not None:
            image = Image.open(camera_image)
            
            # 処理ボタン
            if st.button("感情分析"):
                processed_img, results = process_image(image)
                
                if len(results) > 0:
                    st.image(processed_img, caption="検出結果", width=400)
                    
                    with col2:
                        for i, result in enumerate(results):
                            st.subheader(f"顔 #{i+1} - 感情: {result['emotion']}")
                            
                            # 確率のプロット
                            fig = utils.plot_emotion_probabilities(result['probabilities'])
                            st.pyplot(fig)
                else:
                    st.warning("画像から顔を検出できませんでした。別の画像を試してください。")

# アプリについての追加情報
st.sidebar.markdown("---")
st.sidebar.subheader("アプリについて")
st.sidebar.write("このアプリはFER2013データセットで訓練された深層学習モデルを使用しています。7つの基本感情（怒り、嫌悪、恐怖、幸福、中立、悲しみ、驚き）を認識します。")