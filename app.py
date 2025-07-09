import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import os

# 모델 경로 (학습 후 저장된 경로와 동일하게)
model_path = "C:/Users/samsung/Desktop/face_nationality_ai/face_nationality_ai/face_model.h5"

try:
    model = load_model(model_path)
except Exception as e:
    st.error(f"모델 로드 실패: {e}")
    st.stop()

# 데이터 폴더 (폴더명으로 라벨 매핑)
data_dir = "C:/Users/samsung/Desktop/face_nationality_ai/face_nationality_ai/data/train"
label_map = {
    'korean': '한국인 🇰🇷',
    'japanese': '일본인 🇯🇵',
    'american': '미국인 🇺🇸'
}

try:
    folder_names = sorted([f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))])
except Exception as e:
    st.error(f"데이터 폴더 로드 실패: {e}")
    st.stop()

index_to_label = {i: label_map.get(folder.lower(), "알 수 없음") for i, folder in enumerate(folder_names)}

st.set_page_config(page_title="국적 예측 AI", page_icon="🌍")
st.markdown("<h1 style='text-align: center;'>🌟 얼굴로 국적 예측 AI 🌟</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>얼굴 이미지를 분석해 국적을 예측합니다! 재미로 즐겨주세요 😄</p>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("📸 얼굴 이미지를 업로드하세요", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="🖼️ 업로드한 이미지", use_column_width=True)

        img_array = np.array(image.resize((128, 128))) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        pred = model.predict(img_array)
        pred_index = int(np.argmax(pred))
        confidence = float(np.max(pred))

        st.markdown("---")
        st.markdown(f"<h2 style='text-align: center;'>🔍 예측 결과</h2>", unsafe_allow_html=True)
        st.markdown(f"<h3 style='text-align: center;'>{index_to_label.get(pred_index, '알 수 없음')}</h3>", unsafe_allow_html=True)
        st.write(f"📊 신뢰도: {confidence * 100:.2f}%")

    except Exception as e:
        st.error(f"❌ 오류 발생: {e}")

