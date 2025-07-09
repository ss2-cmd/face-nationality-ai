import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.model_selection import train_test_split

# 1. 데이터 경로 및 이미지 크기
data_dir = 'C:/Users/samsung/Desktop/face_nationality_ai/face_nationality_ai/data/train'
img_size = (128, 128)  # 학습과 예측 모두 같은 크기 사용

# 2. 데이터 로드
X = []
y = []
label_names = sorted([
    d for d in os.listdir(data_dir)
    if os.path.isdir(os.path.join(data_dir, d))
])
label_map = {label: idx for idx, label in enumerate(label_names)}

for label in label_names:
    folder = os.path.join(data_dir, label)
    for img_name in os.listdir(folder):
        img_path = os.path.join(folder, img_name)
        try:
            img = load_img(img_path, target_size=img_size)
            img_array = img_to_array(img) / 255.0
            X.append(img_array)
            y.append(label_map[label])
        except Exception as e:
            print(f"⚠ 이미지 로딩 실패: {img_path} - {e}")

X = np.array(X)
y = to_categorical(y, num_classes=len(label_names))

# 3. 데이터 분할
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. 모델 구성
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(len(label_names), activation='softmax')
])

# 5. 컴파일 및 학습
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=16, validation_data=(X_val, y_val))

# 6. 모델 저장
model.save('C:/Users/samsung/Desktop/face_nationality_ai/face_nationality_ai/face_model.h5')
print("✅ 학습 완료! 모델이 face_model.h5로 저장되었습니다.")
