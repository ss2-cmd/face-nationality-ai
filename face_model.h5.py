import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

# 1. 데이터셋 경로 및 이미지 크기 설정
data_dir = 'C:/Users/samsung/Desktop/face_nationality_ai/face_nationality_ai/data/train'
img_size = (128, 128)  # 64->128로 확대

# 2. 데이터 불러오기 및 전처리
X = []
y = []
label_names = sorted([
    d for d in os.listdir(data_dir)
    if os.path.isdir(os.path.join(data_dir, d))
])
label_map = {label: idx for idx, label in enumerate(label_names)}

failed_count = 0
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
            failed_count += 1
            print(f"⚠ 이미지 로딩 실패 ({failed_count}): {img_path}, 이유: {e}")

X = np.array(X)
y = to_categorical(y, num_classes=len(label_names))

# 3. 데이터 분리 (학습 80%, 검증 20%)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. 모델 구성
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(img_size[0], img_size[1], 3)),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(len(label_names), activation='softmax')
])

# 5. 컴파일 및 콜백
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# 6. 모델 학습
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val), callbacks=[early_stop])

# 7. 모델 저장
model.save("face_model.h5")
print("✅ 학습 완료! 모델이 face_model.h5로 저장되었습니다.")

