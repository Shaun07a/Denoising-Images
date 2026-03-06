import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Input
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split

# -----------------------------
# Dataset Paths
# -----------------------------
clean_path = "clean_images"
noisy_path = "noisy_images"

clean_images = []
noisy_images = []

image_size = (128,128)

# -----------------------------
# Load Dataset
# -----------------------------
for file in sorted(os.listdir(clean_path)):

    clean_img = load_img(
        os.path.join(clean_path, file),
        color_mode="grayscale",
        target_size=image_size
    )

    noisy_img = load_img(
        os.path.join(noisy_path, file.replace("clean","noisy")),
        color_mode="grayscale",
        target_size=image_size
    )

    clean_images.append(img_to_array(clean_img))
    noisy_images.append(img_to_array(noisy_img))

clean_images = np.array(clean_images) / 255.0
noisy_images = np.array(noisy_images) / 255.0

# -----------------------------
# Train Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    noisy_images,
    clean_images,
    test_size=0.2,
    random_state=42
)

# -----------------------------
# Encoder-Decoder Model
# -----------------------------
input_img = Input(shape=(128,128,1))

# Encoder
x = Conv2D(32,(3,3),activation='relu',padding='same')(input_img)
x = MaxPooling2D((2,2),padding='same')(x)

x = Conv2D(64,(3,3),activation='relu',padding='same')(x)
encoded = MaxPooling2D((2,2),padding='same')(x)

# Decoder
x = Conv2D(64,(3,3),activation='relu',padding='same')(encoded)
x = UpSampling2D((2,2))(x)

x = Conv2D(32,(3,3),activation='relu',padding='same')(x)
x = UpSampling2D((2,2))(x)

decoded = Conv2D(1,(3,3),activation='sigmoid',padding='same')(x)

model = Model(input_img, decoded)

# -----------------------------
# Compile Model
# -----------------------------
model.compile(
    optimizer='adam',
    loss='mse'
)

model.summary()

# -----------------------------
# Train Model (5 Epochs)
# -----------------------------
model.fit(
    X_train,
    y_train,
    epochs=5,
    batch_size=8,
    validation_data=(X_test, y_test)
)

# -----------------------------
# Predict Denoised Images
# -----------------------------
decoded_images = model.predict(X_test)

# -----------------------------
# Display Results
# -----------------------------
n = 5
plt.figure(figsize=(12,6))

for i in range(n):

    plt.subplot(3,n,i+1)
    plt.imshow(X_test[i].reshape(128,128), cmap='gray')
    plt.title("Noisy")
    plt.axis("off")

    plt.subplot(3,n,i+1+n)
    plt.imshow(y_test[i].reshape(128,128), cmap='gray')
    plt.title("Original")
    plt.axis("off")

    plt.subplot(3,n,i+1+2*n)
    plt.imshow(decoded_images[i].reshape(128,128), cmap='gray')
    plt.title("Reconstructed")
    plt.axis("off")

plt.tight_layout()
plt.show()