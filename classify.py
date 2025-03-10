import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# File paths
train_csv = "data/fashion-mnist_train.csv"
test_csv = "data/fashion-mnist_test.csv"
model_path = "fashion_mnist_model.h5"

# Ensure dataset files exist
if not os.path.exists(train_csv) or not os.path.exists(test_csv):
    raise FileNotFoundError("⚠️ Dataset files missing! Ensure 'data/' contains the CSV files.")

# Load dataset
train_df = pd.read_csv(train_csv)
test_df = pd.read_csv(test_csv)

# Extract labels & images
train_labels = train_df.iloc[:, 0].values
train_images = train_df.iloc[:, 1:].values
test_labels = test_df.iloc[:, 0].values
test_images = test_df.iloc[:, 1:].values

# Normalize and reshape images
train_images = train_images.reshape(-1, 28, 28, 1) / 255.0
test_images = test_images.reshape(-1, 28, 28, 1) / 255.0

# Load trained model if available
if os.path.exists(model_path):
    print("✅ Loading existing model...")
    model = load_model(model_path)
else:
    print("🚀 Training new model...")

    model = Sequential([
        Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation="relu"),
        Dense(10, activation="softmax")
    ])

    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))

    model.save(model_path)
    print("✅ Model saved!")

# Evaluate accuracy
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"\n🎯 Test Accuracy: {test_acc * 100:.2f}%")

# Function to predict a random image
def predict_random_image():
    index = np.random.randint(0, len(test_images))  # Pick a random image
    img = test_images[index].reshape(28, 28)

    plt.imshow(img, cmap="gray")
    plt.axis("off")
    plt.title(f"Actual: {test_labels[index]}")
    plt.show()

    prediction = model.predict(test_images[index].reshape(1, 28, 28, 1))
    predicted_label = np.argmax(prediction)
    print(f"🔍 Predicted: {predicted_label}")

# Run prediction
predict_random_image()
