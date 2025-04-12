import os
import warnings
import random
import time
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense, Activation, BatchNormalization
from tensorflow.keras.optimizers import Adam

warnings.filterwarnings("ignore")
print("✅ All Libraries Imported")

# Load image paths and labels
image_dir = Path(r'D:\3. SELF STUDY\5. PROJECT\DIABATIC_RETENOPHLYA_DETECTION\DATASET_DIABATIC_RETENOPHLYA')
filepaths = list(image_dir.glob(r'**/*.png'))
labels = [os.path.split(os.path.split(x)[0])[1] for x in filepaths]

image_df = pd.DataFrame({
    'Filepath': [str(fp) for fp in filepaths],
    'Label': labels
}).sample(frac=1).reset_index(drop=True)

# Convert to YCbCr and save to preprocessed folder
preprocessed_image_dir = image_dir / "preprocessed_image_dir"
preprocessed_image_dir.mkdir(parents=True, exist_ok=True)

for index, row in image_df.iterrows():
    image_path = row['Filepath']
    image = cv2.imread(str(image_path))
    if image is None:
        continue
    ycbcr_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    new_path = preprocessed_image_dir / Path(image_path).name
    cv2.imwrite(str(new_path), ycbcr_image)
    image_df.at[index, 'Filepath'] = str(new_path)

# Label to numeric level
label_map = {'No_DR': 0, 'Mild': 1, 'Moderate': 2, 'Severe': 3, 'Proliferate_DR': 4}
image_df['Level'] = image_df['Label'].map(label_map)

# Prepare image arrays
X = []
for path in image_df['Filepath']:
    image = cv2.imread(path)
    if image is None:
        continue
    X.append(image)
X = np.array(X)
Y = image_df['Level'].values

# Split dataset
x_train, x_temp, y_train, y_temp = train_test_split(X, Y, test_size=0.3, random_state=42)
x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.3, random_state=42)
print(f"Train: {len(x_train)}, Val: {len(x_val)}, Test: {len(x_test)}")

# Data augmentation
idg_train = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
idg_test = ImageDataGenerator(rescale=1./255)

idg_train.fit(x_train)
idg_test.fit(x_test)

# Callbacks
early_stop = EarlyStopping(monitor='loss', patience=15, verbose=1, restore_best_weights=True)
checkpoint = ModelCheckpoint("best_model.h5", monitor="val_accuracy", save_best_only=True, mode="max")

# Build ResNet50 Model
resnet50 = ResNet50(input_shape=(224, 224, 3), weights='imagenet', include_top=False)
x = Flatten()(resnet50.output)
x = Dense(64, kernel_initializer='he_uniform')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Dense(128, kernel_initializer='he_uniform')(x)
x = Activation('relu')(x)
x = Dense(256, kernel_initializer='he_uniform')(x)
x = Activation('relu')(x)
x = Dense(128, kernel_initializer='he_uniform')(x)
x = Activation('relu')(x)
x = Dense(64, kernel_initializer='he_uniform')(x)
x = Activation('relu')(x)
output = Dense(5, activation='softmax')(x)

model = Model(inputs=resnet50.input, outputs=output)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    x_train / 255.0,
    y_train,
    validation_data=(x_val / 255.0, y_val),
    epochs=10,
    batch_size=32,
    callbacks=[early_stop, checkpoint]
)

# Evaluate
train_acc = model.evaluate(x_train / 255.0, y_train, verbose=0)[1]
val_acc = model.evaluate(x_val / 255.0, y_val, verbose=0)[1]
test_acc = model.evaluate(x_test / 255.0, y_test, verbose=0)[1]

print(f"✅ Train Accuracy: {train_acc*100:.2f}%")
print(f"✅ Validation Accuracy: {val_acc*100:.2f}%")
print(f"✅ Test Accuracy: {test_acc*100:.2f}%")

# Predict and save model
y_pred = np.argmax(model.predict(x_test / 255.0), axis=1)
print("Sample Predictions:", y_pred[:10])

model.save("DR_resnet50.h5")
print("✅ Model saved as DR_resnet50.h5")
