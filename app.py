import cv2 # type: ignore
import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
from tensorflow.keras import layers, models # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore

# Parameters (Fixed typo from BACH_SIZE to BATCH_SIZE)
IMG_SIZE = 128
BATCH_SIZE = 32  # Corrected spelling
EPOCHS = 50  # Reduced from 500 to prevent overfitting

# Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2,
    fill_mode='nearest',
    height_shift_range=0.2,
    width_shift_range=0.2,
    rotation_range=20
)

test_datagen = ImageDataGenerator(rescale=1./255)

# Dataset Setup for Helmet Detection
train_generator = train_datagen.flow_from_directory(
    'dataset/train',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,  # Corrected spelling
    class_mode='binary',
    classes=['no_helmet', 'helmet']  # Alphabetical order matters!
)

test_generator = test_datagen.flow_from_directory(
    'dataset/test',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,  # Corrected spelling
    class_mode='binary',
    classes=['no_helmet', 'helmet']
)

# Enhanced Model Architecture
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    layers.MaxPooling2D((2,2)),
    layers.BatchNormalization(),
    
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Dropout(0.3),
    
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.BatchNormalization(),
    
    layers.Flatten(),
    layers.Dense(256, activation='relu'),  # Reduced from 512
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

# Added early stopping to prevent overfitting
from tensorflow.keras.callbacks import EarlyStopping # type: ignore
early_stop = EarlyStopping(monitor='val_loss', patience=5)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Training with validation
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=test_generator,
    callbacks=[early_stop]
)

# Evaluation
test_loss, test_acc = model.evaluate(test_generator)
print(f'Test accuracy: {test_acc:.2%}')  # Formatted percentage

# Visualization with helmet labels
class_names = ['No Helmet', 'Helmet']  # Updated class names

images, labels = next(test_generator)
predictions = model.predict(images)
predicted_classes = (predictions > 0.5).astype(int)

plt.figure(figsize=(10, 10))
for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.imshow(images[i])
    plt.title(f"Pred: {class_names[predicted_classes[i][0]]}\nTrue: {class_names[int(labels[i])]}")
    plt.axis('off')
plt.show()