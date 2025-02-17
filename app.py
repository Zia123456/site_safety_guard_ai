import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping

# Parameters
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 50

classes = [
    'no_helmet', 'helmet', 'unprotected_edge', 'protected_edge',
    'unstable_trench', 'protected_trench', 'uneven_surface', 'loose_wires',
    'wet_floor', 'misplaced_tools', 'even_surface', 'no_loose_wires',
    'dry_floor', 'un_misplaced_tools'
]

train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    'dataset/train',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    classes=classes
)

test_generator = test_datagen.flow_from_directory(
    'dataset/test',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    classes=classes
)

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
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(len(classes), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_loss', patience=5)

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=test_generator,
    callbacks=[early_stop]
)

images, labels = next(test_generator)
predictions = model.predict(images)

plt.figure(figsize=(20, 20))
num_images = len(images)
cols = 5
rows = (num_images // cols) + (1 if num_images % cols != 0 else 0)

for i in range(num_images):
    plt.subplot(rows, cols, i+1)
    plt.imshow(images[i])
    plt.axis('off')
    plt.subplots_adjust(right=0.8)
    plt.text(1.05, 0.5, "\n".join([f"{classes[j]}: {predictions[i][j]*100:.2f}%" for j in range(len(classes))]), fontsize=4, va='center', ha='left', transform=plt.gca().transAxes)

plt.tight_layout()
plt.show()
