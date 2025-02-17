import cv2 # type: ignore 
import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
from tensorflow.keras import layers, models # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore

# Parameters
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 50

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

# Dataset Setup with additional safe classes
train_generator = train_datagen.flow_from_directory(
    'dataset/train',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    classes=['no_helmet', 'helmet', 'unprotected_edge', 'protected_edge', 'unstable_trench', 'protected_trench', 'uneven_surface', 'loose_wires', 'wet_floor', 'misplaced_tools', 'even_surface', 'no_loose_wires', 'dry_floor', 'un_misplaced_tools']
)

test_generator = test_datagen.flow_from_directory(
    'dataset/test',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    classes=['no_helmet', 'helmet', 'unprotected_edge', 'protected_edge', 'unstable_trench', 'protected_trench', 'uneven_surface', 'loose_wires', 'wet_floor', 'misplaced_tools', 'even_surface', 'no_loose_wires', 'dry_floor', 'un_misplaced_tools']
)

# Updated model with 14 output classes
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
    layers.Dense(14, activation='softmax')
])

from tensorflow.keras.callbacks import EarlyStopping # type: ignore
early_stop = EarlyStopping(monitor='val_loss', patience=5)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=test_generator,
    callbacks=[early_stop]
)

test_loss, test_acc = model.evaluate(test_generator)
print(f'Test accuracy: {test_acc:.2%}')

class_names = ['No Helmet', 'Helmet', 'Unprotected Edge', 'Protected Edge', 'Unstable Trench', 'Protected Trench', 'Uneven Surface', 'Loose Wires', 'Wet Floor', 'Misplaced Tools', 'Even Surface', 'No Loose Wires', 'Dry Floor', 'Un-Misplaced Tools']
images, labels = next(test_generator)
predictions = model.predict(images)
predicted_classes = np.argmax(predictions, axis=1)
plt.figure(figsize=(10, 10))
for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.imshow(images[i])
    plt.title(f"Pred: {class_names[predicted_classes[i]]}\nTrue: {class_names[np.argmax(labels[i])]}")
    plt.axis('off')
plt.show()
