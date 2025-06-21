import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import pathlib
from utils import save_model, plot_training_history, plot_confusion_matrix, plot_class_accuracy

# Configuración
BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'data' / 'PlantVillage'  # Asegúrate de que exista
OUTPUT_DIR = BASE_DIR / 'output'
PLOTS_DIR = OUTPUT_DIR / 'plots'

# Crear directorios
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Hiperparámetros
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 15  # Ajustado para mejor convergencia

# Generadores de datos
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=25,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_generator = train_datagen.flow_from_directory(
    DATA_DIR / 'train',  # Usar solo el conjunto de entrenamiento
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val_generator = train_datagen.flow_from_directory(
    DATA_DIR / 'train',  # Validación también desde train (split)
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

NUM_CLASSES = len(train_generator.class_indices)
print(f"\nNúmero de clases: {NUM_CLASSES}")
print(f"Clases: {list(train_generator.class_indices.keys())}")

# Modelo (CNN mejorada)
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(*IMG_SIZE, 3)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(NUM_CLASSES, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Entrenamiento
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    verbose=1
)

# Guardar modelo y resultados
save_model(model, OUTPUT_DIR / 'model.keras')
plot_training_history(history, PLOTS_DIR / 'training_history.png')
plot_confusion_matrix(val_generator, model, PLOTS_DIR / 'confusion_matrix.png')
plot_class_accuracy(val_generator, model, PLOTS_DIR / 'class_accuracy.png')

print("\n¡Entrenamiento completado!")