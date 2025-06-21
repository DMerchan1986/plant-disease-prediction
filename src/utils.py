import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from pathlib import Path
from PIL import Image

def preprocess_image(img, target_size=(128, 128)):
    """Preprocesa una imagen para el modelo."""
    if isinstance(img, Image.Image):
        img = img.resize(target_size)
        img = np.array(img)
    img = img / 255.0  # Normalización
    if len(img.shape) == 2:
        img = np.stack((img,)*3, axis=-1)  # Convertir a RGB si es escala de grises
    return img

def get_class_names(data_dir):
    """Obtiene los nombres de las clases desde el directorio."""
    return sorted([d.name for d in Path(data_dir).iterdir() if d.is_dir()])

def save_model(model, path):
    """Guarda el modelo en formato .keras."""
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(path)
    print(f"Modelo guardado en: {path}")

def load_model(path):
    """Carga un modelo .keras."""
    return tf.keras.models.load_model(path)

def plot_training_history(history, save_path=None):
    """Grafica precisión y pérdida del entrenamiento."""
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Precisión')
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Pérdida')
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_confusion_matrix(generator, model, save_path=None):
    """Genera matriz de confusión."""
    y_pred = np.argmax(model.predict(generator), axis=1)
    cm = confusion_matrix(generator.classes, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=generator.class_indices.keys(), yticklabels=generator.class_indices.keys())
    plt.title('Matriz de Confusión')
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_class_accuracy(generator, model, save_path=None):
    """Grafica la precisión por clase y guarda la imagen si se especifica save_path.
    
    Args:
        generator: Generador de datos de Keras
        model: Modelo entrenado
        save_path: Ruta para guardar la gráfica (opcional)
    """
    y_pred = np.argmax(model.predict(generator), axis=1)
    y_true = generator.classes
    class_names = list(generator.class_indices.keys())
    
    # Calcular precisión por clase
    class_acc = {}
    for i, class_name in enumerate(class_names):
        idx = np.where(y_true == i)[0]
        acc = np.mean(y_pred[idx] == i)
        class_acc[class_name] = acc
    
    # Ordenar por precisión
    sorted_acc = sorted(class_acc.items(), key=lambda x: x[1])
    
    # Crear gráfica
    plt.figure(figsize=(12, 8))
    plt.barh([x[0] for x in sorted_acc], [x[1] for x in sorted_acc])
    plt.title('Precisión por Clase')
    plt.xlabel('Precisión')
    plt.xlim([0, 1])
    plt.grid(axis='x')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()