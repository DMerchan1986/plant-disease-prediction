import gradio as gr
import numpy as np
import tensorflow as tf
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
from utils import load_model, preprocess_image, get_class_names

# Configuración
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / 'output' / 'model.keras'
CLASS_NAMES = get_class_names(BASE_DIR / 'data' / 'PlantVillage' / 'train')  # Nombres desde train

# Cargar modelo
model = load_model(MODEL_PATH)

def predict_disease(img):
    try:
        img_processed = preprocess_image(img, target_size=(128, 128))
        preds = model.predict(np.expand_dims(img_processed, axis=0))[0]
        return {CLASS_NAMES[i]: float(preds[i]) for i in range(len(CLASS_NAMES))}
    except Exception as e:
        return {"Error": str(e)}

def create_confidence_plot(pred_dict):
    plt.figure(figsize=(10, 6))
    names = list(pred_dict.keys())
    values = list(pred_dict.values())
    sorted_idx = np.argsort(values)[::-1]
    plt.barh([names[i] for i in sorted_idx], [values[i] for i in sorted_idx], color='skyblue')
    plt.xlabel('Confianza')
    plt.title('Resultados de Predicción')
    return plt

# Interfaz Gradio
with gr.Blocks(title="Clasificador de PlantVillage") as app:
    gr.Markdown("## 🌿 Clasificador de Enfermedades en Plantas (PlantVillage)")
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Sube una hoja de planta")
            submit_btn = gr.Button("Predecir", variant="primary")
        with gr.Column():
            label_output = gr.Label(label="Diagnóstico", num_top_classes=3)
            plot_output = gr.Plot(label="Distribución de Probabilidades")
    
    submit_btn.click(
        fn=predict_disease,
        inputs=image_input,
        outputs=label_output
    ).then(
        fn=create_confidence_plot,
        inputs=label_output,
        outputs=plot_output
    )

if __name__ == "__main__":
    app.launch(server_port=7860)