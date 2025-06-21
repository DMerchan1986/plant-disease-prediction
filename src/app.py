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
CLASS_NAMES = get_class_names(BASE_DIR / 'data' / 'PlantVillage' / 'train')

# Cargar modelo
model = load_model(MODEL_PATH)

# Cargar logo localmente
logo_path = BASE_DIR / 'src' / 'Logo_FitoScan.png'
logo_img = Image.open(logo_path)

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
    plt.barh([names[i] for i in sorted_idx], [values[i] for i in sorted_idx], color='#2c6f4a')
    plt.xlabel('Confianza')
    plt.title('Distribución de Probabilidades')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    return plt

css = """
body {background: #f0f4f8; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;}
.gradio-container {max-width: 900px; margin: auto; padding: 20px;}
h1 {color: #2c6f4a; font-weight: 700; margin-bottom: 10px;}
h2 {color: #4a4a4a; margin-bottom: 30px;}
.gr-button-primary {background-color: #2c6f4a; border-color: #2c6f4a;}
.gr-button-primary:hover {background-color: #1f4d32; border-color: #1f4d32;}
"""

with gr.Blocks(css=css, title="FitoScan - Clasificador de Enfermedades en Plantas") as app:
    gr.Image(value=logo_img, shape=(150, 150), interactive=False)
    gr.Markdown("# 🌿 FitoScan")
    gr.Markdown("### Clasificador de Enfermedades en Plantas (PlantVillage)")
    gr.Markdown("Sube una imagen de una hoja para diagnosticar posibles enfermedades.")

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="pil", label="Sube una hoja de planta", interactive=True)
            submit_btn = gr.Button("Predecir", variant="primary")
        with gr.Column(scale=1):
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
    app.launch(server_port=7860, share=True)