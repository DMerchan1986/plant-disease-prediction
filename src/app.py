import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image
import base64
import os

# Cargar el modelo
model = tf.keras.models.load_model('model.keras')

# Clases de enfermedades
class_names = [
    'Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Blueberry___healthy',
    'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy',
    'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot',
    'Peach___healthy',
    'Pepper,_bell___Bacterial_spot',
    'Pepper,_bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Raspberry___healthy',
    'Soybean___healthy',
    'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch',
    'Strawberry___healthy',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

def predict_disease(image):
    # Preprocesar la imagen
    img = Image.fromarray(image.astype('uint8'), 'RGB')
    img = img.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    img_array = img_array / 255.0
    
    # Hacer predicción
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = float(np.max(predictions[0]))
    
    return f"Enfermedad: {predicted_class}\nConfianza: {confidence:.2%}"

# Cargar logo
logo_path = "src/Logo_FitoScan.png"
logo_img = None

if os.path.exists(logo_path):
    try:
        with open(logo_path, "rb") as f:
            logo_data = f.read()
            logo_img = base64.b64encode(logo_data).decode()
    except Exception as e:
        print(f"Error cargando logo: {e}")

# Crear interfaz
with gr.Blocks(title="FitoScan - Detector de Enfermedades en Plantas", 
               theme=gr.themes.Soft(primary_hue="green")) as demo:
    
    gr.Markdown("# 🌱 FitoScan - Detector de Enfermedades en Plantas")
    
    with gr.Row():
        with gr.Column(scale=1):
            if logo_img:
                # Versión corregida sin el parámetro 'shape'
                gr.HTML(f'<img src="data:image/png;base64,{logo_img}" style="width: 150px; height: 150px; object-fit: contain;" alt="FitoScan Logo" />')
            else:
                gr.Markdown("### 🌿 FitoScan")
        
        with gr.Column(scale=3):
            gr.Markdown("""
            ### ¡Bienvenido a FitoScan!
            
            Sube una imagen de una hoja de planta y nuestro modelo de inteligencia artificial 
            identificará si tiene alguna enfermedad. Admite más de 35 tipos diferentes de 
            enfermedades en plantas como tomate, manzana, uva, maíz y muchas más.
            """)
    
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(label="📸 Subir imagen de la planta")
            predict_btn = gr.Button("🔍 Analizar Planta", variant="primary", size="lg")
        
        with gr.Column():
            result_output = gr.Textbox(label="🎯 Resultado del Análisis", 
                                     placeholder="Los resultados aparecerán aquí...",
                                     lines=3)
    
    predict_btn.click(fn=predict_disease, inputs=image_input, outputs=result_output)
    
    gr.Markdown("""
    ---
    ### 📋 Instrucciones de uso:
    1. **Subir imagen**: Haz clic en el área de carga y selecciona una imagen clara de la hoja
    2. **Analizar**: Presiona el botón "Analizar Planta" 
    3. **Ver resultado**: El diagnóstico aparecerá con el nivel de confianza
    
    ### 🌱 Plantas compatibles:
    Manzana, Arándano, Cereza, Maíz, Uva, Naranja, Durazno, Pimiento, Papa, Frambuesa, Soja, Calabaza, Fresa, Tomate
    
    **Nota**: Para mejores resultados, usa imágenes claras y bien iluminadas de las hojas.
    """)

if __name__ == "__main__":
    demo.launch(share=True, server_name="0.0.0.0", server_port=7860)