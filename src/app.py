import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import base64
import io
import os

# Cargar el modelo (ruta corregida)
model = tf.keras.models.load_model('output/model.keras')

# Lista de clases (ajusta según tu modelo)
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

def format_class_name(class_name):
    """Formatear nombre de clase para mostrar"""
    formatted = class_name.replace('___', ' - ').replace('_', ' ')
    return formatted.title()

def get_plant_type(class_name):
    """Extraer tipo de planta del nombre de clase"""
    return class_name.split('___')[0].replace('_', ' ').title()

def get_disease_name(class_name):
    """Extraer nombre de enfermedad del nombre de clase"""
    parts = class_name.split('___')
    if len(parts) > 1:
        disease = parts[1].replace('_', ' ')
        return disease.title() if disease.lower() != 'healthy' else 'Saludable'
    return 'Desconocido'

def create_prediction_chart(predictions, class_names, top_n=5):
    """Crear gráfico de barras con las mejores predicciones"""
    top_indices = np.argsort(predictions[0])[-top_n:][::-1]
    top_probs = predictions[0][top_indices]
    top_classes = [format_class_name(class_names[i]) for i in top_indices]
    
    # Colores más contrastantes para tema claro
    colors = ['#2E7D32' if i == 0 else '#1976D2' for i in range(len(top_classes))]
    
    fig = go.Figure(data=[
        go.Bar(
            y=top_classes,
            x=top_probs * 100,
            orientation='h',
            marker_color=colors,
            text=[f'{prob:.1f}%' for prob in top_probs * 100],
            textposition='auto',
            textfont=dict(color='white', size=12, family="Arial Black"),
        )
    ])
    
    fig.update_layout(
        title={
            'text': '<b>Distribución de Probabilidades</b>',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18, 'color': '#1B5E20', 'family': 'Arial Black'}
        },
        xaxis_title='<b>Confianza (%)</b>',
        yaxis_title='<b>Diagnóstico</b>',
        template='plotly_white',
        height=400,
        margin=dict(l=20, r=20, t=60, b=20),
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(color='#2E2E2E', size=11, family='Arial'),
        xaxis=dict(
            gridcolor='#E0E0E0',
            range=[0, 100],
            title_font=dict(color='#1B5E20', size=14)
        ),
        yaxis=dict(
            gridcolor='#E0E0E0',
            title_font=dict(color='#1B5E20', size=14)
        )
    )
    
    return fig

def create_confidence_gauge(confidence):
    """Crear medidor de confianza"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = confidence,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {
            'text': "<b>Nivel de Confianza</b>", 
            'font': {'color': '#1B5E20', 'size': 16, 'family': 'Arial Black'}
        },
        number = {'font': {'color': '#2E7D32', 'size': 24, 'family': 'Arial Black'}},
        gauge = {
            'axis': {
                'range': [None, 100], 
                'tickcolor': '#2E2E2E',
                'tickfont': {'color': '#2E2E2E', 'size': 12}
            },
            'bar': {'color': "#2E7D32", 'thickness': 0.8},
            'steps': [
                {'range': [0, 50], 'color': "#FFCDD2"},
                {'range': [50, 80], 'color': "#FFF9C4"},
                {'range': [80, 100], 'color': "#C8E6C9"}
            ],
            'threshold': {
                'line': {'color': "#D32F2F", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(
        template='plotly_white',
        paper_bgcolor='white',
        plot_bgcolor='white',
        font={'color': '#2E2E2E'},
        height=300,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig

def predict_disease(image):
    """Función principal de predicción"""
    if image is None:
        return None, None, "⚠️ **Por favor, sube una imagen primero.**"
    
    try:
        # Preprocesar imagen
        img = Image.fromarray(image)
        img = img.resize((128, 128))  # Cambié de 224 a 128
        img_array = np.array(img)
        
        # Normalizar si es necesario
        if img_array.max() > 1:
            img_array = img_array / 255.0
        
        img_array = np.expand_dims(img_array, axis=0)
        
        # Realizar predicción
        predictions = model.predict(img_array)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx]) * 100
        
        # Obtener información de la predicción
        predicted_class = class_names[predicted_class_idx]
        plant_type = get_plant_type(predicted_class)
        disease_name = get_disease_name(predicted_class)
        
        # Crear gráficos
        prob_chart = create_prediction_chart(predictions, class_names)
        confidence_gauge = create_confidence_gauge(confidence)
        
        # Crear mensaje de resultado con mejor formato
        if 'healthy' in predicted_class.lower():
            result_message = f"""
            <div style='background: linear-gradient(135deg, #E8F5E8 0%, #C8E6C9 100%); padding: 25px; border-radius: 15px; border-left: 5px solid #2E7D32; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
            <h2 style='color: #1B5E20; margin: 0 0 15px 0; font-family: Arial Black;'>
            🌱 <strong>{plant_type}</strong> - ✅ <strong>SALUDABLE</strong>
            </h2>
            <p style='color: #2E7D32; font-size: 18px; margin: 0; font-weight: bold;'>
            🎯 Confianza: {confidence:.1f}%
            </p>
            <p style='color: #388E3C; margin: 10px 0 0 0; font-style: italic;'>
            ¡Excelente! Tu planta se ve en perfecto estado de salud.
            </p>
            </div>
            """
        else:
            result_message = f"""
            <div style='background: linear-gradient(135deg, #FFF3E0 0%, #FFE0B2 100%); padding: 25px; border-radius: 15px; border-left: 5px solid #F57C00; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
            <h2 style='color: #E65100; margin: 0 0 15px 0; font-family: Arial Black;'>
            🍃 <strong>{plant_type}</strong>
            </h2>
            <h3 style='color: #FF6F00; margin: 0 0 15px 0;'>
            🔍 <strong>Enfermedad detectada:</strong> {disease_name} ⚠️
            </h3>
            <p style='color: #F57C00; font-size: 18px; margin: 0; font-weight: bold;'>
            🎯 Confianza: {confidence:.1f}%
            </p>
            <p style='color: #FF8F00; margin: 10px 0 0 0; font-style: italic;'>
            Se recomienda consultar con un especialista en fitopatología.
            </p>
            </div>
            """
        
        return prob_chart, confidence_gauge, result_message
    
    except Exception as e:
        error_msg = f"""
        <div style='background: #FFEBEE; padding: 20px; border-radius: 10px; border-left: 5px solid #F44336;'>
        <h3 style='color: #C62828; margin: 0;'>❌ Error en la predicción</h3>
        <p style='color: #D32F2F; margin: 10px 0 0 0;'>{str(e)}</p>
        </div>
        """
        return None, None, error_msg

# Crear la interfaz de Gradio
def create_interface():
    custom_css = '''
    img.logo {
        max-height: 200px;
        max-width: 100%;
        height: auto;
        width: auto;
        margin: 0 auto;
        display: block;
    }

    @media (max-width: 600px) {
        img.logo {
            max-height: 200px;
        }
    }
    '''

    with gr.Blocks(
        theme=gr.themes.Soft(primary_hue="green", secondary_hue="blue"),
        css=custom_css,
        title="🌱 FitoScan - Detector de Enfermedades en Plantas"
    ) as app:

        def get_logo_base64():
            possible_paths = [
                "src/Logo_FitoScan.png",
                "Logo_FitoScan.png",
                "assets/Logo_FitoScan.png",
                "images/Logo_FitoScan.png"
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    with open(path, "rb") as img_file:
                        return base64.b64encode(img_file.read()).decode()
            return None

        logo_b64 = get_logo_base64()

        with gr.Row():
            with gr.Column(scale=1):
                if logo_b64:
                    gr.HTML(f'''
                    <div style="text-align: center; margin: 0; padding: 0;">
                        <img class="logo" src="data:image/png;base64,{logo_b64}" alt="Logo FitoScan">
                    </div>
                    ''')
                else:
                    gr.HTML('<div style="text-align: center; font-size: 6rem; margin: 0; padding: 0;">🌱</div>')
            with gr.Column(scale=4):
                gr.HTML("""
                <div style='text-align: center; padding: 1rem;'>
                    <h1 style='color: #1B5E20; font-size: 3rem; margin: 0; font-family: Arial Black; text-shadow: 2px 2px 4px rgba(0,0,0,0.1);'>
                        FitoScan
                    </h1>
                    <h2 style='color: #2E7D32; margin: 0.5rem 0; font-size: 1.5rem; font-weight: normal;'>
                        Detector Inteligente de Enfermedades en Plantas 🔬
                    </h2>
                    <p style='color: #388E3C; font-size: 1.1rem; margin: 0.5rem 0;'>
                        Tecnología de IA avanzada para el diagnóstico fitosanitario
                    </p>
                </div>
                """)

        image_input = gr.Image(
            label="Arrastra aquí la imagen o haz clic para seleccionar",
            type="numpy",
            height=400
        )

        predict_btn = gr.Button(
            "🔍 Analizar Planta", 
            variant="primary", 
            size="lg",
            elem_classes="analyze-button"
        )

        gr.HTML("""
        <div class="instructions-box">
            <h4 style='color: #4A148C; margin: 0 0 1rem 0; font-family: Arial Black;'>
                📋 Instrucciones de Uso:
            </h4>
            <ul style='color: #6A1B9A; margin: 0; line-height: 1.6;'>
                <li><strong>Calidad:</strong> Usa imágenes claras y bien iluminadas</li>
                <li><strong>Enfoque:</strong> La hoja debe ocupar la mayor parte de la imagen</li>
                <li><strong>Formato:</strong> JPG, PNG o WEBP</li>
                <li><strong>Resolución:</strong> Mínimo 128x128 píxeles</li>
            </ul>
        </div>
        """)

        with gr.Column(scale=2, min_width=400):
            title_logo = f'<img src="data:image/png;base64,{logo_b64}" style="height: 20px; width: auto; margin-right: 8px;">' if logo_b64 else '📊'
            gr.HTML(f"""
            <div class="results-section">
                <h3 style='color: #1976D2; text-align: center; margin-bottom: 1rem; font-family: Arial Black; display: flex; align-items: center; justify-content: center; gap: 0.5rem;'>
                    {title_logo} Resultados del Análisis
                </h3>
            </div>
            """)

        result_html = gr.HTML("""
        <div style='text-align: center; color: #666; padding: 2rem;'>
            <h4>🕒 Esperando imagen para analizar...</h4>
            <p>Sube una imagen de una hoja para comenzar el diagnóstico</p>
        </div>
        """)

        with gr.Row():
            confidence_plot = gr.Plot(label="Medidor de Confianza")
            probability_plot = gr.Plot(label="Distribución de Probabilidades")

        gr.HTML("""
        <div class="plants-box">
            <h4 style='color: #0D47A1; margin: 0 0 1rem 0; font-family: Arial Black;'>
                🌿 Plantas y Enfermedades Detectables:
            </h4>
            <div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; color: #1565C0;'>
                <div><strong>🍎 Manzana:</strong> Sarna, Podredumbre negra, Roya</div>
                <div><strong>🫐 Arándano:</strong> Estado saludable</div>
                <div><strong>🍒 Cereza:</strong> Oídio, Estado saludable</div>
                <div><strong>🌽 Maíz:</strong> Manchas foliares, Roya, Tizón</div>
                <div><strong>🍇 Uva:</strong> Podredumbre negra, Esca, Tizón foliar</div>
                <div><strong>🍊 Naranja:</strong> Huanglongbing (HLB)</div>
                <div><strong>🍑 Durazno:</strong> Mancha bacteriana</div>
                <div><strong>🌶️ Pimiento:</strong> Mancha bacteriana</div>
                <div><strong>🥔 Papa:</strong> Tizón temprano y tardío</div>
                <div><strong>🍓 Fresa:</strong> Quemadura foliar</div>
                <div><strong>🍅 Tomate:</strong> Múltiples enfermedades</div>
                <div><strong>🫛 Soja:</strong> Estado saludable</div>
            </div>
        </div>
        """)

        # Conectar eventos
        predict_btn.click(
            fn=predict_disease,
            inputs=[image_input],
            outputs=[probability_plot, confidence_plot, result_html]
        )

        image_input.change(
            fn=predict_disease,
            inputs=[image_input],
            outputs=[probability_plot, confidence_plot, result_html]
        )

    return app

# Ejecutar la aplicación
if __name__ == "__main__":
    app = create_interface()
    app.launch(
        server_name="127.0.0.1",
        server_port=8080,
        share=True,
        inbrowser=True,
        show_error=True
    )