import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image
import plotly.graph_objects as go
import base64
import os
from datetime import datetime

# Cargar el modelo
try:
    model = tf.keras.models.load_model('output/model.keras')
except:
    print("⚠️ Modelo no encontrado. Asegúrate de que 'output/model.keras' existe.")
    model = None

# Colores de marca FitoScan
COLORS = {
    'primary_green': '#2a7631',
    'secondary_green': '#6daa2e',
    'light_green': '#c3d21f',
    'dark_green': '#1c441f',
    'forest_green': '#205929',
    'gray': '#6e6e6c',
    'white': '#ffff'
}

# Lista de clases del modelo
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

# Diccionario de tratamientos y recomendaciones
treatment_recommendations = {
    'Apple_scab': {
        'treatment': 'Aplicar fungicidas preventivos como Mancozeb o Captan. Podar para mejorar circulación de aire.',
        'prevention': 'Evitar riego por aspersión, recolectar hojas caídas, elegir variedades resistentes.'
    },
    'Black_rot': {
        'treatment': 'Fungicidas sistémicos como Propiconazol. Remover frutos y ramas infectadas.',
        'prevention': 'Poda sanitaria, evitar heridas en la planta, control de humedad.'
    },
    'Powdery_mildew': {
        'treatment': 'Fungicidas como Azufre en polvo o Bicarbonato de potasio. Mejorar ventilación.',
        'prevention': 'Espaciado adecuado entre plantas, evitar exceso de nitrógeno.'
    },
    'Bacterial_spot': {
        'treatment': 'Bactericidas con cobre, streptomicina en casos severos. Remover material infectado.',
        'prevention': 'Evitar riego foliar, rotación de cultivos, semillas certificadas.'
    },
    'Early_blight': {
        'treatment': 'Fungicidas preventivos con Clorotalonil o Mancozeb. Mejorar drenaje.',
        'prevention': 'Rotación de cultivos, mulching, evitar estrés hídrico.'
    },
    'Late_blight': {
        'treatment': 'Fungicidas sistémicos urgente. Metalaxil + Mancozeb. Destruir plantas infectadas.',
        'prevention': 'Monitoreo constante, evitar humedad nocturna, variedades resistentes.'
    }
}

def format_class_name(class_name):
    formatted = class_name.replace('___', ' - ').replace('_', ' ')
    return formatted.title()

def get_plant_type(class_name):
    plant = class_name.split('___')[0].replace('_', ' ').replace('(', '').replace(')', '')
    return plant.title()

def get_disease_name(class_name):
    parts = class_name.split('___')
    if len(parts) > 1:
        disease = parts[1].replace('_', ' ').replace('(', '').replace(')', '')
        return disease.title() if disease.lower() != 'healthy' else 'Saludable'
    return 'Desconocido'

def get_treatment_info(disease_key):
    for key, info in treatment_recommendations.items():
        if key.lower() in disease_key.lower():
            return info
    return None

def create_prediction_chart(predictions, class_names, top_n=5):
    top_indices = np.argsort(predictions[0])[-top_n:][::-1]
    top_probs = predictions[0][top_indices]
    top_classes = [format_class_name(class_names[i]) for i in top_indices]
    colors = [COLORS['primary_green'] if i == 0 else COLORS['secondary_green'] for i in range(len(top_classes))]
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
            'text': '<b>📊 Análisis de Probabilidades</b>',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18, 'color': COLORS['dark_green'], 'family': 'Arial Black'}
        },
        xaxis_title='<b>Confianza (%)</b>',
        yaxis_title='<b>Diagnóstico</b>',
        template='plotly_white',
        height=400,
        margin=dict(l=20, r=20, t=60, b=20),
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(color=COLORS['gray'], size=11, family='Arial'),
        xaxis=dict(
            gridcolor='#E0E0E0',
            range=[0, 100],
            title_font=dict(color=COLORS['primary_green'], size=14)
        ),
        yaxis=dict(
            gridcolor='#E0E0E0',
            title_font=dict(color=COLORS['primary_green'], size=14)
        )
    )
    return fig

def create_confidence_gauge(confidence):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = confidence,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {
            'text': "<b>🎯 Nivel de Confianza</b>", 
            'font': {'color': COLORS['dark_green'], 'size': 16, 'family': 'Arial Black'}
        },
        number = {'font': {'color': COLORS['primary_green'], 'size': 24, 'family': 'Arial Black'}},
        delta = {'reference': 80, 'increasing': {'color': COLORS['primary_green']}},
        gauge = {
            'axis': {
                'range': [None, 100], 
                'tickcolor': COLORS['gray'],
                'tickfont': {'color': COLORS['gray'], 'size': 12}
            },
            'bar': {'color': COLORS['primary_green'], 'thickness': 0.8},
            'steps': [
                {'range': [0, 50], 'color': "#FFCDD2"},
                {'range': [50, 80], 'color': "#FFF9C4"},
                {'range': [80, 100], 'color': "#C8E6C9"}
            ],
            'threshold': {
                'line': {'color': COLORS['secondary_green'], 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    fig.update_layout(
        template='plotly_white',
        paper_bgcolor='white',
        plot_bgcolor='white',
        font={'color': COLORS['gray']},
        height=300,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig

def create_plant_health_summary(plant_type, disease_name, confidence):
    is_healthy = 'saludable' in disease_name.lower()
    fig = go.Figure()
    colors = [COLORS['primary_green'], COLORS['secondary_green'], COLORS['light_green']] if is_healthy else ['#FF6B6B', '#FF8E53', '#FF6B6B']
    fig.add_trace(go.Scatter(
        x=[1, 2, 3],
        y=[1, 1, 1],
        mode='markers',
        marker=dict(
            size=[60, 50, 40],
            color=colors,
            opacity=0.7
        ),
        text=[f'🌿 {plant_type}', f'🔍 {disease_name}', f'🎯 {confidence:.1f}%'],
        textposition='middle center',
        textfont=dict(size=12, color='white', family='Arial Black'),
        showlegend=False
    ))
    fig.update_layout(
        title={
            'text': f'<b>📋 Resumen del Diagnóstico - {plant_type}</b>',
            'x': 0.5,
            'font': {'size': 16, 'color': COLORS['dark_green']}
        },
        xaxis=dict(showgrid=False, showticklabels=False, range=[0.5, 3.5]),
        yaxis=dict(showgrid=False, showticklabels=False, range=[0.5, 1.5]),
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=200,
        margin=dict(l=20, r=20, t=60, b=20)
    )
    return fig

def predict_disease(image):
    if image is None:
        return None, None, None, "⚠️ **Por favor, sube una imagen primero.**"
    
    if model is None:
        error_msg = "<p style='color:red;'>Error: Modelo no encontrado.</p>"
        return None, None, None, error_msg
    
    try:
        # Aceptar PIL Image o array NumPy
        if isinstance(image, np.ndarray):
            img = Image.fromarray(image)
        else:
            img = image
        
        img = img.convert('RGB')
        img = img.resize((128, 128))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        predictions = model.predict(img_array)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx]) * 100
        
        predicted_class = class_names[predicted_class_idx]
        plant_type = get_plant_type(predicted_class)
        disease_name = get_disease_name(predicted_class)
        
        prob_chart = create_prediction_chart(predictions, class_names)
        confidence_gauge = create_confidence_gauge(confidence)
        health_summary = create_plant_health_summary(plant_type, disease_name, confidence)
        
        treatment_info = get_treatment_info(predicted_class)
        
        if 'healthy' in predicted_class.lower():
            result_message = f"""
            <div style='background: linear-gradient(135deg, #E8F5E8 0%, #C8E6C9 100%); padding: 30px; border-radius: 20px; border-left: 8px solid {COLORS['primary_green']}; box-shadow: 0 6px 12px rgba(0,0,0,0.15); margin: 20px 0;'>
            <div style='display: flex; align-items: center; gap: 15px; margin-bottom: 20px;'>
            <div style='font-size: 3rem;'>🌱</div>
            <div>
            <h2 style='color: {COLORS['dark_green']}; margin: 0; font-family: Arial Black; font-size: 1.8rem;'>
            {plant_type} - ✅ SALUDABLE
            </h2>
            <p style='color: {COLORS['primary_green']}; margin: 5px 0 0 0; font-size: 1.1rem; font-weight: bold;'>
            🎯 Confianza del diagnóstico: {confidence:.1f}%
            </p>
            </div>
            </div>
            <div style='background: rgba(255,255,255,0.7); padding: 15px; border-radius: 10px; border-left: 4px solid {COLORS['secondary_green']};'>
            <h4 style='color: {COLORS['primary_green']}; margin: 0 0 10px 0;'>🎉 ¡Excelentes noticias!</h4>
            <p style='color: {COLORS['forest_green']}; margin: 0; line-height: 1.6;'>
            Tu planta presenta un estado de salud óptimo. Continúa con los cuidados actuales para mantener este excelente estado fitosanitario.
            </p>
            </div>
            <div style='margin-top: 15px; padding: 10px; background: rgba(255,255,255,0.5); border-radius: 8px;'>
            <small style='color: {COLORS['gray']}; font-style: italic;'>
            📅 Análisis realizado: {datetime.now().strftime("%d/%m/%Y %H:%M")} | 🔬 FitoScan AI v2.0
            </small>
            </div>
            </div>
            """
        else:
            treatment_section = ""
            if treatment_info:
                treatment_section = f"""
                <div style='background: rgba(255,255,255,0.9); padding: 20px; border-radius: 12px; border-left: 4px solid #FF8A65; margin-top: 20px;'>
                <h4 style='color: #D84315; margin: 0 0 15px 0; display: flex; align-items: center; gap: 8px;'>
                💊 Tratamiento Recomendado
                </h4>
                <p style='color: #BF360C; margin: 0 0 15px 0; line-height: 1.6; font-weight: 500;'>
                {treatment_info['treatment']}
                </p>
                <h5 style='color: #D84315; margin: 15px 0 10px 0;'>🛡️ Prevención Futura:</h5>
                <p style='color: #BF360C; margin: 0; line-height: 1.6;'>
                {treatment_info['prevention']}
                </p>
                </div>
                """
            result_message = f"""
            <div style='background: linear-gradient(135deg, #FFF8E1 0%, #FFECB3 100%); padding: 30px; border-radius: 20px; border-left: 8px solid #FF8F00; box-shadow: 0 6px 12px rgba(0,0,0,0.15); margin: 20px 0;'>
            <div style='display: flex; align-items: center; gap: 15px; margin-bottom: 20px;'>
            <div style='font-size: 3rem;'>🚨</div>
            <div>
            <h2 style='color: #E65100; margin: 0; font-family: Arial Black; font-size: 1.8rem;'>
            {plant_type}
            </h2>
            <h3 style='color: #FF6F00; margin: 5px 0; font-size: 1.3rem;'>
            🔍 Enfermedad detectada: {disease_name}
            </h3>
            <p style='color: #F57C00; font-size: 1.1rem; margin: 5px 0 0 0; font-weight: bold;'>
            🎯 Confianza del diagnóstico: {confidence:.1f}%
            </p>
            </div>
            </div>
            <div style='background: rgba(255,255,255,0.8); padding: 15px; border-radius: 10px; border-left: 4px solid #FF8F00;'>
            <h4 style='color: #E65100; margin: 0 0 10px 0;'>⚠️ Recomendación Urgente</h4>
            <p style='color: #FF6F00; margin: 0; line-height: 1.6;'>
            Se ha detectado una posible enfermedad en tu planta. Te recomendamos consultar con un ingeniero agrónomo o especialista en fitopatología para confirmar el diagnóstico y establecer un plan de tratamiento adecuado.
            </p>
            </div>
            {treatment_section}
            <div style='margin-top: 15px; padding: 10px; background: rgba(255,255,255,0.6); border-radius: 8px;'>
            <small style='color: {COLORS['gray']}; font-style: italic;'>
            📅 Análisis realizado: {datetime.now().strftime("%d/%m/%Y %H:%M")} | 🔬 FitoScan AI v2.0
            </small>
            </div>
            </div>
            """
        return prob_chart, confidence_gauge, health_summary, result_message
    
    except Exception as e:
        error_msg = f"<p style='color:red;'>Error técnico: {str(e)}</p>"
        print("Error en predict_disease:", e)
        return None, None, None, error_msg

def get_logo_base64():
    try:
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
    except Exception as e:
        print(f"Error cargando logo: {e}")
        return None

def get_placeholder_results():
    return "<p style='color: #6e6e6c; font-style: italic;'>Sube una imagen y haz clic en 'Analizar Imagen' para ver el diagnóstico.</p>"

def create_interface():
    custom_css = f"""
    /* Aquí va tu CSS personalizado, omitido para brevedad */
    """
    with gr.Blocks(
        theme=gr.themes.Soft(
            primary_hue=gr.themes.colors.green,
            secondary_hue=gr.themes.colors.lime,
            neutral_hue=gr.themes.colors.gray
        ),
        css=custom_css,
        title="🌱 FitoScan - Diagnóstico Inteligente de Plantas"
    ) as app:
        logo_b64 = get_logo_base64()
        with gr.Row(elem_classes="main-header"):
            with gr.Column(scale=1, min_width=100):
                if logo_b64:
                    gr.HTML(f'<div style="text-align: center;"><img src="data:image/png;base64,{logo_b64}" style="height: 120px; width: auto;"></div>')
                else:
                    gr.HTML('<div style="text-align: center; font-size: 6rem;">🌱</div>')
            with gr.Column(scale=3):
                gr.HTML("""
                <div style='text-align: center;'>
                <h1 class='brand-title'>FitoScan</h1>
                <h2 class='brand-subtitle'>Diagnóstico Inteligente de Enfermedades en Plantas 🔬</h2>
                <p class='brand-tagline'>"Cultivo sano, futuro seguro"</p>
                </div>
                """)
        with gr.Tabs(elem_classes="main-tabs") as tabs:
            with gr.Tab("📸 Análisis de Imagen", elem_id="tab-image"):
                with gr.Row(elem_classes="analysis-row"):
                    with gr.Column(scale=1, elem_classes="upload-column"):
                        gr.HTML('<h3 style="color: #2d5a27; text-align: center; margin-bottom: 1rem;">📤 Subir Imagen</h3>')
                        image_input = gr.Image(type="pil", label="", height=400, elem_classes="image-upload", container=True)
                        with gr.Row():
                            analyze_btn = gr.Button("🔍 Analizar Imagen", variant="primary", size="lg", elem_classes="analyze-button")
                            clear_btn = gr.Button("🗑️ Limpiar", variant="secondary", size="lg")
                    with gr.Column(scale=1, elem_classes="results-column"):
                        gr.HTML('<h3 style="color: #2d5a27; text-align: center; margin-bottom: 1rem;">📋 Resultados del Análisis</h3>')
                        results_display = gr.HTML(value=get_placeholder_results(), elem_classes="results-container")
                        with gr.Row(visible=False) as action_buttons:
                            save_btn = gr.Button("💾 Guardar Reporte", variant="secondary")
                            share_btn = gr.Button("📤 Compartir", variant="secondary")
            with gr.Tab("📚 Historial de Análisis", elem_id="tab-history"):
                with gr.Column():
                    gr.HTML('<h3 style="color: #2d5a27; margin-bottom: 1rem;">🗂️ Análisis Anteriores</h3>')
                    with gr.Row():
                        date_filter = gr.Dropdown(choices=["Todos", "Hoy", "Esta semana", "Este mes"], value="Todos", label="📅 Filtrar por fecha")
                        disease_filter = gr.Dropdown(choices=["Todas las enfermedades", "Enfermedades fúngicas", "Plagas", "Deficiencias nutricionales"], value="Todas las enfermedades", label="🦠 Filtrar por tipo")
                    history_list = gr.HTML(value="<p>Historial de análisis no implementado.</p>", elem_classes="history-container")
            with gr.Tab("ℹ️ Información y Ayuda", elem_id="tab-info"):
                with gr.Column():
                    gr.HTML("""
                    <div class="info-section">
                    <h3 style="color: #2d5a27;">🌟 Acerca de FitoScan</h3>
                    <p>FitoScan es una herramienta de inteligencia artificial diseñada para ayudar a agricultores, jardineros y profesionales del sector agrícola a identificar enfermedades y problemas en las plantas de manera rápida y precisa.</p>
                    <h3 style="color: #2d5a27;">🔧 Cómo usar FitoScan</h3>
                    <ol>
                    <li><strong>Sube una imagen:</strong> Toma una foto clara de la planta afectada</li>
                    <li><strong>Analiza:</strong> Haz clic en "Analizar Imagen" para obtener el diagnóstico</li>
                    <li><strong>Revisa los resultados:</strong> Obtén información detallada sobre la enfermedad detectada</li>
                    <li><strong>Aplica el tratamiento:</strong> Sigue las recomendaciones proporcionadas</li>
                    </ol>
                    <h3 style="color: #2d5a27;">📝 Consejos para mejores resultados</h3>
                    <ul>
                    <li>Usa imágenes con buena iluminación natural</li>
                    <li>Enfoca claramente la zona afectada</li>
                    <li>Incluye hojas, tallos o frutos con síntomas visibles</li>
                    <li>Evita imágenes borrosas o muy oscuras</li>
                    </ul>
                    <h3 style="color: #2d5a27;">⚠️ Limitaciones importantes</h3>
                    <p><strong>Nota:</strong> FitoScan es una herramienta de apoyo para el diagnóstico. Para casos críticos o cuando tengas dudas, siempre consulta con un especialista en agronomía o fitopatología.</p>
                    </div>
                    """)
        with gr.Row(elem_classes="footer"):
            gr.HTML("""
            <div style='text-align: center; padding: 2rem; color: #666; background: rgba(255,255,255,0.5); border-radius: 15px; margin-top: 2rem;'>
            <p style='margin: 0; font-size: 0.9rem;'>
            🌱 <strong>FitoScan</strong> - Desarrollado con ❤️ para la agricultura sostenible
            </p>
            <p style='margin: 0.5rem 0 0 0; font-size: 0.8rem; color: #888;'>
            v2.0 | © 2024 - Tecnología al servicio del campo
            </p>
            </div>
            """)

        def handle_image_analysis(image):
            if image is None:
                return get_placeholder_results(), gr.Row(visible=False)
            return predict_disease(image)

        def clear_interface():
            return None, get_placeholder_results(), gr.Row(visible=False)

        analyze_btn.click(
            handle_image_analysis,
            inputs=[image_input],
            outputs=[results_display, action_buttons]
        )

        clear_btn.click(
            clear_interface,
            outputs=[image_input, results_display, action_buttons]
        )

    return app

if __name__ == "__main__":
    app = create_interface()
    app.launch()