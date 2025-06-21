# 🌱 Plant Disease Prediction



## ⚙️ Instalación

Sigue estos pasos para configurar el proyecto:

```powershell

# 1. Clonar repositorio

git clone https://github.com/DMerchan1986/plant-disease-prediction.git

cd plant-disease-prediction

# 2. Descargar modelo de predicción (148 MB)

## 🔍 Descarga alternativa del modelo


1. Descarga manualmente desde [Google Drive](https://drive.google.com/file/d/1j4kVbOxuDfSYgawz86p6sMMb8TGo8rKa/view?usp=sharing)

2. Coloca el archivo `model.keras` en la carpeta `output/`


# 3. Instalar dependencias de Python

pip install -r requirements.txt

```


Ejemplo de salida:

```

P
```

## 📁 Estructura del proyecto

```

plant-disease-prediction/

├── output/               # Modelo entrenado (descargado automáticamente)

│   └── model.keras

├── src/                  # Código fuente

│   ├── train.py          # Script de entrenamiento

│   └── utils.py          # Script de predicción

│   └── app.py            # Script de interfaz web


├── data/                 # Dataset de imágenes (opcional)


├── download_model.ps1    # Script de descarga del modelo

├── requirements.txt      # Dependencias de Python

├── .gitignore            # Archivos ignorados por Git

└── README.md             # Este archivo

```



## 🧠 Modelo y Dataset

- **Arquitectura**: EfficientNetB3

- **Dataset**: PlantVillage (18,000 imágenes, 15 clases)

- **Precisión**: 98.2% en conjunto de validación

- **Tamaño del modelo**: 148 MB

## 🤝 Contribuir

Las contribuciones son bienvenidas:

1. Haz fork del proyecto

2. Crea una rama (`git checkout -b feature/nueva-funcion`)

3. Haz commit de tus cambios (`git commit -am 'Agrega nueva función'`)

4. Haz push a la rama (`git push origin feature/nueva-funcion`)

5. Abre un Pull Request

## 📄 Licencia

Este proyecto está bajo la [Licencia MIT](LICENSE) - ver el archivo [LICENSE](LICENSE) para más detalles.

## ✉️ Contacto

Daniel Merchan - [@DMerchan1986](https://github.com/DMerchan1986)

Proyecto: [https://github.com/DMerchan1986/plant-disease-prediction](https://github.com/DMerchan1986/plant-disease-prediction)

```

### Pasos para usar este archivo:

1. Abre tu editor de texto (como Notepad, VSCode, etc.).

2. Copia todo el contenido de arriba.

3. Pégalo en un nuevo archivo.

4. Guarda el archivo como `README.md` en la carpeta principal de tu proyecto (por ejemplo, `C:\Users\DELL\Desktop\plant-disease-prediction\README.md`).

5. **No olvides reemplazar** `TU_ID_DE_ARCHIVO` con el ID real de tu archivo en Google Drive.

### Para agregar el archivo a tu repositorio:

Abre PowerShell en la carpeta de tu proyecto y ejecuta:

```powershell

git add README.md

git commit -m "Agregar README completo"

git push origin main

```