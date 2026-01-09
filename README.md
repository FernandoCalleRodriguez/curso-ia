# Curso de IA con Python y Google Gemini

Este repositorio contiene ejemplos prácticos de aplicaciones de Inteligencia Artificial usando Python y la API de Google Gemini.

## 📋 Contenidos

### Ejemplos Básicos
- **`main.py`**: Generación de planes de estudio estructurados con Pydantic
- **`ml_clasico.py`**: Clasificador de spam con scikit-learn (ML clásico)
- **`check_models.py`**: Lista modelos disponibles de Gemini

### Búsqueda Semántica
- **`embeddings_demo.py`**: Introducción a embeddings y similitud coseno
- **`buscador_semantico.py`**: Buscador semántico básico
- **`buscador_semantico_v2.py`**: Versión mejorada con ChromaDB

## 🚀 Instalación

### 1. Clonar el repositorio
```bash
git clone <tu-repo-url>
cd curso-ia-ferky
```

### 2. Crear entorno virtual
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 3. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 4. Configurar API Key de Google Gemini

1. Obtén tu API key en [Google AI Studio](https://aistudio.google.com/apikey)
2. Copia el archivo `.env.example` a `.env`:
   ```bash
   cp .env.example .env
   ```
3. Edita `.env` y agrega tu API key:
   ```
   GOOGLE_API_KEY=tu_api_key_aqui
   ```

⚠️ **IMPORTANTE**: Nunca subas el archivo `.env` a GitHub. Ya está incluido en `.gitignore`.

## 📚 Uso

### Generar un plan de estudio
```bash
python main.py
```

### Clasificador de spam (ML clásico)
```bash
python ml_clasico.py
```

### Buscador semántico (requiere PDF)
```bash
python buscador_semantico_v2.py
```

## 🛠️ Tecnologías

- **Python 3.10+**
- **Google Gemini API** - Modelos de lenguaje y embeddings
- **Pydantic** - Validación de datos
- **ChromaDB** - Base de datos vectorial
- **scikit-learn** - Machine learning clásico
- **pypdf** - Procesamiento de PDFs

## 📝 Notas

- El buscador semántico incluye rate limiting (1 segundo entre llamadas) para no exceder la cuota gratuita de la API
- Los embeddings se generan con el modelo `text-embedding-004` de Gemini
- ChromaDB se usa en modo in-memory (los datos no persisten entre ejecuciones)

## 🔒 Seguridad

Este proyecto usa variables de entorno para gestionar API keys. Asegúrate de:
- ✅ Nunca hacer commit del archivo `.env`
- ✅ Usar `.env.example` como plantilla
- ✅ No hardcodear API keys en el código

## 📄 Licencia

Este proyecto es de uso educativo.
