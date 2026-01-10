# 🤖 Guía Completa del Sistema RAG con LanceDB + Gemini

## 📚 Índice

1. [Stack Tecnológico](#stack-tecnológico)
2. [Archivo 1: buscador_lancedb.py](#archivo-1-buscador_lancedbpy)
3. [Archivo 2: asistente_rag_completo.py](#archivo-2-asistente_rag_completopy)
4. [Preguntas Frecuentes](#preguntas-frecuentes)
5. [Flujo Completo End-to-End](#flujo-completo-end-to-end)

---

## 🔧 Stack Tecnológico

| Librería | Propósito | Dónde se usa |
|----------|-----------|--------------|
| **`lancedb`** | Base de datos vectorial (Rust) | Ambos archivos |
| **`google.genai`** | API de Google Gemini (embeddings + LLM) | Ambos archivos |
| **`pypdf`** | Leer archivos PDF | Solo `buscador_lancedb.py` |
| **`dotenv`** | Cargar variables de entorno (.env) | Ambos archivos |
| **`os`** | Operaciones del sistema (paths, archivos) | Ambos archivos |
| **`time`** | Delays para rate limiting | Solo `buscador_lancedb.py` |
| **`shutil`** | Operaciones de archivos (borrar carpetas) | Solo `buscador_lancedb.py` |
| **`pandas`** | Manipulación de datos tabulares | Indirectamente (LanceDB devuelve DataFrames) |

---

## 📁 Archivo 1: `buscador_lancedb.py`

### **Propósito**: Indexador - Crear la base de datos vectorial

---

### FASE 1: Imports y Configuración

```python
import os
import time
import shutil
import lancedb
from typing import List
from dotenv import load_dotenv
from google import genai
from pypdf import PdfReader
```

| Import | Framework | Función usada | Para qué sirve |
|--------|-----------|---------------|----------------|
| `os` | Python stdlib | `os.path.exists()` | Verificar si existe el PDF |
| `time` | Python stdlib | `time.sleep(1.0)` | Rate limiting (1 seg entre llamadas API) |
| `shutil` | Python stdlib | `shutil.rmtree()` | Borrar carpeta de DB al inicio |
| `lancedb` | LanceDB | `lancedb.connect()`, `db.create_table()` | Crear y gestionar DB vectorial |
| `typing.List` | Python stdlib | Type hints | Documentar tipos (mejor IDE support) |
| `dotenv` | python-dotenv | `load_dotenv()` | Cargar `GOOGLE_API_KEY` desde `.env` |
| `google.genai` | Google SDK | `genai.Client()`, `embed_content()` | Generar embeddings |
| `pypdf` | pypdf | `PdfReader()` | Leer archivos PDF |

---

### FASE 2: Configuración del Cliente

```python
load_dotenv()
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
```

| Función | Librería | Propósito |
|---------|----------|-----------|
| `load_dotenv()` | python-dotenv | Carga variables de `.env` a `os.environ` |
| `os.getenv("GOOGLE_API_KEY")` | os | Lee variable de entorno |
| `genai.Client(api_key=...)` | google.genai | Inicializa cliente autenticado de Gemini |

---

### FASE 3: Funciones Auxiliares

#### Función 1: `sanitize_vector()`

```python
def sanitize_vector(vector_data) -> List[float]:
    return [float(x) for x in vector_data]
```

**Propósito**: Convertir embeddings de Gemini a floats puros de Python (LanceDB requiere tipos nativos).

#### Función 2: `reset_db_folder()`

```python
def reset_db_folder(path: str):
    if os.path.exists(path):
        shutil.rmtree(path)
        time.sleep(1)
```

| Función | Librería | Para qué sirve |
|---------|----------|----------------|
| `os.path.exists(path)` | os | Verifica si existe la carpeta |
| `shutil.rmtree(path)` | shutil | Borra recursivamente toda la carpeta |
| `time.sleep(1)` | time | Espera para evitar conflictos de I/O |

---

### FASE 4: Generación de Vectores

#### Función 3: `generar_vectores()`

```python
def generar_vectores(chunks: List[str]) -> List[List[float]]:
    vectors = []
    for i, text in enumerate(chunks):
        time.sleep(1.0)  # Rate limiting
        
        result = client.models.embed_content(
            model="text-embedding-004",
            contents=text
        )
        
        vec = sanitize_vector(result.embeddings[0].values)
        vectors.append(vec)
    
    return vectors
```

| Paso | Función/Método | Librería | Para qué sirve |
|------|----------------|----------|----------------|
| 1 | `enumerate(chunks)` | Python | Itera con índice (para feedback cada 10) |
| 2 | `time.sleep(1.0)` | time | **Rate limiting**: Evita exceder cuota de API |
| 3 | `client.models.embed_content()` | google.genai | **Genera embedding** (vector de 768 dims) |
| 4 | `result.embeddings[0].values` | google.genai | Extrae el vector del objeto respuesta |
| 5 | `sanitize_vector()` | Custom | Convierte a floats puros |
| 6 | `vectors.append(vec)` | Python list | Acumula vectores en lista |

**Parámetros clave**:
- `model="text-embedding-004"`: Modelo de embeddings de Gemini (768 dimensiones)
- `contents=text`: El texto a vectorizar

---

### FASE 5: Main - Flujo Principal

```python
def main():
    # 1. Conexión a LanceDB
    db_path = "./lancedb_data"
    db = lancedb.connect(db_path)
    
    # 2. Leer PDF
    reader = PdfReader(pdf_file)
    full_text = ""
    for page in reader.pages:
        full_text += page.extract_text() or ""
    
    # 3. Chunking
    chunk_size = 500
    chunks = [full_text[i:i+chunk_size] 
              for i in range(0, len(full_text), chunk_size)]
    
    # 4. Vectorización
    vectors = generar_vectores(chunks)
    
    # 5. Preparar datos en formato NoSQL
    data = []
    for i in range(len(chunks)):
        data.append({
            "vector": vectors[i],    # Embedding (768 floats)
            "text": chunks[i],        # Texto original
            "source": pdf_file,      # Metadata
            "id": i                  # ID único
        })
    
    # 6. Crear tabla en LanceDB
    tbl = db.create_table("documentos", data=data)
    
    # 7. Bucle de búsqueda interactiva
    while True:
        query = input("Buscar: ")
        
        # Vectorizar query
        q_res = client.models.embed_content(
            model="text-embedding-004",
            contents=query
        )
        q_vec = sanitize_vector(q_res.embeddings[0].values)
        
        # Búsqueda vectorial
        results_df = tbl.search(q_vec).limit(3).to_pandas()
        
        # Mostrar resultados
        for _, row in results_df.iterrows():
            print(f"Distancia: {row['_distance']:.4f}")
            print(f"Texto: {row['text'][:200]}...")
```

#### Detalle por paso:

| Paso | Función/Método | Librería | Para qué sirve |
|------|----------------|----------|----------------|
| **1. Conexión DB** | `lancedb.connect(path)` | lancedb | Crea/abre DB en carpeta local |
| **2. Leer PDF** | `PdfReader(file)` | pypdf | Inicializa lector de PDF |
| | `reader.pages` | pypdf | Itera sobre páginas |
| | `page.extract_text()` | pypdf | Extrae texto de la página |
| **3. Chunking** | List comprehension | Python | Divide texto en fragmentos de 500 chars |
| **4. Vectorizar** | `generar_vectores()` | Custom | Genera embeddings para todos los chunks |
| **5. Preparar datos** | Dict comprehension | Python | Formato NoSQL (lista de diccionarios) |
| **6. Crear tabla** | `db.create_table()` | lancedb | Crea tabla e inserta datos |
| **7. Búsqueda** | `tbl.search(vector)` | lancedb | **Búsqueda vectorial** (similitud coseno) |
| | `.limit(3)` | lancedb | Top 3 resultados |
| | `.to_pandas()` | lancedb | Convierte a DataFrame de Pandas |
| | `iterrows()` | pandas | Itera sobre filas del DataFrame |

---

## 📁 Archivo 2: `asistente_rag_completo.py`

### **Propósito**: Sistema RAG - Consultar DB + Generar respuestas con LLM

### ¿Qué es RAG?

**RAG** = **R**etrieval + **A**ugmented **G**eneration

1. **Retrieval**: Busca fragmentos relevantes en la DB
2. **Augmented Generation**: LLM genera respuesta basada en esos fragmentos

---

### FASE 1: Imports y Configuración

```python
import os
import lancedb
from dotenv import load_dotenv
from google import genai

load_dotenv()
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
```

**Nota**: Solo usa `lancedb` y `genai` (no necesita pypdf, time, shutil - solo consulta).

---

### FASE 2: Retrieval - Búsqueda

#### Función 1: `buscar_contexto()`

```python
def buscar_contexto(query: str, db_path: str = "./lancedb_data") -> str:
    # 1. Conectar a DB existente
    db = lancedb.connect(db_path)
    tbl = db.open_table("documentos")
    
    # 2. Vectorizar pregunta
    q_res = client.models.embed_content(
        model="text-embedding-004",
        contents=query
    )
    q_vec = [float(x) for x in q_res.embeddings[0].values]
    
    # 3. Búsqueda vectorial (Top 5)
    results = tbl.search(q_vec).limit(5).to_pandas()
    
    # 4. Concatenar fragmentos en un solo string
    contexto_unificado = ""
    for _, row in results.iterrows():
        contexto_unificado += f"\n--- FRAGMENTO ---\n{row['text']}\n"
    
    return contexto_unificado
```

| Paso | Función/Método | Librería | Para qué sirve |
|------|----------------|----------|----------------|
| 1 | `lancedb.connect()` | lancedb | Conecta a DB **existente** |
| 2 | `db.open_table("documentos")` | lancedb | **Abre tabla** (no crea, solo lee) |
| 3 | `embed_content()` | google.genai | Vectoriza la pregunta |
| 4 | `tbl.search(q_vec).limit(5)` | lancedb | Top 5 fragmentos más similares |
| 5 | `.to_pandas()` | lancedb | Convierte a DataFrame |
| 6 | `iterrows()` | pandas | Itera sobre resultados |
| 7 | String concatenation | Python | Une fragmentos en un solo texto |

---

### FASE 3: Generation - Síntesis con LLM

#### Función 2: `generar_respuesta()`

```python
def generar_respuesta(query: str, contexto: str):
    # Construir prompt con instrucciones + contexto + pregunta
    prompt = f"""
    Eres un Asistente Técnico experto en IA.
    Tu misión es responder BASÁNDOTE SOLO en el contexto proporcionado.
    
    CONTEXTO RECUPERADO DE LA BASE DE CONOCIMIENTO:
    {contexto}
    
    PREGUNTA DEL USUARIO:
    "{query}"
    
    INSTRUCCIONES:
    1. Si la respuesta está en el contexto, explícala detalladamente.
    2. Cita herramientas o lenguajes específicos (Python, Java, etc.).
    3. Si NO está en el contexto, di: "El documento no menciona...".
    4. Ignora pies de página o texto irrelevante.
    """
    
    # Llamar al LLM
    response = client.models.generate_content(
        model="gemini-flash-latest",
        contents=prompt
    )
    
    return response.text
```

| Paso | Función/Método | Librería | Para qué sirve |
|------|----------------|----------|----------------|
| 1 | f-string formatting | Python | Construye prompt estructurado |
| 2 | `generate_content()` | google.genai | **Llama al LLM** (Gemini Flash) |
| 3 | `response.text` | google.genai | Extrae respuesta en texto plano |

**Diferencia clave**: 
- `embed_content()` → Vectores (Retrieval)
- `generate_content()` → Texto (Generation)

---

### FASE 4: Main - Bucle RAG Completo

```python
def main():
    print("Sistema RAG listo")
    
    while True:
        query = input("Pregunta: ")
        
        # PASO 1: RETRIEVAL (Búsqueda)
        contexto = buscar_contexto(query)
        
        # PASO 2: GENERATION (Síntesis)
        respuesta = generar_respuesta(query, contexto)
        
        # PASO 3: Mostrar respuesta
        print(respuesta)
```

---

## ❓ Preguntas Frecuentes

### 1. ¿Qué es un DataFrame?

Un **DataFrame** es una estructura tabular de **Pandas** (como Excel en código).

#### Ejemplo:

```python
results_df = tbl.search(q_vec).limit(3).to_pandas()
```

Resultado:

| _distance | id | text | source | vector |
|-----------|----|----|--------|--------|
| 0.8562 | 5 | "Código\nExplicación..." | "curso.pdf" | [0.023, -0.15, ...] |
| 0.8577 | 12 | "s empezando\n• Podéis..." | "curso.pdf" | [0.012, 0.08, ...] |

**Uso**:
```python
for _, row in results_df.iterrows():
    print(row['text'])        # Columna de texto
    print(row['_distance'])   # Columna de similitud
```

---

### 2. ¿Cómo se recupera el texto si solo guardas embeddings?

**¡Se guardan AMBOS!**

```python
data.append({
    "vector": vectors[i],    # ← EMBEDDING (768 floats)
    "text": chunks[i],        # ← TEXTO PLANO
    "source": pdf_file,      # ← Metadata
    "id": i
})
```

**Flujo de recuperación**:
1. LanceDB **busca** por similitud de vectores
2. **Devuelve** TODAS las columnas de las filas encontradas
3. Accedes al texto con `row['text']`

| Columna | Tipo | Ejemplo |
|---------|------|---------|
| `vector` | List[float] | `[0.023, -0.15, ...]` (768 nums) |
| `text` | str | `"Código\nExplicación..."` |
| `source` | str | `"curso.pdf"` |

---

### 3. ¿Cómo funciona la búsqueda de similitud?

#### Similitud del Coseno

Fórmula matemática:

```
similitud = cos(θ) = (A · B) / (||A|| × ||B||)
```

Donde:
- **A** = Vector de la pregunta (768 dims)
- **B** = Vector de cada chunk (768 dims)
- **·** = Producto escalar
- **||A||** = Magnitud del vector

#### Implementación en Python:

```python
import numpy as np

def similitud_coseno(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_a = np.linalg.norm(vec1)
    norm_b = np.linalg.norm(vec2)
    return dot_product / (norm_a * norm_b)

# Ejemplo:
pregunta = [0.5, 0.3, 0.2]
chunk1 = [0.6, 0.2, 0.1]
chunk2 = [0.1, 0.9, 0.3]

print(similitud_coseno(pregunta, chunk1))  # 0.95 (muy similar)
print(similitud_coseno(pregunta, chunk2))  # 0.72 (menos similar)
```

#### Proceso de LanceDB:

```python
results = tbl.search(q_vec).limit(3)
```

**Internamente**:
1. Calcula similitud entre `q_vec` y TODOS los vectores guardados
2. Ordena de mayor a menor similitud
3. Devuelve top 3

#### Interpretación de `_distance`:

| Valor | Significado |
|-------|-------------|
| 0.0 | Idéntico |
| 0.8-1.0 | Muy similar |
| 0.5-0.8 | Algo similar |
| < 0.5 | Poco similar |

**Nota**: LanceDB devuelve `_distance = 1 - similitud`, por eso valores **menores** son mejores.

---

## 🔄 Flujo Completo End-to-End

### Indexación (buscador_lancedb.py):

```
1. PDF → pypdf.PdfReader → Texto completo (36,181 chars)
2. Chunking → 73 fragmentos de 500 chars
3. Vectorización → genai.embed_content → 73 vectores de 768 dims
4. Almacenamiento → lancedb.create_table → DB: ./lancedb_data/
```

### Consulta RAG (asistente_rag_completo.py):

```
1. Usuario pregunta → "mejor lenguaje de programación"
2. Vectorizar → genai.embed_content → [0.5, 0.3, ...]
3. RETRIEVAL → lancedb.search → Top 5 fragmentos similares
4. Concatenar → String con los 5 fragmentos
5. GENERATION → genai.generate_content → Respuesta sintetizada
6. Usuario recibe → Respuesta en lenguaje natural
```

---

## 📊 Comparación: Buscador vs RAG

| Característica | `buscador_lancedb.py` | `asistente_rag_completo.py` |
|----------------|----------------------|----------------------------|
| **Output** | Fragmentos crudos | Respuesta sintetizada |
| **Formato** | 3 chunks de texto | Párrafo coherente |
| **LLM** | No usa | Usa Gemini Flash |
| **Experiencia** | "Google Search" | "ChatGPT personalizado" |
| **Uso** | Explorar documentos | Responder preguntas |

---

## 🚀 Resumen de Funciones por Librería

### LanceDB

| Función | Archivo | Para qué |
|---------|---------|----------|
| `lancedb.connect(path)` | Ambos | Conecta a DB local |
| `db.create_table(name, data)` | buscador | Crea tabla nueva |
| `db.open_table(name)` | asistente | Abre tabla existente |
| `tbl.search(vector).limit(N)` | Ambos | Búsqueda vectorial |
| `.to_pandas()` | Ambos | Convierte a DataFrame |

### Google Genai

| Función | Modelo | Para qué |
|---------|--------|----------|
| `embed_content()` | text-embedding-004 | Vectorizar texto (768D) |
| `generate_content()` | gemini-flash-latest | Generar texto con LLM |

### PyPDF

| Función | Para qué |
|---------|----------|
| `PdfReader(file)` | Inicializa lector |
| `reader.pages` | Itera páginas |
| `page.extract_text()` | Extrae texto |

---

## 💡 Conceptos Clave

### Grounding

```python
"Responde BASÁNDOTE SOLO en el contexto proporcionado"
```

**Beneficios**:
- ✅ Evita alucinaciones
- ✅ Respuestas verificables
- ✅ Transparencia sobre qué sabe y qué no

### Rate Limiting

```python
time.sleep(1.0)  # 1 segundo entre llamadas
```

**Por qué**: API gratuita de Gemini tiene límites de requests por minuto.

### Chunking

```python
chunk_size = 500  # Caracteres por fragmento
```

**Por qué**: 
- Embeddings funcionan mejor con textos cortos y coherentes
- Facilita recuperar secciones específicas
- Balance entre contexto y precisión

---

**Fecha**: Enero 2026  
**Autor**: Sistema RAG con LanceDB + Google Gemini  
**Versión**: 1.0
