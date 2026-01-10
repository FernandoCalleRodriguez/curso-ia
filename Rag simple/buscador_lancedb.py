import os
import time
import shutil
import lancedb
from typing import List
from dotenv import load_dotenv
from google import genai
from pypdf import PdfReader

# 1. Configuración
load_dotenv()
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

# --- UTILIDADES ---
def sanitize_vector(vector_data) -> List[float]:
    """Asegura que los datos sean floats puros de Python."""
    return [float(x) for x in vector_data]

def reset_db_folder(path: str):
    if os.path.exists(path):
        try:
            shutil.rmtree(path)
            print(f"🧹 Carpeta de datos '{path}' limpia.")
            time.sleep(1)
        except Exception as e:
            print(f"⚠️ Aviso: {e}")

# --- FASE 1: GENERACIÓN (Igual que antes) ---
def generar_vectores(chunks: List[str]) -> List[List[float]]:
    vectors = []
    print(f"⚡ Generando vectores para {len(chunks)} fragmentos...")
    
    for i, text in enumerate(chunks):
        try:
            time.sleep(1.0) # Throttling API
            result = client.models.embed_content(
                model="text-embedding-004",
                contents=text
            )
            vec = sanitize_vector(result.embeddings[0].values)
            vectors.append(vec)
            
            if i % 10 == 0:
                print(f"   ✓ {i} procesados...")
        except Exception as e:
            print(f"   ❌ Error en {i}: {e}")
            vectors.append([0.0] * 768)
    return vectors

# --- MAIN ---
def main():
    db_path = "./lancedb_data" # Carpeta local donde se guardarán los datos
    
    print("--- INICIANDO MIGRACIÓN A LANCEDB (RUST ENGINE) ---")
    reset_db_folder(db_path)
    
    # 1. Conexión a DB (Serverless, solo una carpeta)
    db = lancedb.connect(db_path)
    
    # 2. Leer PDF
    pdf_file = "Los Mejores Cursos de IA para 2026 - by Daniel.pdf"
    if not os.path.exists(pdf_file):
        print("❌ Falta PDF.")
        return

    reader = PdfReader(pdf_file)
    full_text = ""
    for page in reader.pages:
        full_text += page.extract_text() or ""
        
    chunk_size = 500
    chunks = [full_text[i:i+chunk_size] for i in range(0, len(full_text), chunk_size)]
    print(f"📚 {len(chunks)} chunks detectados.")

    # 3. Vectorización (Compute)
    vectors = generar_vectores(chunks)

    # 4. Ingestión (Storage)
    print("\n💾 Guardando en LanceDB...")
    
    # Preparamos los datos como una lista de diccionarios (Formato NoSQL)
    data = []
    for i in range(len(chunks)):
        data.append({
            "vector": vectors[i],
            "text": chunks[i],
            "source": pdf_file,
            "id": i
        })
    
    # CREACIÓN DE LA TABLA
    # LanceDB infiere el esquema automáticamente (vector size, tipos, etc.)
    try:
        tbl = db.create_table("documentos", data=data)
        print("✅ ¡TABLA CREADA Y DATOS PERSISTIDOS!")
    except Exception as e:
        print(f"❌ Error creando tabla: {e}")
        return

    # 5. Bucle de Búsqueda
    while True:
        query = input("\n🔎 LanceDB Search (o 'salir'): ")
        if query.lower() in ['salir', 'exit']:
            break
        
        # Embed Query
        q_res = client.models.embed_content(
            model="text-embedding-004",
            contents=query
        )
        q_vec = sanitize_vector(q_res.embeddings[0].values)
        
        # Search Query (Sintaxis Fluida)
        # .search(vector) -> .limit(3) -> .to_pandas()
        results_df = tbl.search(q_vec).limit(3).to_pandas()
        
        # Iterar resultados (ahora es un DataFrame de Pandas, muy cómodo)
        for index, row in results_df.iterrows():
            # LanceDB devuelve una columna '_distance' automáticamente
            dist = row['_distance'] 
            print(f"\n--- RESULTADO (Dist: {dist:.4f}) ---")
            print(f"📜 ...{row['text'][:200]}...")

if __name__ == "__main__":
    main()