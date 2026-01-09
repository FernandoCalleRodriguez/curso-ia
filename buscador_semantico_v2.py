import os
import sys
import time
import chromadb
from typing import List
from dotenv import load_dotenv
from google import genai
from pypdf import PdfReader

# 1. Configuración
load_dotenv()
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

# --- FASE 1: LÓGICA DE NEGOCIO (Generación) ---
def generar_embeddings_manualmente(chunks: List[str]) -> List[List[float]]:
    vectors = []
    total = len(chunks)
    print(f"⚡ Iniciando vectorización de {total} fragmentos...")
    
    for i, text in enumerate(chunks):
        try:
            # Rate Limiting (Freno de mano para la API gratuita)
            time.sleep(1.0) 
            
            result = client.models.embed_content(
                model="text-embedding-004",
                contents=text
            )
            
            # Conversión explícita a lista de floats
            vector = list(result.embeddings[0].values)
            vectors.append(vector)
            
            # Feedback visual
            if i % 5 == 0:
                print(f"   ✓ Vector {i}/{total} generado ok")
                
        except Exception as e:
            print(f"   ❌ Fallo en chunk {i}: {e}")
            # Vector vacío de relleno para mantener la alineación
            vectors.append([0.0] * 768)
            
    return vectors

# --- FASE 2: CAPA DE DATOS (Repositorio) ---
def main():
    print("--- INICIANDO BUSCADOR SEMÁNTICO V2 (DESACOPLADO) ---")
    
    pdf_file = "Los Mejores Cursos de IA para 2026 - by Daniel.pdf"
    if not os.path.exists(pdf_file):
        print(f"❌ No encuentro el archivo: {pdf_file}")
        return

    # A. Leemos PDF
    reader = PdfReader(pdf_file)
    full_text = ""
    for page in reader.pages:
        full_text += page.extract_text() or ""

    # B. Chunking
    chunk_size = 500
    chunks = [full_text[i:i+chunk_size] for i in range(0, len(full_text), chunk_size)]
    print(f"📚 Texto dividido en {len(chunks)} fragmentos.")

    # C. Cómputo (Aquí ocurre el gasto de tiempo/API)
    print("\n--- FASE 1: GENERACIÓN DE VECTORES ---")
    vectors = generar_embeddings_manualmente(chunks)
    
   # D. Almacenamiento (Con Batch Insert para evitar crash)
    print("\n--- FASE 2: GUARDADO EN BASE DE DATOS (BATCHING) ---")
    try:
        print("   [*] Inicializando cliente ChromaDB...")
        chroma_client = chromadb.Client()
        print("   [OK] Cliente creado")
        
        print("   [*] Creando coleccion 'mis_documentos'...")
        collection = chroma_client.create_collection(name="mis_documentos")
        print("   [OK] Coleccion creada")
        
        # Preparamos los metadatos y IDs fuera del loop
        ids = [f"id_{i}" for i in range(len(chunks))]
        metadatas = [{"source": pdf_file} for _ in range(len(chunks))]
        print(f"   [OK] Preparados {len(ids)} IDs y metadatos")
        
        # --- BATCH INSERT PATTERN ---
        batch_size = 10  # Insertamos de 10 en 10
        total_chunks = len(chunks)
        
        for i in range(0, total_chunks, batch_size):
            end_idx = min(i + batch_size, total_chunks)
            print(f"   [*] Insertando lote {i} a {end_idx}...", end="", flush=True)
            
            collection.add(
                documents=chunks[i:end_idx],
                embeddings=vectors[i:end_idx],
                ids=ids[i:end_idx],
                metadatas=metadatas[i:end_idx]
            )
            print(" OK")
            time.sleep(0.2) # Pequeña pausa técnica

        print("[OK] Base de Datos indexada correctamente.")
        print("   [*] Iniciando bucle de busqueda...")
        
        # E. Bucle de Búsqueda
        while True:
            query = input("\n[?] Que quieres buscar? (o 'salir'): ")
            if query.lower() in ['salir', 'exit']:
                break
                
            print("   Calculando vector de la pregunta...")
            q_result = client.models.embed_content(
                model="text-embedding-004",
                contents=query
            )
            q_vector = list(q_result.embeddings[0].values)
            
            print("   Consultando ChromaDB...")
            results = collection.query(
                query_embeddings=[q_vector], 
                n_results=3
            )
            
            for k in range(len(results['documents'][0])):
                doc = results['documents'][0][k]
                score = results['distances'][0][k]
                print(f"\n--- RESULTADO #{k+1} (Dist: {score:.4f}) ---")
                print(f"📜 ...{doc.replace(chr(10), ' ')}...")
                
    except Exception as e:
        print(f"\n❌ Error crítico en la base de datos: {e}")

if __name__ == "__main__":
    main()