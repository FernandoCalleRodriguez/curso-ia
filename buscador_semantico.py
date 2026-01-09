import os
import time
import chromadb
from typing import List
from dotenv import load_dotenv
from google import genai
from pypdf import PdfReader
from chromadb.utils import embedding_functions

# 1. Configuración
load_dotenv()
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

# --- CAPA DE SERVICIO ---
class GeminiEmbeddingFunction(embedding_functions.EmbeddingFunction):
    def __init__(self):
        pass

    def __call__(self, input: List[str]) -> List[List[float]]:
        vectors = []
        total = len(input)
        print(f"   ⚙️  Generando {total} vectores...")
        
        for i, text in enumerate(input):
            try:
                # Throttling preventivo
                time.sleep(1.0) 
                
                result = client.models.embed_content(
                    model="text-embedding-004",
                    contents=text
                )
                
                # --- FIX CRÍTICO: CONVERSIÓN A LISTA PURA ---
                # El SDK devuelve un tipo 'RepeatedComposite', Chroma necesita 'list'
                vector_puro = list(result.embeddings[0].values)
                vectors.append(vector_puro)
                
                if i % 5 == 0:
                    print(f"   ⏳ {i}/{total} ok...")
                    
            except Exception as e:
                print(f"   ❌ Error en chunk {i}: {e}")
                vectors.append([0.0] * 768) 
                
        return vectors

# --- CAPA DE DATOS ---
class DocumentRepository:
    def __init__(self):
        print("💾 Iniciando Base de Datos Vectorial en Memoria RAM...")
        # Usamos Client() normal (no persistente) para evitar bloqueos de archivos en Windows
        self.chroma_client = chromadb.Client()
        
        self.collection = self.chroma_client.create_collection(
            name="mis_documentos",
            embedding_function=GeminiEmbeddingFunction()
        )

    def add_document(self, filename: str):
        print(f"--- LEYENDO PDF: {filename} ---")
        
        reader = PdfReader(filename)
        full_text = ""
        for page in reader.pages:
            full_text += page.extract_text() or ""
            
        # Chunking
        chunk_size = 500
        chunks = [full_text[i:i+chunk_size] for i in range(0, len(full_text), chunk_size)]
        
        print(f"📚 {len(chunks)} fragmentos detectados. Indexando...")
        
        try:
            # Aquí es donde fallaba antes
            self.collection.add(
                documents=chunks,
                ids=[f"id_{i}" for i in range(len(chunks))],
                metadatas=[{"source": filename} for _ in range(len(chunks))]
            )
            print("✅ ¡INDEXACIÓN COMPLETADA EXITOSAMENTE!") # Si ves esto, funciona.
        except Exception as e:
            print(f"❌ Error fatal guardando en Chroma: {e}")

    def search(self, query: str, n_results=3):
        print(f"\n🔎 Buscando: '{query}'")
        
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        # Mostrar resultados
        if results['documents']:
            for i in range(len(results['documents'][0])):
                doc = results['documents'][0][i]
                print(f"\n--- HALLAZGO #{i+1} ---")
                print(f"📜 ...{doc.replace(chr(10), ' ')}...")
        else:
            print("No se encontraron coincidencias.")

# --- MAIN ---
def main():
    repo = DocumentRepository()
    pdf_file = "Los Mejores Cursos de IA para 2026 - by Daniel.pdf"
    
    if os.path.exists(pdf_file):
        repo.add_document(pdf_file)
        
        # Bucle de preguntas
        while True:
            query = input("\nPregunta (o 'salir'): ")
            if query.lower() in ['salir', 'exit']:
                break
            repo.search(query)
    else:
        print(f"❌ Archivo {pdf_file} no encontrado.")

if __name__ == "__main__":
    main()