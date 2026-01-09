import os
import chromadb
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

print("[TEST] Inicializando cliente ChromaDB...")
try:
    chroma_client = chromadb.Client()
    print("[OK] Cliente creado")
    
    # Intentar limpiar colección existente primero
    try:
        chroma_client.delete_collection(name="test_collection_final")
        print("[INFO] Coleccion anterior eliminada")
    except:
        print("[INFO] No habia coleccion anterior")
    
    print("[TEST] Creando coleccion...")
    collection = chroma_client.create_collection(name="test_collection_final")
    print("[OK] Coleccion creada")
    
    # Generar un embedding simple
    print("[TEST] Generando embedding...")
    result = genai.embed_content(
        model="models/text-embedding-004",
        content="Texto de prueba"
    )
    embedding = result['embedding']
    print(f"[OK] Embedding generado: dimension={len(embedding)}")
    
    # Probar inserción simple
    print("[TEST] Insertando documento...")
    collection.add(
        documents=["Documento de prueba"],
        embeddings=[embedding],
        ids=["id_test_1"],
        metadatas=[{"source": "test"}]
    )
    print("[OK] Documento insertado exitosamente!")
    
    # Probar búsqueda
    print("[TEST] Probando busqueda...")
    results = collection.query(
        query_embeddings=[embedding],
        n_results=1
    )
    print(f"[OK] Busqueda exitosa: {results['documents'][0][0]}")
    
    print("\n[SUCCESS] TODOS LOS TESTS PASARON!")
    
except Exception as e:
    import traceback
    print(f"\n[ERROR] Fallo en el test: {e}")
    print("\n[TRACE] Traceback completo:")
    traceback.print_exc()
