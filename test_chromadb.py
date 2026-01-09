import os
import sys
import time
import chromadb
from typing import List
from dotenv import load_dotenv
import google.generativeai as genai

# Configurar encoding UTF-8 para Windows
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# 1. Configuración
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Test con un solo embedding
print("🧪 TEST: Generando un embedding de prueba...")
result = genai.embed_content(
    model="models/text-embedding-004",
    content="Hola mundo"
)

embedding_test = result['embedding']
print(f"   ✓ Embedding generado: tipo={type(embedding_test)}, dimensión={len(embedding_test)}")
print(f"   Primeros valores: {embedding_test[:5]}")

# Test ChromaDB
print("\n🧪 TEST: Insertando en ChromaDB...")
try:
    chroma_client = chromadb.Client()
    collection = chroma_client.create_collection(name="test_collection")
    print("   ✓ Colección creada")
    
    collection.add(
        documents=["Documento de prueba"],
        embeddings=[embedding_test],
        ids=["id_0"],
        metadatas=[{"source": "test"}]
    )
    print("   ✓ Embedding insertado correctamente!")
    
    # Test de búsqueda
    print("\n🧪 TEST: Probando búsqueda...")
    query_result = genai.embed_content(
        model="models/text-embedding-004",
        content="saludo al planeta"
    )
    query_embedding = query_result['embedding']
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=1
    )
    print(f"   ✓ Resultado: {results['documents'][0][0]}")
    print("\n✅ TODOS LOS TESTS PASARON - El problema NO está en ChromaDB")
    
except Exception as e:
    import traceback
    print(f"\n❌ Error en el test: {e}")
    print("\n📋 Traceback completo:")
    traceback.print_exc()
