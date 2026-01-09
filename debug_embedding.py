import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Test de embedding
print("[TEST] Generando embedding de prueba...")
result = genai.embed_content(
    model="models/text-embedding-004",
    content="Hola mundo"
)

embedding = result['embedding']
print(f"[INFO] Tipo: {type(embedding)}")
print(f"[INFO] Dimension: {len(embedding)}")
print(f"[INFO] Tipo del primer elemento: {type(embedding[0])}")
print(f"[INFO] Primeros 5 valores: {embedding[:5]}")

# Verificar si todos son floats
all_floats = all(isinstance(x, float) for x in embedding)
print(f"[INFO] Todos son floats? {all_floats}")
