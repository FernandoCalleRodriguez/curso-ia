import os
import numpy as np
from dotenv import load_dotenv
from google import genai
from sklearn.metrics.pairwise import cosine_similarity

# 1. Configuración
load_dotenv()
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

def get_embedding(text: str):
    """Convierte texto en un vector de 768 dimensiones."""
    result = client.models.embed_content(
        model="text-embedding-004",
        contents=text
    )
    # Dependiendo de la versión de la API, el embedding puede venir en diferentes atributos.
    # En el nuevo SDK suele ser .embeddings[0].values
    return result.embeddings[0].values

def main():
    print("--- INICIANDO MOTOR DE EMBEDDINGS ---")
    
    # 2. Datos de Prueba
    # Fíjate que "Perro" y "Canino" NO comparten ninguna letra.
    phrases = [
        "El perro ladra en el parque",  # Frase 0
        "Un canino hace ruido en el jardín", # Frase 1 (Semánticamente idéntica a la 0)
        "Me encanta programar en Python",    # Frase 2 (Tema totalmente distinto)
        "La inteligencia artificial es el futuro", # Frase 3
        "Hoy voy a comer pizza" # Frase 4
    ]
    
    # 3. Vectorización (Aquí ocurre la magia)
    print("Calculando vectores matemáticos...")
    embeddings = []
    for text in phrases:
        vector = get_embedding(text)
        embeddings.append(vector)
        # Un vector es solo una lista de números float, ej: [0.012, -0.931, ...]
        print(f"✅ '{text[:20]}...' -> Vector de {len(vector)} dimensiones")

    # 4. Cálculo de Similitud (Álgebra)
    # Convertimos a matriz numpy para cálculo rápido
    matrix = np.array(embeddings)
    
    # Similitud del Coseno: 1.0 es idéntico, 0.0 es nada que ver
    similarity_matrix = cosine_similarity(matrix)

    print("\n--- MATRIZ DE SIMILITUD (¿Qué tanto se parecen?) ---")
    
    # Comparamos la primera frase ("El perro...") contra todas las demás
    base_phrase_idx = 0 
    base_phrase = phrases[base_phrase_idx]
    
    print(f"\nComparando todo contra: '{base_phrase}'\n")
    
    for i in range(len(phrases)):
        score = similarity_matrix[base_phrase_idx][i]
        text = phrases[i]
        
        # Formato visual
        bar = "█" * int(score * 20) 
        print(f"{bar} {score:.4f} | {text}")

if __name__ == "__main__":
    main()