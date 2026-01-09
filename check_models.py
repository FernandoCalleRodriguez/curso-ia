import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

def list_available_models():
    try:
        client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
        print("--- RECUPERANDO LISTA DE MODELOS ---")
        
        # Iteramos sin filtros complejos para evitar errores de atributos
        pager = client.models.list()
        
        for model in pager:
            # Imprimimos solo el ID que necesitas copiar
            # Algunos IDs vienen con 'models/' al principio, otros no.
            print(f"ID: {model.name}")
            
    except Exception as e:
        print(f"❌ Error crítico: {e}")

if __name__ == "__main__":
    list_available_models()