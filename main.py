import os
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import List
from google import genai # <--- OJO: El import cambia. Ahora es "from google import genai"

# 1. Configuración del entorno
load_dotenv()

# 2. Inicializar Cliente (Nueva sintaxis)
# El nuevo SDK usa un cliente instanciado, no métodos estáticos globales.
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

# 3. Definir Estructura (Tu DTO - Esto no cambia)
class CoursePlan(BaseModel):
    topic: str
    weeks: int
    difficulty_level: str
    modules: List[str]

def generate_plan(topic_request: str) -> CoursePlan:
    print(f"--- CONSULTANDO A GEMINI 2.0 SOBRE: {topic_request} ---")
    
    prompt = f"Actúa como un arquitecto de soluciones senior. Diseña un plan de estudio para: {topic_request}"
    
    # 4. Generación (Sintaxis v1 del nuevo SDK)
    response = client.models.generate_content(
        model="gemini-flash-latest", # Usamos el modelo más nuevo y rápido
        contents=prompt,
        config={
            "response_mime_type": "application/json",
            "response_schema": CoursePlan, # Inyección directa del modelo Pydantic
        }
    )
    
    # 5. Deserialización
    # El nuevo SDK a veces devuelve un objeto ya parseado si usas la config correcta,
    # pero para asegurar robustez absoluta, validamos el texto JSON manualmente.
    return CoursePlan.model_validate_json(response.text)

def main():
    try:
        user_topic = "Automatización de procesos con Python y Google Gemini"
        
        plan = generate_plan(user_topic)
        
        print("\n--- PLAN GENERADO POR GEMINI 2.0 ---")
        print(f"Tema: {plan.topic}")
        print(f"Nivel: {plan.difficulty_level} ({plan.weeks} semanas)")
        print("Módulos:")
        for module in plan.modules:
            print(f"  [+] {module}")
            
    except Exception as e:
        print(f"Error en el pipeline de Google: {e}")

if __name__ == "__main__":
    main()