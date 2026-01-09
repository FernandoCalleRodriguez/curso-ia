import json
from pydantic import BaseModel
from typing import List

# 1. Definimos la estructura de datos deseada (El Output)
class CoursePlan(BaseModel):
    topic: str
    weeks: int
    modules: List[str]

# 2. Simulamos un "Input Sucio" (como vendría de una API o LLM)
raw_data_from_llm = """
{
    "topic": "Arquitectura de IA con Python",
    "weeks": 15,
    "modules": ["Fundamentos", "RAG", "Agentes", "Despliegue"]
}
"""

def main():
    print("--- INICIANDO SISTEMA ---")
    
    try:
        # 3. Parsing: Convertir String JSON a Diccionario
        data_dict = json.loads(raw_data_from_llm)
        
        # 4. Validación: Convertir Diccionario a Objeto Tipado (Pydantic)
        plan = CoursePlan(**data_dict)
        
        # 5. Lógica de negocio (usando propiedades tipadas)
        print(f"Plan generado: {plan.topic}")
        print(f"Duración: {plan.weeks} semanas")
        
        # 6. List Comprehension (Stream) para procesar
        modules_upper = [m.upper() for m in plan.modules]
        print(f"Módulos procesados: {modules_upper}")
        
        print("--- SISTEMA VALIDADO ---")
        
    except Exception as e:
        print(f"Error crítico en el pipeline: {e}")

# Entry point estándar (como tu public static void main)
if __name__ == "__main__":
    main()