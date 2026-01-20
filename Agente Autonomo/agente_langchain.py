import os
import lancedb
from dotenv import load_dotenv

# --- IMPORTS DE ORQUESTACIÓN (CAPA 6) ---
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_core.prompts import ChatPromptTemplate

# 1. Configuración
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# --- HERRAMIENTAS (TOOLS) ---

@tool
def consultar_knowledge_base(query: str) -> str:
    """
    Úsalo para responder preguntas teóricas, buscar opiniones, consejos
    o contenido específico dentro del documento PDF/Curso.
    """
    print(f"\n   🦜 [LangChain Tool] RAG activado: '{query}'")
    
    # Lógica de resilencia de rutas (para que no falle por carpetas)
    db_paths = ["../lancedb_data", "./lancedb_data"]
    db_path = next((p for p in db_paths if os.path.exists(p)), None)
    
    if not db_path:
        return "Error crítico: No encuentro la carpeta lancedb_data."

    try:
        # Conexión a LanceDB (Capa 1)
        db = lancedb.connect(db_path)
        tbl = db.open_table("documentos")
        
        # Embeddings "on the fly" usando cliente raw para velocidad
        import google.genai as genai
        client_raw = genai.Client(api_key=GOOGLE_API_KEY)
        q_res = client_raw.models.embed_content(model="text-embedding-004", contents=query)
        q_vec = [float(x) for x in q_res.embeddings[0].values]
        
        # Retrieval
        results = tbl.search(q_vec).limit(3).to_pandas()
        
        # Formateo de salida
        contexto = "\n".join([f"- {row['text'][:300]}..." for _, row in results.iterrows()])
        return contexto if contexto else "No hay información en el PDF sobre esto."
        
    except Exception as e:
        return f"Error leyendo DB: {e}"

@tool
def calcular_horas_estudio(semanas: int, horas_diarias: float) -> str:
    """
    Úsalo SOLO para realizar cálculos matemáticos numéricos sobre tiempo y planificación.
    """
    print(f"\n   🦜 [LangChain Tool] Calculadora: {semanas} semanas, {horas_diarias}h/día")
    total = semanas * 7 * horas_diarias
    return f"El cálculo matemático exacto es: {total} horas totales."

# --- ARQUITECTURA DEL AGENTE ---

def main():
    print("--- AGENTE ORQUESTADOR (LANGCHAIN + GEMINI 1.5) ---")
    
    # 1. El Cerebro (LLM) - Usamos Gemini Pro que es estable en LangChain
    llm = ChatGoogleGenerativeAI(
        model="gemini-flash-latest",
        temperature=0,
        google_api_key=GOOGLE_API_KEY,
        convert_system_message_to_human=True
    )

    # 2. El Kit de Herramientas
    tools = [consultar_knowledge_base, calcular_horas_estudio]

    # 3. & 4. Ensamblaje del Agente (Forma moderna con LangGraph)
    # LangGraph crea un grafo de ejecución que maneja el flujo automáticamente
    agent_executor = create_react_agent(llm, tools)

    # 5. Bucle de Interacción
    while True:
        user_input = input("\nUsuario: ")
        if user_input.lower() in ["salir", "exit"]:
            break
            
        try:
            # LangGraph usa un formato diferente - recibe mensajes
            response = agent_executor.invoke({"messages": [("user", user_input)]})
            # La respuesta viene en el último mensaje
            print(f"🤖 Agente: {response['messages'][-1].content}")
            
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()