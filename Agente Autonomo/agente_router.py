import os
import lancedb
from dotenv import load_dotenv

# Importaciones de LangChain (La "Capa de Abstracción")
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate

# 1. Configuración
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# --- CAPA DE HERRAMIENTAS (Decoradores) ---

@tool
def consultar_knowledge_base(query: str) -> str:
    """
    Útil para buscar información teórica, explicaciones o contenido del curso en el PDF.
    Úsalo para preguntas como '¿Qué es...?', '¿Cómo...?', 'Resumen de...'.
    """
    print(f"\n   🦜 [LangChain] RAG Tool invocada: '{query}'")
    
    # Lógica de LanceDB (Idéntica a antes, pero encapsulada)
    # FIX RUTA: Intentamos ruta relativa y absoluta
    db_paths = ["../lancedb_data", "./lancedb_data"]
    db_path = next((p for p in db_paths if os.path.exists(p)), None)
    
    if not db_path:
        return "Error: No encuentro la base de datos."

    try:
        db = lancedb.connect(db_path)
        tbl = db.open_table("documentos")
        
        # OJO: LangChain tiene su propio embedding, pero para no liar dependencias
        # usaremos el cliente 'raw' de google solo para embeddear la query rápido
        import google.genai as genai
        raw_client = genai.Client(api_key=GOOGLE_API_KEY)
        q_res = raw_client.models.embed_content(model="text-embedding-004", contents=query)
        q_vec = [float(x) for x in q_res.embeddings[0].values]
        
        results = tbl.search(q_vec).limit(3).to_pandas()
        return "\n".join([f"- {row['text']}" for _, row in results.iterrows()])
        
    except Exception as e:
        return f"Error en DB: {e}"

@tool
def calcular_horas_estudio(semanas: int, horas_diarias: float) -> str:
    """
    Útil para realizar cálculos matemáticos sobre tiempo de estudio.
    """
    print(f"\n   🦜 [LangChain] Calculadora invocada: {semanas} sem, {horas_diarias}h")
    total = weeks = semanas * 7 * horas_diarias
    return f"El cálculo total es de {total} horas."

# --- ARQUITECTURA DEL AGENTE ---

def main():
    print("--- AGENTE LANGCHAIN (ABSTRACCIÓN) ---")
    
    # 1. El Cerebro (LLM)
    # LangChain maneja los reintentos y protocolos internamente
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash", # Intentamos el modelo estándar
        temperature=0,
        google_api_key=GOOGLE_API_KEY
    )

    # 2. Las Herramientas
    tools = [consultar_knowledge_base, calcular_horas_estudio]

    # 3. El Prompt (System Instruction)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Eres un asistente inteligente. Usa tus herramientas si es necesario. Si no, responde directamente."),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"), # Aquí LangChain inyecta su "pensamiento"
    ])

    # 4. El Ensamblaje (Wiring)
    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)

    # 5. Loop de Interacción
    while True:
        user_input = input("\nUsuario: ")
        if user_input.lower() in ["salir", "exit"]:
            break
            
        try:
            # LangChain gestiona el bucle de "Pensar -> Ejecutar Tool -> Volver a pensar -> Responder"
            response = agent_executor.invoke({"input": user_input})
            print(f"🤖 Agente: {response['output']}")
            
        except Exception as e:
            print(f"❌ Error de LangChain: {e}")
            print("💡 Pista: Si es un 404, prueba a cambiar el modelo a 'gemini-pro'")

if __name__ == "__main__":
    main()