import os
import lancedb
from dotenv import load_dotenv
from google import genai

# 1. Configuración
load_dotenv()
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

def buscar_contexto(query: str, db_path: str = "./lancedb_data") -> str:
    db = lancedb.connect(db_path)
    try:
        tbl = db.open_table("documentos")
    except:
        return ""

    q_res = client.models.embed_content(
        model="text-embedding-004",
        contents=query
    )
    q_vec = [float(x) for x in q_res.embeddings[0].values]

    # AUMENTAMOS EL LÍMITE A 10 CHUNKS (Para tener más contexto)
    results = tbl.search(q_vec).limit(10).to_pandas()
    
    contexto_unificado = ""
    
    print("\n--- DEBUG: LO QUE LA IA ESTÁ LEYENDO ---") 
    for i, row in results.iterrows():
        # Imprimimos los primeros 100 caracteres de cada hallazgo
        print(f"[{i}] {row['text'][:100]}...") 
        contexto_unificado += f"\nFragmento {i}: {row['text']}\n"
        
    print("----------------------------------------\n")
    return contexto_unificado

# --- CAPA DE GENERACIÓN (LLM) ---
def generar_respuesta(query: str, contexto: str):
    """El cerebro: Combina la pregunta con los datos recuperados."""
    
    if not contexto:
        return "No tengo información en mi base de datos sobre este tema."

    # PROMPT DE ARQUITECTURA (RAG)
    # Le damos personalidad y reglas estrictas (Grounding)
    prompt = f"""
    Eres un Asistente Técnico experto en IA.
    Tu misión es responder a la pregunta del usuario BASÁNDOTE SOLO en el contexto proporcionado.
    
    CONTEXTO RECUPERADO DE LA BASE DE CONOCIMIENTO:
    {contexto}
    
    PREGUNTA DEL USUARIO:
    "{query}"
    
    INSTRUCCIONES:
    1. Si la respuesta está en el contexto, explícala detalladamente.
    2. Si el contexto menciona herramientas o lenguajes específicos (Python, Java, etc.), cítalos.
    3. Si la respuesta NO está en el contexto, di: "El documento no menciona nada específico sobre eso".
    4. Ignora pies de página, cookies o texto irrelevante del contexto.
    """

    print("🤖 Generando respuesta con Gemini...")
    response = client.models.generate_content(
        model="gemini-flash-latest",
        contents=prompt
    )
    return response.text

# --- MAIN ---
def main():
    print("--- SISTEMA RAG COMPLETO (LanceDB + Gemini) ---")
    print("🧠 Memoria cargada desde ./lancedb_data")
    
    while True:
        query = input("\nPregunta al Experto (o 'salir'): ")
        if query.lower() in ['salir', 'exit']:
            break
            
        # PASO 1: RETRIEVAL (Búsqueda)
        print("🔍 Buscando en la base de datos...")
        contexto = buscar_contexto(query)
        
        if contexto:
            # PASO 2: GENERATION (Síntesis)
            respuesta = generar_respuesta(query, contexto)
            
            print("\n" + "="*50)
            print(f"RESPUESTA GENERADA:")
            print("="*50)
            print(respuesta)
            print("-" * 50)
        else:
            print("❌ Error: No se encontró la base de datos o está vacía. Ejecuta el indexador primero.")

if __name__ == "__main__":
    main()