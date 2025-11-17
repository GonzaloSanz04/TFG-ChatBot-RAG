import os
import json 
import requests
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
from google import genai
from google.genai import types
import warnings
from groq import Groq
# --- 1. Configuración Global ---

ELASTIC_URL = "http://localhost:9200"
INDEX_NAME = "guias_docentes"
MODEL_NAME = 'all-MiniLM-L6-v2'
EMBEDDING_DIM = 384

GEMINI_API_KEY = "AIzaSyAXpfeZtBGyX1uahQmngAMHTqRlMPmylQ0"
GROQ_API_KEY = "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"

try:
    with open('config.json', 'r') as f:
        CONFIG = json.load(f)
except FileNotFoundError:
    print("ERROR: No se encuentra el archivo config.json.")
    exit()

# --- 2. Conexión y Carga de Modelos ---

def connect_to_elastic():
    """Se conecta a Elasticsearch y devuelve el cliente."""
    print(f"Conectando a Elasticsearch en {ELASTIC_URL}...")
    try:
        # Suprimir advertencias
        warnings.filterwarnings("ignore", "Connecting to",)
        
        client = Elasticsearch(
            ELASTIC_URL,
            verify_certs=False,
            ssl_show_warn=False,
            request_timeout=10 
        )
        if client.ping():
            print("¡Conexión con Elasticsearch exitosa!")
            return client
        else:
            print("No se pudo conectar con Elasticsearch.")
            return None
    except Exception as e:
        print(f"Error conectando a Elasticsearch: {e}")
        return None

def load_embedding_model():
    """Carga el modelo SentenceTransformer en memoria."""
    print(f"Cargando modelo de embedding...")
    try:
        model = SentenceTransformer(MODEL_NAME)
        print("Modelo de embedding listo.")
        return model
    except Exception as e:
        print(f"Error al cargar el modelo de embedding: {e}")
        return None

# --- 3. Lógica del RAG (Retrieve & Generate) ---

def search_retriever(client, model, query_text, top_k=5):
    """
    Vectoriza la consulta y realiza una búsqueda kNN en Elasticsearch
    para encontrar los chunks de contexto más relevantes.
    """
    
    # 1. Vectorizar la consulta
    query_vector = model.encode(query_text).tolist()

    # 2. Construir la consulta para Elasticsearch
    knn_query = {
        "knn": {
            "field": "embedding_vector", # El campo que definimos en el mapping
            "query_vector": query_vector,
            "k": top_k,
            "num_candidates": 20
        },
        "_source": ["chunk_text", "document_url"] # Solo queremos el texto y la URL
    }
    
    try:
        response = client.search(index=INDEX_NAME, body=knn_query)
        
        # 3. Parsear los resultados
        hits = response['hits']['hits']
        context_chunks = []
        sources = set() # Usamos un set para evitar URLs duplicadas
        
        for hit in hits:
            context_chunks.append(hit['_source']['chunk_text'])
            sources.add(hit['_source']['document_url'])
            
        return context_chunks, list(sources)
    
    except Exception as e:
        print(f"Error en la búsqueda de Elasticsearch: {e}")
        return [], []

def build_rag_prompt(query, context_chunks):
    """Función auxiliar para construir el prompt RAG."""
    context = "\n---\n".join(context_chunks)
    
    # Nota: Los 'system_instruction' se manejan de forma diferente en cada API.
    # Los definiremos dentro de cada función de llamada.
    
    rag_prompt = f"""
    CONTEXTO:
    ---
    {context}
    ---
    
    PREGUNTA: {query}
    
    RESPUESTA (basada solo en el contexto):
    """
    return rag_prompt

def generate_response(query, context_chunks):
    """
    Función principal modular: Lee el config y llama al LLM activo.
    """
    # Primero, comprueba si hay contexto
    if not context_chunks:
        return "Lo siento, no he podido encontrar información relevante sobre esa consulta en las guías docentes."

    # Construye el prompt
    rag_prompt = build_rag_prompt(query, context_chunks)
    
    # Lee la configuración
    active_llm = CONFIG["active_llm"]
    options = CONFIG["llm_options"][active_llm]
    
    # Decide a qué función llamar
    if active_llm == "gemini":
        # Usa la clave API definida globalmente
        if not GEMINI_API_KEY:
            return "ERROR: Clave GEMINI_API_KEY no definida en el script."
        return generate_with_gemini(options["model"], GEMINI_API_KEY, rag_prompt)
    
    elif active_llm == "groq":
        # Usa la clave API definida globalmente
        if not GROQ_API_KEY:
            return "ERROR: Clave GROQ_API_KEY no definida en el script."
        return generate_with_groq(options["model"], GROQ_API_KEY, rag_prompt)

    elif active_llm == "ollama":
        return generate_with_ollama(options["model"], options["api_url"], rag_prompt)

    else:
        return f"ERROR: LLM '{active_llm}' no reconocido en config.json."

def generate_with_gemini(model, api_key, rag_prompt):
    """Llama a la API de Gemini."""
    try:
        client = genai.Client(api_key=api_key)
        system_instruction = (
            "Eres un Asistente de Guías Docentes de la UPM. "
            "Basa tu respuesta *única y exclusivamente* en el CONTEXTO proporcionado. "
            "Si la información no está en el CONTEXTO, indica que no puedes responder."
        )
        response = client.models.generate_content(
            model=model,
            contents=rag_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.0
            )
        )
        return response.text
    except Exception as e:
        return f"ERROR en API Gemini: {e}"

def generate_with_groq(model, api_key, rag_prompt):
    """Llama a la API de Groq (usando la estructura de OpenAI)."""
    try:
        client = Groq(api_key=api_key)
        system_instruction = (
            "Eres un Asistente de Guías Docentes de la UPM. "
            "Basa tu respuesta *única y exclusivamente* en el CONTEXTO proporcionado. "
            "Si la información no está en el CONTEXTO, indica que no puedes responder."
        )
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": rag_prompt}
            ],
            model=model,
            temperature=0.0
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"ERROR en API Groq: {e}"

def generate_with_ollama(model, api_url, rag_prompt):
    """Llama a un servidor local de Ollama."""
    try:
        system_instruction = (
            "Eres un Asistente de Guías Docentes de la UPM. "
            "Basa tu respuesta *única y exclusivamente* en el CONTEXTO proporcionado. "
            "Si la información no está en el CONTEXTO, indica que no puedes responder."
        )
        full_prompt = f"{system_instruction}\n\n{rag_prompt}"

        response = requests.post(
            api_url,
            json={
                "model": model,
                "prompt": full_prompt,
                "stream": False,
                "options": {"temperature": 0.0}
            },
            timeout=600
        )
        response.raise_for_status()
        return response.json()["response"]
    except Exception as e:
        return f"ERROR en API Ollama (¿Está corriendo en localhost:11434?): {e}"   

# --- 4. Bucle Principal de la Aplicación ---

if __name__ == "__main__":
    
    # 1. Cargar todo al inicio
    es_client = connect_to_elastic()
    embedding_model = load_embedding_model()

    if not es_client or not embedding_model:
        print("\nError fatal al inicializar los componentes.")
        exit()

    print("\n" + "="*50)
    print(f"   Asistente RAG (LLM Activo: {CONFIG['active_llm']})")    
    print("="*50)
    print("¡Hola! Escribe tu pregunta sobre las guías docentes.")
    print("Escribe 'salir' o presiona Ctrl+C para terminar.")

    try:
        while True:
            # 2. Obtener la pregunta del usuario
            user_query = input("\n[Pregunta]: ")
            if user_query.lower() in ['salir', 'exit', 'quit']:
                break
            
            print("... buscando en la base de datos vectorial ...")
            
            # 3. Fase de Recuperación (Retrieve)
            chunks, sources = search_retriever(
                es_client, 
                embedding_model, 
                user_query,
                top_k=8 # Pedimos 5 chunks relevantes
            )
            
            print(f"... {len(chunks)} fragmentos relevantes encontrados ...")
            
            # 4. Fase de Generación (Generate)
            answer = generate_response(user_query, chunks)
            
            # 5. Mostrar la respuesta
            print("\n[Respuesta]:")
            print(answer)
            
            if sources:
                print("\nFuentes consultadas:")
                for url in sources:
                    print(f"- {url}")

    except KeyboardInterrupt:
        print("\nCerrando aplicación...")
    
    print("\n¡Hasta luego!")