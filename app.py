import os
import json
import requests
import warnings
from abc import ABC, abstractmethod 
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
from google import genai
from google.genai import types
from groq import Groq

#    CONFIGURACION GLOBAL

ELASTIC_URL = "http://localhost:9200"
INDEX_NAME = "guias_docentes"
MODEL_NAME = 'all-MiniLM-L6-v2'

try:
    with open('config.json', 'r') as f:
        CONFIG = json.load(f)
except FileNotFoundError:
    print("ERROR: No se encuentra el archivo config.json.")
    exit()

#   DEFINICIÓN DE LA INTERFAZ ABSTRACTA

class LLMProvider(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str:
        pass

#   IMPLEMENTACIONES CONCRETAS DE LOS PROVEEDORES

class GeminiProvider(LLMProvider):
    def __init__(self, api_key, model_name):
        self.api_key = api_key
        self.model_name = model_name
        
    def generate(self, prompt: str) -> str:
        if not self.api_key:
            return "ERROR: Clave GEMINI_API_KEY no configurada."
        try:
            client = genai.Client(api_key=self.api_key)
            # Instrucción del sistema (específica de Gemini)
            sys_instr = "Eres un Asistente de Guías Docentes. Responde basándote SOLO en el contexto."
            
            response = client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=sys_instr,
                    temperature=0.0
                )
            )
            return response.text
        except Exception as e:
            return f"ERROR Gemini: {e}"

class GroqProvider(LLMProvider):
    def __init__(self, api_key, model_name):
        self.api_key = api_key
        self.model_name = model_name

    def generate(self, prompt: str) -> str:
        if not self.api_key:
            return "ERROR: Clave GROQ_API_KEY no configurada."
        try:
            client = Groq(api_key=self.api_key)
            sys_instr = "Eres un Asistente de Guías Docentes. Responde basándote SOLO en el contexto."
            
            chat = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": sys_instr},
                    {"role": "user", "content": prompt}
                ],
                model=self.model_name,
                temperature=0.0
            )
            return chat.choices[0].message.content
        except Exception as e:
            return f"ERROR Groq: {e}"

class OllamaProvider(LLMProvider):
    def __init__(self, api_url, model_name):
        self.api_url = api_url
        self.model_name = model_name

    def generate(self, prompt: str) -> str:
        try:
            # Ollama no usa system instruction, así que lo pegamos al prompt
            sys_instr = "Eres un Asistente de Guías Docentes. Responde basándote SOLO en el contexto.\n\n"
            full_prompt = sys_instr + prompt
            
            response = requests.post(
                self.api_url,
                json={
                    "model": self.model_name, 
                    "prompt": full_prompt, 
                    "stream": False,
                    "options": {"temperature": 0.0}
                },
                timeout=300
            )
            response.raise_for_status()
            return response.json()["response"]
        except Exception as e:
            return f"ERROR Ollama: {e}"

#   FACTORY 

def get_llm_provider() -> LLMProvider:
    """
    Lee la configuración y devuelve la instancia de la clase correcta.
    """
    active_llm = CONFIG["active_llm"]
    options = CONFIG["llm_options"].get(active_llm)
    
    if not options:
        raise ValueError(f"Configuración no encontrada para: {active_llm}")

    if active_llm == "gemini":
        return GeminiProvider(
            api_key=options.get("api_key"),
            model_name=options["model"]
        )
    elif active_llm == "groq":
        return GroqProvider(
            api_key=options.get("api_key"),
            model_name=options["model"]
        )
    elif active_llm == "ollama":
        return OllamaProvider(
            api_url=options["api_url"],
            model_name=options["model"]
        )
    else:
        raise ValueError(f"Proveedor desconocido: {active_llm}")

#   Conexión con la base de datos
def connect_to_elastic():
    print(f"Conectando a Elasticsearch en {ELASTIC_URL}...")
    try:
        warnings.filterwarnings("ignore", "Connecting to",)
        client = Elasticsearch(
            [{"host": "localhost", "port": 9200, "scheme": "http"}],
            verify_certs=False, ssl_show_warn=False, request_timeout=10 
        )
        if client.info():
            print("¡Conexión con Elasticsearch exitosa!")
            return client
    except Exception as e:
        print(f"Error conectando a Elasticsearch: {e}")
        return None

#   Cargar el embedding
def load_embedding_model():
    print(f"Cargando modelo de embedding...")
    try:
        return SentenceTransformer(MODEL_NAME)
    except Exception as e:
        print(f"Error al cargar modelo: {e}")
        return None

#   Búsqueda de los chunks más relevantes
def search_retriever(client, model, query_text, top_k=8):
    try:
        query_vector = model.encode(query_text).tolist()
        knn_query = {
            "knn": {
                "field": "embedding_vector",
                "query_vector": query_vector,
                "k": top_k,
                "num_candidates": 20
            },
            "_source": ["chunk_text", "document_url"]
        }
        response = client.search(index=INDEX_NAME, body=knn_query)
        hits = response['hits']['hits']
        context_chunks = [hit['_source']['chunk_text'] for hit in hits]
        sources = list(set(hit['_source']['document_url'] for hit in hits))
        return context_chunks, sources
    except Exception as e:
        print(f"Error búsqueda: {e}")
        return [], []

#   Generar el prompt
def build_rag_prompt(query, context_chunks):
    context = "\n---\n".join(context_chunks)
    return f"""
    CONTEXTO:
    ---
    {context}
    ---
    PREGUNTA: {query}
    RESPUESTA (basada solo en el contexto):
    """

#   BUCLE DEL CHATBOT

if __name__ == "__main__":
    
    es_client = connect_to_elastic()
    embedding_model = load_embedding_model()
    
    try:
        llm_engine = get_llm_provider() # Cargamos el LLM activo 
        print(f"Model cargado: {type(llm_engine).__name__}")
    except Exception as e:
        print(f"Error al configurar LLM: {e}")
        exit()

    if not es_client or not embedding_model:
        exit()

    print("\n" + "="*50)
    print(f"   Asistente RAG (LLM Activo: {CONFIG['active_llm']})")    
    print("="*50)

    try:
        while True:
            user_query = input("\n[Pregunta]: ")
            if user_query.lower() in ['salir', 'exit']: break
            
            print("... recuperando contexto ...")
            chunks, sources = search_retriever(es_client, embedding_model, user_query, top_k=8)
            
            if not chunks:
                print("No se encontró información relevante.")
                continue

            # Usamos el método generate() de la interfaz abstracta
            print("... generando respuesta ...")
            prompt = build_rag_prompt(user_query, chunks)
            answer = llm_engine.generate(prompt)
            
            print("\n[Respuesta]:")
            print(answer)
            
            if sources:
                print("\nFuentes:")
                for url in sources: print(f"- {url}")

    except KeyboardInterrupt:
        print("\nCerrando...")
