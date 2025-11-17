import os
import numpy as np
import faiss #BD Vectorial
from sentence_transformers import SentenceTransformer #Modelo de Embedding Texto-> Vectores
from google import genai
from google.genai import types
from rouge_score import rouge_scorer
from markitdown import MarkItDown
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI

# ---------------------------------------------------------------------------------------------

DATA_DIR = "data"
MODEL_NAME = 'all-MiniLM-L6-v2' # Modelo base para empezar. Convierte las frases en vectores de 384 dimensiones
INDEX_FILE = "faiss_index.bin"
CHUNKS_FILE = "chunks.npy"
GEMINI_API_KEY = "AIzaSyAXpfeZtBGyX1uahQmngAMHTqRlMPmylQ0"
OPENAI_API_KEY = "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"

# ---------------------------------------------------------------------------------------------

# Función de Indexación
def create_index(chunks): #Vectoriza los chunks de texto y crea un índice FAISS.
    
    # 1. Cargar el modelo de Embedding
    model = SentenceTransformer(MODEL_NAME) 

    # 2. Vectorizar los Chunks
    document_embeddings = model.encode(chunks, show_progress_bar=True)
    
    # 3. Crear el Índice FAISS
    dimension = document_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(document_embeddings)
    
    # 4. Guardar el índice y los chunks para uso futuro
    faiss.write_index(index, INDEX_FILE)
    np.save(CHUNKS_FILE, np.array(chunks))
    
    return index, model

# ---------------------------------------------------------------------------------------------

# Función que convierte los pdf a Markdown
def convert_pdf_to_markdown(file_path):
    md_converter = MarkItDown(enable_plugins=False)
    
    try:
        with open(file_path, 'rb') as f:
             result = md_converter.convert_stream(f)
             
        return result.text_content
    except Exception as e:
        print(f"Error al convertir PDF con MarkItDown: {e}")
        return None

# ---------------------------------------------------------------------------------------------

# Función que splitea el Markdown haciendo chunks
def get_chunks_from_markdown(markdown_content):
    if not markdown_content:
        return []
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=100,
        separators=["\n# ",
            "\n## ",
            "\n\n",
            ".\n",
            "\n",
            " ", ""]
    )
    
    chunks = text_splitter.split_text(markdown_content)
    
    return chunks

# ---------------------------------------------------------------------------------------------

# Función de Búsqueda -> Busca los chunks más relevantes
def search_index(query, index, model, chunks, top_k):
    
    # 1. Vectorizar la Consulta (Query)
    # Se convierte la pregunta del usuario en un vector.
    query_embedding = model.encode([query])
    
    # 2. Búsqueda en FAISS
    # index.search devuelve:
    # D: Distancias (scores) de los resultados.
    # I: Índices (posiciones) de los chunks más cercanos.
    D, I = index.search(query_embedding, top_k)
    
    # 3. Recuperar los Chunks de Texto originales
    # Usamos los índices I para obtener los textos de la lista original de chunks.
    relevant_chunks = [chunks[i] for i in I[0]]
    
    # 4. Devolver resultados con su score de similitud
    results = []
    for score, chunk in zip(D[0], relevant_chunks):
        # Un score menor significa mayor similitud.
        results.append({
            "chunk": chunk,
            "score": score
        })
        
    return results

# ---------------------------------------------------------------------------------------------

# Función de Generación 
def generate_response(query, relevant_documents):
        
    # 1. Verificar la clave API
    try:
        client = genai.Client(api_key = GEMINI_API_KEY)
    except Exception as e:
        return e.__str__

    # 2. Construir el Prompt RAG con los chunks relevantes
    context = "\n---\n".join([doc['chunk'] for doc in relevant_documents])
    
    system_instruction = (
        "Eres un Asistente de Guías Docentes de una universidad española. "
        "Tu única fuente de información es el CONTEXTO proporcionado. "
        "Debes responder de manera concisa, profesional y en español, combinando la información de los textos proporcionados. "
        "Si la información NO está en el CONTEXTO, debes indicar educadamente que no puedes responder."
    )
    
    rag_prompt = f"""
    CONTEXTO (Guía Docente):
    ---
    {context}
    
    ---
    
    Pregunta del usuario: {query}
    
    Genera una respuesta basada en el CONTEXTO:
    """

    
    # 3. Llamada a la API de Gemini
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=rag_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction
            )
        )
        return response.text
    
    except Exception as e:
        return f"ERROR al llamar a la API de Gemini: {e}"
    """

    #Llamada a la API de OpenAI
    client = OpenAI(
        api_key=OPENAI_API_KEY
    )

    response = client.responses.create(
        model="gpt-5-nano",
        input=rag_prompt,
        store=True,
    )

    return response.output_text

"""
# ---------------------------------------------------------------------------------------------

# Función para evaluar las respuestas de los modelos
def evaluate_with_rouge(generated_text, reference_text):

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    scores = scorer.score(reference_text, generated_text)
    
    rouge_results = {
        'ROUGE-1 F1': scores['rouge1'].fmeasure, # Mide la superposición de unigramas (palabras individuales).
        'ROUGE-2 F1': scores['rouge2'].fmeasure, # Mide la superposición de bigramas (pares de palabras consecutivas).
        'ROUGE-L F1': scores['rougeL'].fmeasure, # Mide la superposición de la subsecuencia común más larga , capturando la fluidez y estructura.

    }
    return rouge_results

# ---------------------------------------------------------------------------------------------

# Ejecución del Prototipo
if __name__ == "__main__":
    if not os.path.exists(INDEX_FILE):
        all_chunks = []
        
        # Encontrar todos los PDFs
        pdf_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.pdf')]
        
        if not pdf_files:
            print(f"ERROR: No se encontraron archivos PDF en la carpeta '{DATA_DIR}'.")
            exit()

        for pdf_file in pdf_files:
            pdf_path = os.path.join(DATA_DIR, pdf_file)
            
            markdown_text = convert_pdf_to_markdown(pdf_path)
            
            if markdown_text:
                chunks = get_chunks_from_markdown(markdown_text)
                all_chunks.extend(chunks)
            
        if not all_chunks:
            print("FALLO CRÍTICO: No se pudo obtener contenido estructurado de ningún PDF para indexar.")
            exit()
        # Crear el índice único con todos los chunks combinados
        index, model = create_index(all_chunks)
        chunks_cargados = all_chunks
    else:
        # Cargar el índice y el modelo para la fase de búsqueda
        index = faiss.read_index(INDEX_FILE)
        model = SentenceTransformer(MODEL_NAME)
        chunks_cargados = np.load(CHUNKS_FILE, allow_pickle=True).tolist()
    
    
    # Prueba para el embedding
    user_query = "¿Quién es el coordinador y cuáles son los requisitos de nota para los exámenes?"
    
    print(f"Consulta del Usuario: '{user_query}'")
    
    relevant_documents = search_index(
        query=user_query, 
        index=index, 
        model=model, 
        chunks=chunks_cargados, 
        top_k=8 
    )

    final_answer = generate_response(
        query=user_query, 
        relevant_documents=relevant_documents
    )

    print("-----------------------------------------------------------------------------------------------------------------------------------------\n")

    print("RESPUESTA FINAL DEL CHATBOT:")
    print(final_answer)

    reference_answer = (
    "The teaching coordinator is Juan Garbajosa Sopeña."
    "The written exams (Topics 1-4 and Topics 5-8) have a weight of 30% each and"
    "They require a minimum grade of 3/10 so that their grades can be accumulated." 
    "In addition, for the global evaluation (Scenario 2 - Final Exam) the minimum grade required is 4/10."
    "And for the extraordinary call (extraordinary exam) the minimum grade required is 4/10"
    )

    rouge_scores = evaluate_with_rouge(
    generated_text=final_answer, 
    reference_text=reference_answer
)

print("-----------------------------------------------------------------------------------------------------------------------------------------\n")

print(f"Respuesta de Referencia: {reference_answer}\n")
print("Puntuaciones ROUGE (F1-Score):")
for metric, score in rouge_scores.items():
    print(f"- {metric}: {score * 100:.2f}%")
