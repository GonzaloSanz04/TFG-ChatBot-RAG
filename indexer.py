import os
import requests                             # Para llamar a la API de CKAN
import schedule                             # Para el trabajo periódico
import time
import io                                   # Para manejar los bytes del PDF descargado
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk      # Para inserción en lote
from sentence_transformers import SentenceTransformer 
from markitdown import MarkItDown
from langchain_text_splitters import RecursiveCharacterTextSplitter

#    Configuración Global

# Configuración de Elasticsearch
ELASTIC_URL = "http://localhost:9200"
INDEX_NAME = "guias_docentes"   

# Configuración del Modelo de Embedding (debe coincidir con el mapping)
MODEL_NAME = 'all-MiniLM-L6-v2'
EMBEDDING_DIM = 384

# Configuración de la API de CKAN (la fuente de datos)
CKAN_API_URL = "https://portal.guia.linkeddata.es/api/3/action/resource_search"

#     Funciones de Elasticsearch

def connect_to_elastic():
    """Se conecta a Elasticsearch y devuelve el cliente."""
    print(f"Conectando a Elasticsearch en {ELASTIC_URL}...")
    try:
        client = Elasticsearch(
            [{"host": "localhost", "port": 9200, "scheme": "http"}],
            verify_certs=False,
            ssl_show_warn=False,
            request_timeout=5 
        )
        info = client.info()
        print("Conexión con Elasticsearch exitosa")
        print(f"Versión del clúster: {info['version']['number']}")
        return client
    except Exception as e:
        print(f"Error conectando a Elasticsearch: {e}")
        return None

def create_index_mapping(client):
    if client.indices.exists(index=INDEX_NAME):
        print(f"El índice '{INDEX_NAME}' ya existe.")
        return

    mapping_body = {
        "mappings": {
            "properties": {
                # ID del recurso PDF original de CKAN
                "document_id": { "type": "keyword" }, 
                
                # URL del PDF original
                "document_url": { "type": "keyword" },
                
                # Texto del chunk
                "chunk_text": { "type": "text" }, 
                
                # Vector de embedding del chunk
                "embedding_vector": {
                    "type": "dense_vector",
                    "dims": EMBEDDING_DIM 
                },
                
                # Sello de tiempo de la API de CKAN
                "modified_date": { "type": "date" } 
            }
        }
    }
    try:
        client.indices.create(index=INDEX_NAME, body=mapping_body)
        print(f"Índice '{INDEX_NAME}' creado con éxito.")
    except Exception as e:
        print(f"Error al crear el índice: {e}")

#    Lógica de Procesamiento

def convert_pdf_to_markdown(pdf_content_bytes):
    """Convierte el contenido de un PDF (en bytes) a Markdown."""
    md_converter = MarkItDown(enable_plugins=False)
    try:
        with io.BytesIO(pdf_content_bytes) as f:
            result = md_converter.convert_stream(f)
        return result.text_content
    except Exception as e:
        print(f"Error al convertir PDF con MarkItDown: {e}")
        return None

def get_chunks_from_markdown(markdown_content):
    """Divide el Markdown usando RCTS."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500, 
        chunk_overlap=150,
        separators=["\n# ", "\n## ", "\n\n", ".\n", "\n", " ", ""]
    )
    return text_splitter.split_text(markdown_content)

#    Lógica del Sincronizador

def fetch_ckan_resources():
    """Obtiene la lista de todos los recursos PDF de la API."""
    params = {"query": "mimetype:application/pdf", "limit": 1000} 
    print(f"Llamando a la API de CKAN: {CKAN_API_URL}")
    try:
        response = requests.get(CKAN_API_URL, params=params)
        response.raise_for_status()
        data = response.json()
        
        if data.get("success"):
            results = data["result"]["results"]
            print(f"CKAN reporta {len(results)} recursos PDF.")
            return results
        else:
            print(f"La API de CKAN devolvió un error: {data.get('error')}")
            return []
    except requests.RequestException as e:
        print(f"Error al llamar a la API de CKAN: {e}")
        return []

def get_stored_version(client, document_id):
    """Busca en ES la fecha de modificación de un documento ya indexado."""
    try:
        query = {
            "query": { "term": { "document_id": document_id } },
            "sort": { "modified_date": "desc" },    # Ordena por fecha
            "_source": ["modified_date"],           # Obtenemos la fecha
            "size": 1                               # Nos quedamos el más reciente
        }
        response = client.search(index=INDEX_NAME, body=query)
        if response["hits"]["total"]["value"] > 0:
            return response["hits"]["hits"][0]["_source"]["modified_date"]
    except Exception as e:
        pass
    return None # Si no existe, devolvemos None

def delete_document_chunks(client, document_id):
    """Elimina todos los chunks antiguos de un documento antes de re-indexar."""
    print(f"-> Eliminando versiones antiguas de: {document_id}")
    try:
        query = { "query": { "term": { "document_id": document_id } } }
        # delete_by_query es la forma de borrar documentos que coincidan con una búsqueda
        client.delete_by_query(index=INDEX_NAME, body=query, refresh=True)
    except Exception as e:
        print(f"Error eliminando chunks antiguos: {e}")

def process_and_index_document(client, model, resource):
    """
    Descarga, procesa (PDF->MD->Chunks->Vectores) e indexa
    un solo documento en Elasticsearch.
    """
    doc_id = resource['id']
    doc_url = resource['url']
    mod_date = resource['metadata_modified']    # "Sello" de version
    
    # 1. Descargar el PDF
    print(f"-> Descargando: {doc_url}")
    try:
        pdf_response = requests.get(doc_url, timeout=30)    # 30s de timeout
        pdf_response.raise_for_status()
        pdf_content = pdf_response.content                  # Contenido implicito del PDF
    except requests.RequestException as e:
        print(f"ERROR al descargar {doc_url}: {e}")
        return # Saltamos este documento

    # 2. Procesar (MarkItDown y Chunking)
    print("-> Procesando con MarkItDown...")
    markdown_text = convert_pdf_to_markdown(pdf_content)    
    if not markdown_text:
        print("ERROR: MarkItDown no devolvió contenido.")
        return 
    
    print("-> Dividiendo en Chunks (RCTS)...")
    chunks = get_chunks_from_markdown(markdown_text)        # Contenido dividido según 
                                                            # ["\n# ", "\n## ", "\n\n", ".\n", "\n", " ", ""]
    if not chunks:
        print("ERROR: No se generaron chunks.")
        return
        
    # 3. Vectorizar (Embedding)
    print(f"-> Vectorizando {len(chunks)} chunks...")
    embeddings = model.encode(chunks, show_progress_bar=False)

    # 4. Preparar para Elasticsearch (Inserción en Lote)
    actions = []
    for i, chunk in enumerate(chunks):
        action = {
            "_index": INDEX_NAME, 
            "_source": {
                "document_id": doc_id,
                "document_url": doc_url,
                "chunk_text": chunk,
                "embedding_vector": embeddings[i],
                "modified_date": mod_date
            }
        }
        actions.append(action)

    # 5. Insertar en Lote (Bulk Insert)
    if actions:
        print(f"-> Indexando {len(actions)} chunks en Elasticsearch...")
        try:
            bulk(client, actions, refresh=True) # refresh=True lo hace visible inmediatamente
            print(f"¡Éxito! Documento {doc_id} indexado.")
        except Exception as e:
            print(f"ERROR durante la indexación en lote (bulk): {e}")

def run_sync_job(client, model):
    """
    El trabajo principal: Compara CKAN con ES y actualiza lo necesario.
    """
    print(f"\n--- [ {time.ctime()} ] ---")
    print("Iniciando trabajo de sincronización de guías docentes...")
    
    ckan_resources = fetch_ckan_resources()
    if not ckan_resources:
        print("No se obtuvieron recursos de CKAN. Finalizando trabajo.")
        return

    total = len(ckan_resources)
    for i, resource in enumerate(ckan_resources):
        doc_id = resource.get('id')
        ckan_mod_date = resource.get('metadata_modified')
        doc_name = resource.get('name', 'Nombre Desconocido')
        
        print(f"\nProcesando documento {i+1}/{total}: {doc_name} ({doc_id})")
        
        if not doc_id or not ckan_mod_date or resource.get('mimetype') != 'application/pdf':
            print("-> Omitido (recurso sin ID, fecha, o no es PDF).")
            continue 

        stored_mod_date = get_stored_version(client, doc_id)    # Busca sello de versión del doc
        
        if not stored_mod_date:                                 # No tiene sello de versión         -> Es nuevo
            print(f"-> Documento NUEVO encontrado.")
            process_and_index_document(client, model, resource)
        elif ckan_mod_date > stored_mod_date:                   # Tiene sello de versión antiguo    -> Lo reemplazamos
            print(f"-> Documento ACTUALIZADO encontrado (CKAN: {ckan_mod_date} > Local: {stored_mod_date}).")
            delete_document_chunks(client, doc_id)
            process_and_index_document(client, model, resource)
        else:                                                   # Tiene el mismo sello de versión   -> Lo omitimos pues ya está actualizado
            print(f"-> Documento ya está actualizado. Omitiendo.")
            pass
    
    print("\n--- [ Trabajo de sincronización finalizado ] ---")

#    Ejecución Principal

if __name__ == "__main__":
    
    es_client = connect_to_elastic()
    
    if es_client:
        # 1. Asegurarse de que el mapping existe
        create_index_mapping(es_client)
        
        # 2. Cargar el modelo de embedding (solo una vez)
        print(f"\nCargando modelo de embedding ({MODEL_NAME}) en memoria...")
        # (Esto puede tardar un poco la primera vez)
        embedding_model = SentenceTransformer(MODEL_NAME)
        print("Modelo de embedding cargado y listo.")
        
        # 1. Ejecutar el trabajo una vez inmediatamente al arrancar
        print("\nEjecutando la primera sincronización al inicio...")
        run_sync_job(es_client, embedding_model)
        
        # 2. Programar el trabajo para que se ejecute cada X tiempo     -> Para pruebas, poner .every(1).minutes
        print("\nProgramando el trabajo para ejecutarse cada 24 horas...")
        schedule.every(24).hours.do(run_sync_job, client=es_client, model=embedding_model)
        
        # 3. Bucle infinito para mantener el script vivo y funcionando
        print("El indexador está ahora en modo 'schedule'. Presiona Ctrl+C para salir.")
        while True:
            schedule.run_pending()
            time.sleep(60) # Dormir por un minuto antes de comprobar nuevamente
            
    else:
        print("\nERROR: No se pudo conectar a Elasticsearch. Saliendo.")
