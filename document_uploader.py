from global_settings import STORAGE_PATH, CACHE_FILE
from logging_functions import log_action
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.ingestion import IngestionPipeline, IngestionCache
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core.extractors import SummaryExtractor
from llama_index.embeddings.openai import OpenAIEmbedding

def ingest_documents():
    documents = SimpleDirectoryReader(
        STORAGE_PATH,
        filename_as_id=True
    ).load_data()
    for doc in documents:
        print(doc.id)
        log_action(
            f"File'{doc.id_}'uploaded_user",
            action_type="UPLOAD"
        )

    try:
        cached_hashes = IngestionCache.from_persist_path(
            CACHE_FILE
        )
        print("Cache file found. Running using cache..")
    except:
        cached_hashes = ""
        print("No cache file found. Running without..")
    pipeline = IngestionPipeline(
        transformations=[
            TokenTextSplitter(
                chunk_size=1024,
                chunk_overlap=20
            ),
            SummaryExtractor(summaries=['self']),
            OpenAIEmbedding()
        ],
        cache=cached_hashes
    )
    nodes = pipeline.run(documents=documents)
    pipeline.cache.persist(CACHE_FILE)

    return nodes

if __name__ == "__main__":
    embedded_nodes = ingest_documents()