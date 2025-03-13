from basic_rag import RAGgish
from config import EmbeddingSettings, LLMSettings, Settings
from loguru import logger
import argparse

if __name__== "__main__":
    logger.info('Parsing command line arguments...')
    parser = argparse.ArgumentParser(description="Raggish: A RAG model for question answering")
    parser.add_argument("--query", type=str, help="Query string")
    try:
        query = parser.parse_args().query
    except:
        raise ValueError("Please provide a query string using the --query flag")
    logger.info('Loading Raggish model...')
    basic_rag = RAGgish(embed_name=EmbeddingSettings.embed_name,
                    llm_name=LLMSettings.llm_name,
                    context_window=LLMSettings.context_window,
                    max_new_tokens=LLMSettings.max_new_tokens,
                    generate_kwargs=LLMSettings.generate_kwargs,
                    chunk_size=Settings.sentence_splitter_chunk
                    )
    logger.info('Loading data...')
    documents = basic_rag.load_data(input_dir="data/", required_exts=[".pdf"])  # Load data
    logger.info('Parsing documents...')
    nodes = basic_rag.parse_documents(documents)  # Parse documents into nodes
    logger.info('Creating index...')
    vector_index = basic_rag.create_index(nodes)  # Create a vector store index
    logger.info('Creating query engine...')
    basic_rag.create_query_engine(vector_index)  # Create a query engine
    logger.info('Querying the engine...')
    response = basic_rag.answer(query)  # Query the engine


