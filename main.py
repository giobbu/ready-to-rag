from basic_rag import RAGgish
from config import EmbeddingSettings, LLMSettings, Settings, PromptTemplate
from loguru import logger
import argparse
from utils import exist_QA_files

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
                    chunk_size=Settings.sentence_splitter_chunk
                    )
    if EmbeddingSettings.run_finetuning:
        if not exist_QA_files(EmbeddingSettings):
            embed_model_finetuned = basic_rag.finetune_embeddings(input_dir_train=EmbeddingSettings.input_dir_train,
                            input_dir_val=EmbeddingSettings.input_dir_val, 
                            prompt_tmpl=PromptTemplate.prompt_tmpl,
                            save_train_path=EmbeddingSettings.out_dir_train,
                            save_val_path=EmbeddingSettings.out_dir_val)
            logger.info('Sentence Tranformer Loaded and Finetuned!')
        else:
            try:
                logger.info('Loading finetuned model...')
                embed_model_finetuned = basic_rag.load_finetuned_model(EmbeddingSettings.embed_name, 
                                                                    EmbeddingSettings.model_output_path)
            except Exception as e:
                logger.error(f"Error loading finetuned model: {e}")
                raise ValueError("Please check the model_output_path in the config file")
    logger.info(' ')
    logger.info('----------------- Testing ---------------------------')
    logger.info('Loading data...')
    documents = basic_rag.load_data(input_dir="data/", required_exts=[".pdf"])  # Load data
    
    logger.info('Parsing documents...')
    nodes = basic_rag.parse_documents(documents)  # Parse documents into nodes
    
    logger.info('Creating index...')
    if EmbeddingSettings.use_finetuned_model:
        logger.info('Using finetuned model...')
        vector_index = basic_rag.create_index(nodes, embed_model_finetuned)  # Create an index
    else:
        logger.info('Using base model...')
        vector_index = basic_rag.create_index(nodes, basic_rag.embed_model)
    
    logger.info('Creating query engine...')
    basic_rag.create_query_engine(vector_index)  # Create a query engine
    
    logger.info('Querying the engine...')
    response = basic_rag.answer(query)  # Query the engine


