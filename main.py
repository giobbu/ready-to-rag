from basic_rag import RAGgish
from config import Settings
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
    basic_rag = RAGgish(embed_name=Settings.embed_name, llm_name=Settings.llm_name)
    if Settings.run_finetuning:
        if not exist_QA_files(Settings):
            parse_file = True
            embed_model_finetuned = basic_rag.finetune_embeddings(input_dir_train=Settings.input_dir_train,
                            input_dir_val=Settings.input_dir_val, 
                            prompt_tmpl=Settings.prompt_tmpl,
                            chunk_size=Settings.sentence_splitter_chunk,
                            save_train_path=Settings.out_dir_train,
                            save_val_path=Settings.out_dir_val,
                            save_model_path=Settings.model_output_path,
                            parse_files=parse_file)
            logger.info('Finetuning completed. Please check the output files.')
        else:
            parse_file = False
            embed_model_finetuned = basic_rag.finetune_embeddings(input_dir_train=Settings.input_dir_train,
                                                                    input_dir_val=Settings.input_dir_val, 
                                                                    prompt_tmpl=Settings.prompt_tmpl,
                                                                    chunk_size=Settings.sentence_splitter_chunk,
                                                                    save_train_path=Settings.out_dir_train,
                                                                    save_val_path=Settings.out_dir_val,
                                                                    save_model_path=Settings.model_output_path,
                                                                    parse_files=parse_file)
            logger.info('Finetuning completed. Please check the output files.')
    else:
        try:
            logger.info('Loading finetuned model...')
            embed_model_finetuned = basic_rag.load_finetuned_model(Settings.embed_name, 
                                                                Settings.model_output_path)
        except Exception as e:
            logger.error(f"Error loading finetuned model: {e}")
            raise ValueError("Please check the model_output_path in the config file")
    
    logger.info(' ')
    logger.info('----------------- Testing ---------------------------')
    
    logger.info('Loading data...')
    documents = basic_rag.load_data(input_dir=Settings.input_dir_test,
                                    required_exts=[".pdf"])  # Load data
    
    logger.info('Parsing documents...')
    nodes = basic_rag.parse_documents(documents, 
                                    chunk_size=Settings.sentence_splitter_chunk)  # Parse documents into nodes
    
    logger.info('Creating indexes...')
    if Settings.use_finetuned_model:
        logger.info('Using finetuned model...')
        vector_index = basic_rag.create_vector_index(nodes, embed_model_finetuned)  # Create an index
        summary_index = basic_rag.create_summary_index(nodes, embed_model_finetuned)  # Create a summary index
    else:
        logger.info('Using base model...')
        vector_index = basic_rag.create_vector_index(nodes, basic_rag.embed_model)
        summary_index = basic_rag.create_summary_index(nodes, basic_rag.embed_model)

    list_tools = []
    for tools in Settings.list_tools:
        if tools == 'Base':
            list_tools.append(basic_rag.create_base_query_tool(vector_index))
            logger.info('Creating base query engine...')
        elif tools == 'Meta':
            list_tools.append(basic_rag.create_metadata_query_tool(vector_index))
            logger.info('Creating metadata query engine...')
        elif tools == 'Summary':
            list_tools.append(basic_rag.create_summary_query_tool(summary_index))
            logger.info('Creating summary query engine...')
        else:
            raise ValueError(f"Tool {tools} not recognized. Please check the config file.")
    response = basic_rag.answer(query, list_tools)
    
    


