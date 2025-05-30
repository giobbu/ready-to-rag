from basic_rag import RAGgish
from config.setting import Settings
from loguru import logger
from config.logging_setting import setup_logger
import argparse
from utils import exist_QA_files

params = Settings()
logger = setup_logger(path_to_save="logs/operation")

if __name__== "__main__":
    parser = argparse.ArgumentParser(description="Raggish: A RAG model for question answering")
    parser.add_argument("--query", type=str, help="Query string")
    try:
        query = parser.parse_args().query
    except:
        raise ValueError("Please provide a query string using the --query flag")
    logger.info(' ')
    logger.info('----------------- RAGgish ---------------------------')
    logger.info(' ')
    logger.info('------------ Loading RAG Configurations')
    basic_rag = RAGgish(embed_name=params.embed_name, llm_name=params.llm_name, temperature=params.temperature)
    if params.run_finetuning:
        logger.info(' ')
        logger.info('----------------- Finetuning ---------------------------')
        logger.info(' ')
        parse_file = False if exist_QA_files(params) else True
        embed_model_finetuned = basic_rag.finetune_embeddings(input_dir_train=params.input_dir_train,
                                                                input_dir_val=params.input_dir_val, 
                                                                prompt_tmpl=params.prompt_tmpl,
                                                                chunk_size=params.sentence_splitter_chunk,
                                                                sent_transf_params=params.sent_transf_params,
                                                                save_model_path=params.model_output_path,
                                                                save_train_path=params.out_dir_train,
                                                                save_val_path=params.out_dir_val,
                                                                parse_files=parse_file)
        logger.info('Finetuning completed. Please check the output files.')
    else:
        try:
            logger.info('Loading finetuned embedding model...')
            embed_model_finetuned = basic_rag.load_finetuned_model(params.embed_name,
                                                                params.model_output_path)
        except Exception as e:
            logger.error(f"Error loading finetuned model: {e}")
            raise ValueError("Please check the model_output_path in the config file")
    logger.info(' ')
    logger.info('----------------- Operation ---------------------------')
    logger.info('------------ Loading data')
    documents = basic_rag.load_data(input_dir=params.input_dir_test, required_exts=[".pdf"])
    logger.info('------------ Parsing documents')
    nodes = basic_rag.parse_documents(documents, chunk_size=params.sentence_splitter_chunk)
    logger.info('------------ Creating OR Loading Indexes')
    if params.use_finetuned_model:
        logger.info('Using finetuned model')
        vector_index = basic_rag.create_or_load_vector_idx(nodes=nodes,
                                                            vec_store_path=params.vec_store_idx_dir,
                                                            vec_store_idx=params.vec_store_idx_name,
                                                            embed_model=embed_model_finetuned)                                          
        summary_index = basic_rag.create_or_load_summary_idx(nodes=nodes,
                                                            summary_path=params.summ_idx_dir,
                                                            summary_idx=params.summ_idx_name,
                                                            embed_model=embed_model_finetuned)
    else:
        logger.info('Using non-finetuned model')
        vector_index = basic_rag.create_or_load_vector_idx(nodes=nodes,
                                                            vec_store_path=params.vec_store_idx_dir,
                                                            vec_store_idx=params.vec_store_idx_name)
        summary_index = basic_rag.create_or_load_summary_idx(nodes=nodes,
                                                            summary_path=params.summ_idx_dir,
                                                            summary_idx=params.summ_idx_name)
    logger.info('------------ Creating Tools')
    list_tools = []
    for tools in params.list_tools:
        if tools == 'Base':
            list_tools.append(basic_rag.create_base_query_tool(vector_index))
        elif tools == 'Meta':
            list_tools.append(basic_rag.create_metadata_query_tool(vector_index))
        elif tools == 'Summary':
            list_tools.append(basic_rag.create_summary_query_tool(summary_index))
        else:
            raise ValueError(f"Tool {tools} not recognized. Please check the config file.")
    
    logger.info('------------ Synthetizing response')
    response = basic_rag.answer(query, list_tools)
    
    


