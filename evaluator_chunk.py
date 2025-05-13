
from loguru import logger
import os
import nest_asyncio
import asyncio
from llama_index.core import SimpleDirectoryReader
from llama_index.llms.openai import OpenAI
from llama_index.core.evaluation import (
    DatasetGenerator,
    FaithfulnessEvaluator,
    RelevancyEvaluator
)

from evaluate.utils.cache_chunksize import check_cache_questions, load_cache_questions, save_to_cache_questions
from config.eval_setting import EvalChunkSettings
from config.logging_setting import setup_logger

logger = setup_logger(f"logs/evaluate/chunksize")

from dotenv import load_dotenv
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

nest_asyncio.apply()
async def evaluate():
    pass

def run_evaluator(doc_dir:str, required_exts:list, path_questions:str, nr_of_questions:int = 5) -> None:
    documents = SimpleDirectoryReader(input_dir=doc_dir, required_exts=required_exts).load_data()
    cache_path_filename = check_cache_questions(path_questions=path_questions, nr_of_questions=nr_of_questions)
    if os.path.exists(cache_path_filename):
        logger.info(f"Loading cached questions from: {cache_path_filename}")
        eval_questions = load_cache_questions(cache_path_filename)
    else:
        logger.info(f"Saving questions to cache: {cache_path_filename}")
        data_generator = DatasetGenerator.from_documents(documents)
        eval_questions = data_generator.generate_questions_from_nodes(num=nr_of_questions)
        save_to_cache_questions(cache_path_filename, eval_questions)
    logger.info(f"Generated {len(eval_questions)} questions.")
    logger.info(f"Display first two questions: {eval_questions[:2]}")


if __name__ == "__main__":
    logger.info("Starting evaluation...")
    params = EvalChunkSettings()
    run_evaluator(doc_dir=params.documents_path, 
                required_exts=[".pdf"],
                path_questions=params.questions_path,
                nr_of_questions=params.nr_of_questions)
    logger.info("Evaluation completed.")