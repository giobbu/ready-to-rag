
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

from config.eval_setting import EvalChunkSettings
from config.logging_setting import setup_logger
params = EvalChunkSettings()
logger = setup_logger(f"logs/evaluate/chunksize")

from dotenv import load_dotenv
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

nest_asyncio.apply()
async def evaluate():
    pass

def run_evaluator(doc_dir:str, 
                  required_exts:list,
                  nr_of_questions:int = 5) -> None:
    documents = SimpleDirectoryReader(input_dir=doc_dir, 
                                        required_exts=required_exts).load_data()
    data_generator = DatasetGenerator.from_documents(documents)
    eval_questions = data_generator.generate_questions_from_nodes(num=nr_of_questions)
    logger.info(f"Generated {len(eval_questions)} questions.")
    logger.info(f"Display first 5 questions: {eval_questions[:5]}")


if __name__ == "__main__":
    logger.info("Starting evaluation...")
    run_evaluator(doc_dir=params.documents_path, 
                  required_exts=[".pdf"],
                  nr_of_questions=5)
    logger.info("Evaluation completed.")