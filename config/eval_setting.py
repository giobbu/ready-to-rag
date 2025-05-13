from pydantic import BaseModel, Field
from typing import List

class EvalEmbeddingSettings(BaseModel):
    embed_name: str = "BAAI/bge-small-en-v1.5"
    finetune: bool = True
    path_finetuned: str = "save/embedding/baai/linear_adapter_model_output"
    cache_path:str = "evaluate/cache_embedding"
    top_k: int = 2
    dataset_path_list: List[str] = ["save/qa/train_dataset_gpt.json", 
                                    "save/qa/val_dataset_gpt.json"]
    

class EvalChunkSettings(BaseModel):
    documents_path: str = "data/finetune-embed/train"
    questions_path: str = "evaluate/cache_chunksize/questions"
    nr_of_questions: int = 5
