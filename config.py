from pydantic import BaseModel, Field
from typing import List


class EmbeddingSettings(BaseModel):
    embed_name: str = "BAAI/bge-small-en-v1.5"
    input_dir_train: str = "data/finetune-embed/train"
    input_dir_val: str = "data/finetune-embed/val"
    input_dir_test: str = "data/"
    out_dir_train: str = "save/gpt/train_dataset_gpt.json"
    out_dir_val: str = "save/gpt/val_dataset_gpt.json"
    run_finetuning: bool = False
    use_finetuned_model: bool = True
    adapter: str = "linear"
    model_output_path: str = "model/linear_adapter_model_output"

class LLMSettings(BaseModel):
    llm_name: str = "gpt-3.5-turbo"

class PromptTemplate(BaseModel):
    prompt_tmpl: str = Field(default_factory=lambda: """
        Context information is below.

        ---------------------
        {context_str}
        ---------------------

        Given the context information and not prior knowledge, generate only questions based on the below query.

        You are a Teacher/ Professor. Your task is to setup {num_questions_per_chunk} questions for an upcoming quiz/examination.
        The questions should be diverse in nature across the document. Restrict the questions to the context information provided."
        """)

class Settings(EmbeddingSettings, LLMSettings, PromptTemplate):
    sentence_splitter_chunk: int = 1000
    list_tools: List[str] = ["Base"]  # Could be ['Base', 'Meta', 'Summary']

class EvalSettings(BaseModel):
    embed_name: str = "BAAI/bge-small-en-v1.5"
    finetune: bool = True
    path_finetuned: str = "model/linear_adapter_model_output"
    top_k: int = 2
    dataset_type_list: List[str] = ['train', 'val']