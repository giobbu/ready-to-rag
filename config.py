from dataclasses import dataclass

@dataclass(frozen=True)
class EmbeddingSettings:
    embed_name = "BAAI/bge-small-en-v1.5"
    # finetuning
    input_dir_train = "data/finetune-embed/train"
    input_dir_val = "data/finetune-embed/val"
    input_dir_test = "data/"
    out_dir_train = "save/gpt/train_dataset_gpt.json"
    out_dir_val = "save/gpt/val_dataset_gpt.json"
    run_finetuning = False
    use_finetuned_model = False
    adapter = 'linear'
    model_output_path = "model/linear_adapter_model_output",

@dataclass(frozen=True)
class LLMSettings:
    llm_name = "gpt-3.5-turbo"

@dataclass(frozen=True)
class PromptTemplate:
    prompt_tmpl = """
        Context information is below.

        ---------------------
        {context_str}
        ---------------------

        Given the context information and not prior knowledge, generate only questions based on the below query.

        You are a Teacher/ Professor. Your task is to setup {num_questions_per_chunk} questions for an upcoming quiz/examination.
        The questions should be diverse in nature across the document. Restrict the questions to the context information provided."
        """

@dataclass(frozen=True)
class Settings(EmbeddingSettings, LLMSettings, PromptTemplate):
    sentence_splitter_chunk = 1000
    list_tools = ['Base', 'Meta', 'Summary']  # ['Base', 'Meta', 'Summary']: the list of tools to be used for the embedding

@dataclass(frozen=True)
class EvalSettings:
    embed_name = "BAAI/bge-small-en-v1.5"
    finetune = True
    path_finetuned = "model/linear_adapter_model_output"
    top_k = 2
    dataset_type_list = ['train', 'val']