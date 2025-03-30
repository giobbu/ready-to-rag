from dataclasses import dataclass

@dataclass(frozen=True)
class EmbeddingSettings:
    embed_name = "BAAI/bge-small-en-v1.5"
    # finetuning
    input_dir_train = "data/finetune-embed/train"
    input_dir_val = "data/finetune-embed/val"
    out_dir_train = "save/gpt/train_dataset_gpt.json"
    out_dir_val = "save/gpt/val_dataset_gpt.json"
    run_finetuning = True
    use_finetuned_model = True
    adapter = 'linear'
    model_output_path = "model/linear_adapter_model_output",

@dataclass(frozen=True)
class LLMSettings:
    llm_name = "gpt-3.5-turbo"

@dataclass(frozen=True)
class Settings:
    sentence_splitter_chunk = 1000

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
