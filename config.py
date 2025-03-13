from dataclasses import dataclass

@dataclass(frozen=True)
class EmbeddingSettings:
    embed_name = "BAAI/bge-small-en-v1.5"  #BAAI/bge-small-en

@dataclass(frozen=True)
class LLMSettings:
    llm_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    context_window = 2048,
    max_new_tokens = 156,
    generate_kwargs = {"temperature": 0.7, "top_k": 50, "top_p": 0.95, "do_sample": True}

@dataclass(frozen=True)
class Settings:
    sentence_splitter_chunk = 1000
