from typing import List
from config.embedding.model_setting import EmbeddingSettings
from config.llm.model_setting import LLMSettings
from config.prompt.template_setting import PromptTemplate
from config.index.store_setting import StoreIndex

class Settings(EmbeddingSettings, LLMSettings, PromptTemplate, StoreIndex):
    """General Settings class that combines all settings for the application."""
    sentence_splitter_chunk: int = 1000
    list_tools: List[str] = ["Base", "Meta", "Summary"]  # Could be ['Base', 'Meta', 'Summary']
