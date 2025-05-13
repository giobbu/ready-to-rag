import pytest
import numpy as np
import os

@pytest.fixture
def load_rag():
    from basic_rag import RAGgish
    from config.setting import Settings
    params = Settings()
    basic_rag = RAGgish(embed_name=params.embed_name, llm_name=params.llm_name, temperature=params.temperature)
    return basic_rag