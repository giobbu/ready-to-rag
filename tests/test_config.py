
from config import EmbeddingSettings, LLMSettings
import numpy as np

def test_config():
    " Test the config settings "
    assert isinstance(EmbeddingSettings.embed_name, str)
    assert isinstance(LLMSettings.llm_name, tuple)
    assert isinstance(LLMSettings.context_window, tuple)
    assert isinstance(LLMSettings.max_new_tokens, tuple)
    assert isinstance(LLMSettings.generate_kwargs, dict)
    assert np.isclose(LLMSettings.generate_kwargs["temperature"], 0.7)
    assert LLMSettings.generate_kwargs["top_k"] == 50
    assert np.isclose(LLMSettings.generate_kwargs["top_p"], 0.95)
    assert LLMSettings.generate_kwargs["do_sample"] == True