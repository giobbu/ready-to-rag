from pydantic import BaseModel

class LLMSettings(BaseModel):
    llm_name: str = "gpt-3.5-turbo"
    temperature: float = 0.0