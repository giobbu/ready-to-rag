from pydantic import BaseModel, Field
from typing import List

class BasicOutput(BaseModel):
    """Output containing the response, page numbers, and confidence."""
    response: str = Field(..., 
                        description="The answer to the question.")
    confidence: float = Field(...,
                            description="Confidence value between 0-1 of the correctness of the result.",
    )
    confidence_explanation: str = Field(
        ..., description="Explanation for the confidence score"
    )

class MetaVectorOutput(BasicOutput):
    """Output containing the response, page numbers, and confidence."""
    page_numbers: List[int] = Field(...,
                                    description="The page numbers of the sources used to answer this question. Do not include a page number if the context is irrelevant.",
    )

