from pydantic import BaseModel, Field

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