from llama_index.core import Settings
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core import ChatPromptTemplate
from loguru import logger
from config import EmbeddingSettings, LLMSettings

class RAGgish:
    " RAG model for question answering "
    def __init__(self, embed_name, llm_name, context_window, max_new_tokens, generate_kwargs):
        self.context_window = context_window
        self.max_new_tokens = max_new_tokens
        self.generate_kwargs = generate_kwargs
        self._set_embed_model(embed_name)
        self._set_llm_model(llm_name)

    def _set_embed_model(self, embed_name):
        """ Set the embedding model """
        Settings.embed_model = HuggingFaceEmbedding(model_name=embed_name)
        logger.debug(f"Loaded embedding model: {Settings.embed_model.model_name}")

    def _set_llm_model(self, llm_name):
        """ Set the LLM model """
        if isinstance(llm_name, tuple):
            llm_name = llm_name[0]
            self.context_window = self.context_window[0]
            self.max_new_tokens = self.max_new_tokens[0]
        Settings.llm = HuggingFaceLLM(
            model_name=llm_name,
            tokenizer_name=llm_name,
            context_window=self.context_window,
            max_new_tokens=self.max_new_tokens,
            generate_kwargs=self.generate_kwargs
        )
        logger.debug(f"Loaded LLM model: {Settings.llm.model_name}")

    def load_data(self, input_dir, required_exts):
        " Load data from input_dir "
        documents = SimpleDirectoryReader(input_dir=input_dir, required_exts=required_exts).load_data()
        logger.debug(f"Read {len(documents)} documents from {input_dir}")
        return documents
    
    def create_index(self, documents):
        " Create a vector store index "
        index = VectorStoreIndex.from_documents(documents)
        logger.debug("Created VectorStoreIndex")
        return index
    
    def create_query_engine(self, index):
        " Create a query engine "
        chat_text_qa_msgs = [
            (
                "user",
                """You are a Q&A assistant. Your goal is to answer questions as
        accurately as possible based on the instructions and context provided.

        Context:

        {context_str}

        Question:

        {query_str}
        """
            )
        ]
        text_qa_template = ChatPromptTemplate.from_messages(chat_text_qa_msgs)
        self.query_engine = index.as_query_engine(text_qa_template=text_qa_template)
        logger.debug("Created QueryEngine")

    def answer(self, query):
        " Query the QA engine "
        response = self.query_engine.query(query)
        logger.info("__________________________________________________________")
        logger.info("\n")
        logger.info(f'Query: \n >>> {query}')
        logger.info(f"Response: \n >>> {response}")
        logger.info("__________________________________________________________")
        logger.info("\n")
        return response
