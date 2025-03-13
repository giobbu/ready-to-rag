from llama_index.core import Settings
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
# from llama_index.finetuning import SentenceTransformersFinetuneEngine
from loguru import logger

class RAGgish:
    " RAG model for question answering "
    def __init__(self, embed_name, llm_name, context_window, max_new_tokens, generate_kwargs, chunk_size=1000):
        self.context_window = context_window
        self.max_new_tokens = max_new_tokens
        self.generate_kwargs = generate_kwargs
        self._set_embed_model(embed_name)
        self._set_llm_model(llm_name)
        self.chunk_size = chunk_size

    def _set_embed_model(self, embed_name):
        """ Set the embedding model """
        self.embed_model = HuggingFaceEmbedding(model_name=embed_name)
        logger.debug(f"Loaded embedding model: {embed_name}")

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
        logger.debug(f"Load from PATH: {input_dir}")
        documents = SimpleDirectoryReader(input_dir=input_dir, required_exts=required_exts).load_data()
        logger.debug(f"Loaded {len(documents)} documents")
        return documents
    
    def parse_documents(self, documents):
        " Parse documents into nodes"
        logger.debug("Parsing documents into nodes...")
        parser = SentenceSplitter(chunk_size=self.chunk_size)
        nodes = parser.get_nodes_from_documents(documents, show_progress=True)
        return nodes

    def create_index(self, nodes):
        " Create a vector store index "
        logger.debug("Creating VectorStoreIndex...")
        index = VectorStoreIndex(
        nodes=nodes, 
        embed_model=self.embed_model, 
        insert_batch_size=1000,
        show_progress=True
    )
        logger.debug("...VectorStoreIndex created")
        return index
    
    def create_query_engine(self, index):
        " Create a query engine "
        self.query_engine = index.as_query_engine()
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
