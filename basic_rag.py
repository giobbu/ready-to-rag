from loguru import logger
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.finetuning import generate_qa_embedding_pairs

from llama_index.core.evaluation import EmbeddingQAFinetuneDataset
from llama_index.finetuning import EmbeddingAdapterFinetuneEngine
from llama_index.core.embeddings.utils import resolve_embed_model
from llama_index.embeddings.adapter import AdapterEmbeddingModel

from llama_index.llms.openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
import os
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

class RAGgish:
    " RAG model for question answering "
    def __init__(self, embed_name, llm_name, chunk_size=1000):
        self.embed_name = embed_name
        self.llm_name = llm_name
        self._set_embed_model
        self._set_llm_model
        self.chunk_size = chunk_size

    @property
    def _set_embed_model(self):
        """ Set the embedding model """
        try:
            self.embed_model = HuggingFaceEmbedding(model_name=self.embed_name)
        except:
            raise ValueError(f"Error loading embedding model: {self.embed_name}")
        logger.debug(f"Loaded embedding model: {self.embed_name}")

    @property
    def _set_llm_model(self):
        """ Set the LLM model """
        try: 
            Settings.llm = OpenAI(temperature=0.0, model=self.llm_name)
        except:
            raise ValueError(f"Error loading LLM model: {self.llm_name}")
        logger.debug(f"Loaded LLM model: {self.llm_name}")

    def load_data(self, input_dir, required_exts):
        " Load data from input_dir "
        logger.debug(f"Load from PATH: {input_dir}")
        documents = SimpleDirectoryReader(input_dir=input_dir, 
                                        required_exts=required_exts).load_data()
        logger.debug(f"Loaded {len(documents)} documents")
        return documents
    
    def parse_documents(self, documents):
        " Parse documents into nodes"
        logger.debug("Parsing documents into nodes...")
        parser = SentenceSplitter(chunk_size=self.chunk_size)
        nodes = parser.get_nodes_from_documents(documents, show_progress=True)
        return nodes
    
    def parse_save_data_finetune(self, 
                                input_dir_train, 
                                input_dir_val, 
                                required_exts, 
                                prompt_tmpl,
                                save_train_path, 
                                save_val_path):
        """ Parse and save data for finetuning """
        logger.debug("- Load and parse data for training")
        documents_train = self.load_data(input_dir_train, required_exts)
        nodes_train = self.parse_documents(documents_train)
        logger.debug("- Load and parse data for validation")
        documents_val = self.load_data(input_dir_val, required_exts)
        nodes_val = self.parse_documents(documents_val)
        logger.debug("- Generate QA pairs - training")
        prompts={}
        prompts["EN"] = prompt_tmpl
        generate_qa_embedding_pairs(llm=OpenAI(temperature=0.0, 
                                                                model="gpt-3.5-turbo"),
                                                                nodes=nodes_train,
                                                                qa_generate_prompt_tmpl = prompts["EN"],
                                                                num_questions_per_chunk=1,
                                                                output_path=save_train_path
                                                            )
        logger.debug("- Generate QA pairs - validation")
        generate_qa_embedding_pairs(llm=OpenAI(temperature=0.0, 
                                                                model="gpt-3.5-turbo"),
                                                                nodes=nodes_val,
                                                                qa_generate_prompt_tmpl = prompts["EN"],
                                                                num_questions_per_chunk=1,
                                                                output_path=save_val_path
                                                            )
    
    def load_finetune_data(self, train_data_path, val_data_path):
        " Load finetune data "
        logger.debug("Loading data for finetuning...")
        train_dataset = EmbeddingQAFinetuneDataset.from_json(train_data_path)
        val_dataset = EmbeddingQAFinetuneDataset.from_json(val_data_path)
        logger.debug("...data loaded")
        return train_dataset, val_dataset
        
    def get_sentence_transformer_finetune(self, train_dataset, model_output_path):
        " Get the finetune engine "
        logger.debug('...Loading base embedding model and setting up finetuning engine')
        base_embed_model = resolve_embed_model(f"local:{self.embed_name}")
        finetuned_engine = EmbeddingAdapterFinetuneEngine(
            train_dataset,
            base_embed_model,
            model_output_path = model_output_path,
            bias=True,
            epochs=10,
            verbose=True,
        )
        logger.debug('...Engine Fine-tuned')
        return finetuned_engine
    
    def finetune_embeddings(self, 
                            input_dir_train, 
                            input_dir_val,
                            prompt_tmpl,
                            save_train_path, 
                            save_val_path):
        """ Finetune the embeddings """
        logger.debug("Finetuning embeddings...")
        self.parse_save_data_finetune(
            input_dir_train=input_dir_train,
            input_dir_val=input_dir_val,
            required_exts=['.pdf'],
            prompt_tmpl=prompt_tmpl,
            save_train_path=save_train_path,
            save_val_path=save_val_path
        )
        train_dataset, _ = self.load_finetune_data(
            train_data_path=save_train_path,
            val_data_path=self.out_dir_val
        )
        finetune_engine = self.get_sentence_transformer_finetune(
            train_dataset=train_dataset,
            model_output_path=save_val_path
        )
        finetune_engine.finetune()
        embed_model_finetuned = finetune_engine.get_finetuned_model()
        logger.debug("...Finetuning complete and finetuned model loaded")
        return embed_model_finetuned
    
    def load_finetuned_model(self, embed_name: str, model_output_path: str):
            """ Load the finetuned model """
            base_embed_model = resolve_embed_model(f"local:{embed_name}")
            embed_model_finetuned = AdapterEmbeddingModel(
                                    base_embed_model,
                                    adapter_path=model_output_path[0],
                                )
            return embed_model_finetuned

    def create_index(self, nodes, embed_model):
        " Create a vector store index "
        logger.debug("Creating VectorStoreIndex...")
        index = VectorStoreIndex(
        nodes=nodes, 
        embed_model=embed_model, 
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
