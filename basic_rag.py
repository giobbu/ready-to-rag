from loguru import logger
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.finetuning import generate_qa_embedding_pairs

from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core import SummaryIndex
from llama_index.core.tools import FunctionTool

from llama_index.core.evaluation import EmbeddingQAFinetuneDataset
from llama_index.finetuning import EmbeddingAdapterFinetuneEngine
from llama_index.core.embeddings.utils import resolve_embed_model
from llama_index.embeddings.adapter import AdapterEmbeddingModel
from llama_index.llms.openai import OpenAI

from models.output import BasicOutput, MetaVectorOutput
import os
import json

from dotenv import load_dotenv
load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

from config.logging_setting import setup_logger
logger = setup_logger(path_to_save="logs/operation")

class RAGgish:
    " RAG model for question answering "
    def __init__(self, embed_name: str, llm_name: str, temperature: float = 0.0):
        """ Initialize the RAG model """
        self.embed_name = embed_name
        self.llm_name = llm_name
        self.temperature = temperature

    def _set_embed_model(self):
        """ Set Embedding model """
        try:
            self.embed_model = HuggingFaceEmbedding(model_name=self.embed_name)
        except:
            raise ValueError(f"Error loading embedding model: {self.embed_name}")
        logger.debug(f"Loaded embedding model: {self.embed_name}")

    def _set_llm_model(self):
        """ Set LLM model """
        try: 
            llm = OpenAI(temperature=self.temperature, model=self.llm_name)
        except:
            raise ValueError(f"Error loading LLM model: {self.llm_name}")
        logger.debug(f"Loaded LLM model: {self.llm_name}")
        return llm

    def _set_QA_llm(self):
        """ Set LLM model for QA """
        try:
            llm = OpenAI(temperature=self.temperature, model=self.llm_name)
        except:
            raise ValueError(f"Error loading LLM model: {self.llm_name}")
        logger.debug(f"Loaded LLM model for QA: {self.llm_name}")
        return llm

    def load_data(self, input_dir: str, required_exts: list):
        " Load data from input_dir "
        logger.debug(f"Load from PATH: {input_dir}")
        documents = SimpleDirectoryReader(input_dir=input_dir, 
                                        required_exts=required_exts).load_data()
        logger.debug(f"Loaded {len(documents)} documents")
        return documents
    
    def parse_documents(self, documents: list, chunk_size: int):
        " Parse documents into nodes"
        parser = SentenceSplitter(chunk_size=chunk_size)
        nodes = parser.get_nodes_from_documents(documents, show_progress=True)
        return nodes
    
    def parse_files_to_QA_data(self, 
                                input_dir_train: str,
                                input_dir_val: str,
                                required_exts: list, 
                                prompt_tmpl: str,
                                chunk_size: int,
                                save_train_path: str, 
                                save_val_path: str):
        """ Parse and save QA data for finetuning """
        logger.debug("Setting Prompt template")
        prompts={}
        prompts["EN"] = prompt_tmpl
        logger.debufg(f"-- QA data with chunk size: {chunk_size}")
        logger.debug(f"-- training")
        documents_train = self.load_data(input_dir_train, required_exts)
        nodes_train = self.parse_documents(documents_train, chunk_size)
        generate_qa_embedding_pairs(llm=self._set_QA_llm(),
                                        nodes=nodes_train,
                                        qa_generate_prompt_tmpl = prompts["EN"],
                                        num_questions_per_chunk=1,
                                        output_path=save_train_path
                                                            )
        logger.debug("-- validation")
        documents_val = self.load_data(input_dir_val, required_exts)
        nodes_val = self.parse_documents(documents_val, chunk_size)
        generate_qa_embedding_pairs(llm=self._set_QA_llm(),
                                        nodes=nodes_val,
                                        qa_generate_prompt_tmpl = prompts["EN"],
                                        num_questions_per_chunk=1,
                                        output_path=save_val_path
                                        )
    
    def load_QA_data(self, train_data_path: str, val_data_path: str):
        " Load finetune data "
        logger.debug("Loading QA data for finetuning")
        train_dataset = EmbeddingQAFinetuneDataset.from_json(train_data_path)
        val_dataset = EmbeddingQAFinetuneDataset.from_json(val_data_path)
        return train_dataset, val_dataset
        
    def get_sentence_transformer_finetune(self, 
                                          train_dataset: EmbeddingQAFinetuneDataset, 
                                          model_output_path: str, 
                                          bias: bool = True, 
                                          epochs: int = 10):
        """ Get the finetune engine """
        " Get the finetune engine "
        logger.debug('Loading base embedding model and setting up finetuning engine')
        base_embed_model = resolve_embed_model(f"local:{self.embed_name}")
        finetuned_engine = EmbeddingAdapterFinetuneEngine(
                                                        train_dataset,
                                                        base_embed_model,
                                                        model_output_path = model_output_path,
                                                        bias=bias,
                                                        epochs=epochs,
                                                        verbose=True,
                                                    )
        return finetuned_engine
    
    def finetune_embeddings(self, 
                            input_dir_train: str,
                            input_dir_val: str,
                            prompt_tmpl: str,
                            chunk_size: int,
                            sent_transf_params: dict,
                            save_train_path: str, 
                            save_val_path: str,
                            save_model_path: str,
                            parse_files: bool = True):
        """ Finetune the embeddings """
        if parse_files:
            self.parse_files_to_QA_data(
                                        input_dir_train=input_dir_train,
                                        input_dir_val=input_dir_val,
                                        required_exts=['.pdf'],
                                        prompt_tmpl=prompt_tmpl,
                                        chunk_size=chunk_size,
                                        save_train_path=save_train_path,
                                        save_val_path=save_val_path
                                    )
        train_dataset, _ = self.load_QA_data(train_data_path=save_train_path,
                                            val_data_path=save_val_path
                                            )
        logger.debug(f"--- Finetuning the embedding model with params {sent_transf_params}")
        finetune_engine = self.get_sentence_transformer_finetune(
            train_dataset=train_dataset,
            model_output_path=save_model_path,
            bias=sent_transf_params["bias"],
            epochs=sent_transf_params["epochs"]
        )
        finetune_engine.finetune()
        embed_model_finetuned = finetune_engine.get_finetuned_model()
        return embed_model_finetuned
    
    def load_finetuned_model(self, embed_name: str, model_output_path: str):
            """ Load the finetuned model """
            base_embed_model = resolve_embed_model(f"local:{embed_name}")
            embed_model_finetuned = AdapterEmbeddingModel(base_embed_model,
                                                            adapter_path=model_output_path,
                                                            )
            return embed_model_finetuned

    def create_or_load_vector_idx(self, 
                                  nodes: list, 
                                  vec_store_path: str,
                                  vec_store_idx: str,
                                  embed_model: str = None):
        " Create a vector store index "
        if os.path.exists(vec_store_path):
            logger.debug("* Loading VectorStore Index")
            storage_context = StorageContext.from_defaults(persist_dir=vec_store_path)
            index = load_index_from_storage(storage_context, index_id=vec_store_idx)
            return index
        
        logger.debug("* Creating VectorStore Index")
        index = VectorStoreIndex(
                                nodes=nodes, 
                                embed_model=self._set_embed_model() if embed_model is None else embed_model,
                                insert_batch_size=1000,
                                show_progress=True
                                )
        index.set_index_id(vec_store_idx)
        index.storage_context.persist(persist_dir=vec_store_path)   
        return index
    
    def create_or_load_summary_idx(self, 
                                   nodes: list,
                                   summary_path: str, 
                                   summary_idx: str,
                                   embed_model: str = None):
        " Create a summary index "
        if os.path.exists(summary_path):
            logger.debug("* Loading Summary Index")
            storage_context = StorageContext.from_defaults(persist_dir=summary_path)
            index = load_index_from_storage(storage_context, index_id="summary_index")
            return index
        
        logger.debug("* Creating Summary Index")
        index = SummaryIndex(nodes=nodes,
                            embed_model=self._set_embed_model() if embed_model is None else embed_model,
                            show_progress=True)
        index.set_index_id(summary_idx)
        index.storage_context.persist(persist_dir=summary_path)
        return index

    def create_base_query_tool(self, index: VectorStoreIndex):
        " Create a base query tool "

        def vector_query(query: str) -> str:
            " Query the index "
            llm = self._set_llm_model()
            structured_llm = llm.as_structured_llm(output_cls=BasicOutput)
            query_engine = index.as_query_engine(llm=structured_llm)
            response = query_engine.query(query)
            return response
        
        vector_query_tool = FunctionTool.from_defaults(
                                    name="vector_tool",
                                    fn=vector_query,
                                    description=(
                                        "Useful if you want to get "
                                        "basic answers to your queries. "
                                    ))
        logger.debug("- Created Vector QueryEngine Tool")
        schema = vector_query_tool.metadata.get_parameters_dict()
        print("Vector Tool Schema:")
        print(schema)
        return vector_query_tool

    def create_metadata_query_tool(self, index: VectorStoreIndex):
        " Create a metadata query engine "
        from llama_index.core.vector_stores import MetadataFilters
        from llama_index.core.vector_stores import FilterCondition
        from llama_index.core.tools import FunctionTool
        from typing import List

        def metadata_vector_query(query: str, page_numbers: List[str]) -> str:
            " Query the index with metadata filters "
            llm = self._set_llm_model()
            structured_llm = llm.as_structured_llm(output_cls=MetaVectorOutput)
            meta_data = [{"key": "page_label", "value": p} for p in page_numbers]
            query_engine = index.as_query_engine(
                llm=structured_llm,
                filters= MetadataFilters.from_dicts(
                meta_data,
                condition=FilterCondition.OR)
            )
            response = query_engine.query(query)
            return response
        
        metadata_vector_query_tool = FunctionTool.from_defaults(
                                                            name="metadata_vector_tool",
                                                            fn=metadata_vector_query,
                                                            description=(
                                                            "Useful if you want to get metadata based on page numbers."
                                                            ))
        logger.debug("- Created Metadata QueryEngine Tool")
        schema = metadata_vector_query_tool.metadata.get_parameters_dict()
        print("Metadata Tool Schema:")
        print(schema)
        return metadata_vector_query_tool

    def create_summary_query_tool(self, summary_index: SummaryIndex):
        " Create a summary query engine "
        from llama_index.core.tools import QueryEngineTool
        llm = self._set_llm_model()
        structured_llm = llm.as_structured_llm(output_cls=BasicOutput)
        summary_query_engine = summary_index.as_query_engine(
            llm=structured_llm,
            response_mode="tree_summarize",
            use_async=True,
        )
        summary_tool = QueryEngineTool.from_defaults(
                                                    name="summary_tool",
                                                    query_engine=summary_query_engine,
                                                    description=(
                                                    "Useful if you want to get a summary of the scientific paper."
                                                    ),
                                                )
        logger.debug("- Created Summary QueryEngine Tool")
        schema = summary_tool.metadata.get_parameters_dict()
        print("Summary Tool Schema:")
        print(schema)
        return summary_tool

    def answer(self, query: str, list_tools: list):
        """ Answer the query by employing differnt tools """
        llm = self._set_llm_model()
        if list_tools is None:
            logger.error("No tools provided. Please check the config file.")
            raise ValueError("No tools provided. Please check the config file.")
        elif len(list_tools) > 3:
            logger.error("Too many tools provided. Please check the config file.")
            raise ValueError("Too many tools provided. Please check the config file.")
        
        print("--------------------------------------")
        print(f"Query: {query}")
        print("--------------------------------------")
        response = llm.predict_and_call(list_tools,  
                                        query,
                                        verbose=True)
        response_dict = json.loads(response.response)
        self.display_response(query, response, response_dict)
        return response_dict
    
    def display_response(self, query: str, response: str, response_dict: dict):
        """ Display the response """
        logger.info("__________________________________________________________")
        logger.info(f'Query: {query}')
        logger.info(f"Structured Output: {response}")
        logger.info("__________________________________________________________")
        logger.info(f'Response: {response_dict["response"]}')
        if response_dict['confidence'] > 0.9:
            logger.success(f"Confidence Score: {response_dict['confidence']}")
        elif response_dict['confidence'] > 0.8:
            logger.warning(f"Confidence Score: {response_dict['confidence']}")
            logger.warning(f"Confidence Explanation: {response_dict['confidence_explanation']}")
        else:
            logger.error(f"Confidence Score: {response_dict['confidence']}")
            logger.error(f"Confidence Explanation: {response_dict['confidence_explanation']}")
            logger.error("Please check the query and try again.")
        logger.info("__________________________________________________________")


        



