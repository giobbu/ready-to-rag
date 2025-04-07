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
    def __init__(self, embed_name, llm_name):
        self.embed_name = embed_name
        self.llm_name = llm_name
        self._set_embed_model
        self._set_llm_model

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
    
    def parse_documents(self, documents, chunk_size):
        " Parse documents into nodes"
        logger.debug("Parsing documents into nodes...")
        parser = SentenceSplitter(chunk_size=chunk_size)
        nodes = parser.get_nodes_from_documents(documents, show_progress=True)
        return nodes
    
    def parse_files_to_QA_data(self, 
                                input_dir_train, 
                                input_dir_val, 
                                required_exts, 
                                prompt_tmpl,
                                chunk_size,
                                save_train_path, 
                                save_val_path):
        """ Parse and save QA data for finetuning """
        logger.debug("- Load QA data for training")
        documents_train = self.load_data(input_dir_train, required_exts)
        nodes_train = self.parse_documents(documents_train, chunk_size)
        logger.debug("- Load QA data for validation")
        documents_val = self.load_data(input_dir_val, required_exts)
        nodes_val = self.parse_documents(documents_val, chunk_size)
        logger.debug("- Parse QA pairs for training")
        prompts={}
        prompts["EN"] = prompt_tmpl
        generate_qa_embedding_pairs(llm=OpenAI(temperature=0.0, 
                                                                model=self.llm_name),
                                                                nodes=nodes_train,
                                                                qa_generate_prompt_tmpl = prompts["EN"],
                                                                num_questions_per_chunk=1,
                                                                output_path=save_train_path
                                                            )
        logger.debug("- Parse QA pairs for validation")
        generate_qa_embedding_pairs(llm=OpenAI(temperature=0.0, 
                                            model=self.llm_name),
                                            nodes=nodes_val,
                                            qa_generate_prompt_tmpl = prompts["EN"],
                                            num_questions_per_chunk=1,
                                            output_path=save_val_path
                                        )
    
    def load_QA_data(self, train_data_path, val_data_path):
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
                            chunk_size,
                            save_train_path, 
                            save_val_path,
                            save_model_path,
                            parse_files=False):
        """ Finetune the embeddings """
        logger.debug("Finetuning embeddings...")

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
        train_dataset, _ = self.load_QA_data(
            train_data_path=save_train_path,
            val_data_path=save_val_path
        )
        finetune_engine = self.get_sentence_transformer_finetune(
            train_dataset=train_dataset,
            model_output_path=save_model_path[0]
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
                                    adapter_path=model_output_path,
                                )
            return embed_model_finetuned

    def create_vector_index(self, nodes, embed_model):
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
    
    def create_summary_index(self, nodes, embed_model):
        " Create a summary index "
        from llama_index.core import SummaryIndex
        logger.debug("Creating SummaryIndex...")
        index = SummaryIndex(nodes=nodes,
                            embed_model=embed_model,
                            show_progress=True)
        logger.debug("...SummaryIndex created")
        return index

    def create_base_query_tool(self, index):
        " Create a base query tool "
        from llama_index.core.tools import FunctionTool

        def vector_query(query: str) -> str:
            " Query the index "
            query_engine = index.as_query_engine()
            response = query_engine.query(query)
            return response
        
        vector_query_tool = FunctionTool.from_defaults(
                                    name="vector_tool",
                                    fn=vector_query,
                                    description=(
                                        "Useful if you want to get "
                                        "basic answers to your queries. "
                                    ))
        logger.debug("Created Vector QueryEngine Tool")
        return vector_query_tool

    def create_metadata_query_tool(self, index):
        " Create a metadata query engine "
        from llama_index.core.vector_stores import MetadataFilters
        from llama_index.core.vector_stores import FilterCondition
        from llama_index.core.tools import FunctionTool
        from typing import List

        def metadata_vector_query(query: str, page_numbers: List[str]) -> str:
            " Query the index with metadata filters "
            meta_data = [{"key": "page_label", "value": p} for p in page_numbers]
            query_engine = index.as_query_engine(
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
                                        "Useful if you want to get metadata "
                                        "based on page numbers. "
                                    ))
        logger.debug("Created Metadata QueryEngine Tool")
        return metadata_vector_query_tool

    def create_summary_query_tool(self, summary_index):
        " Create a summary query engine "
        from llama_index.core.tools import QueryEngineTool
        summary_query_engine = summary_index.as_query_engine(
            response_mode="tree_summarize",
            use_async=True,
        )
        summary_tool = QueryEngineTool.from_defaults(
            name="summary_tool",
            query_engine=summary_query_engine,
            description=(
                "Useful if you want to get a summary of the scientific paper. "
            ),
        )
        logger.debug("Created Summary QueryEngine Tool")
        return summary_tool

    def answer(self, query, list_tools=None):
        """ Answer the query by employing differnt tools """
        llm = OpenAI(temperature=0.0, model=self.llm_name)
        if len(list_tools) == 0:
            raise ValueError("No tools provided. Please check the config file.")
        elif len(list_tools) > 3:
            raise ValueError("Too many tools provided. Please check the config file.")
        
        response = llm.predict_and_call(list_tools,  
                                        query,
                                        verbose=True)

        from llama_index.core.evaluation import FaithfulnessEvaluator
        evaluator = FaithfulnessEvaluator(llm=llm)
        eval_result = evaluator.evaluate_response(response=response)

        logger.info("__________________________________________________________")
        logger.info("\n")
        logger.info(f'Query: \n >>> {query}')
        logger.info(f"Response: \n >>> {response}")
        logger.info("__________________________________________________________")
        logger.info("\n")
        logger.info(f"Evaluation score >>> {eval_result.score}")
        return response
        



