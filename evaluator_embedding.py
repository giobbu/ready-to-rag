from loguru import logger
import pandas as pd
import os
import nest_asyncio
import asyncio

from llama_index.core.schema import TextNode
from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.evaluation import EmbeddingQAFinetuneDataset, RetrieverEvaluator
from llama_index.embeddings.adapter import AdapterEmbeddingModel
from llama_index.core.embeddings.utils import resolve_embed_model

from evaluate.utils.cache_embedding import check_cache, load_cache, save_to_cache
from evaluate.utils.display_embedding import create_df_results_embedding, plot_results_embedding
from config.eval_setting import EvalEmbeddingSettings
from config.logging_setting import setup_logger


class EmbeddingEvaluator:
    def __init__(self, params: EvalEmbeddingSettings):
        self.params = params
        self.logger = setup_logger(
            f"logs/evaluate/{params.embed_name}/{'finetune' if params.finetune else 'baseline'}"
        )
        nest_asyncio.apply()
        self.embed_model = f"local:{self.params.embed_name}"
        self.model_name = f"finetune_{self.params.embed_name}" if self.params.finetune else self.params.embed_name

    async def _evaluate_async(self, dataset_path: str, top_k: int, finetune: bool, path_finetuned: str = None):
        """Evaluate the retriever using the given dataset and embedding model."""
        dataset = EmbeddingQAFinetuneDataset.from_json(dataset_path)
        corpus = dataset.corpus
        embed_model = self.embed_model or Settings.embed_model
        nodes = [TextNode(id_=id_, text=text) for id_, text in corpus.items()]

        if finetune:
            base_embed_model = resolve_embed_model(embed_model)
            embed_model = AdapterEmbeddingModel(base_embed_model, adapter_path=path_finetuned)

        index = VectorStoreIndex(nodes, embed_model=embed_model, show_progress=True)
        retriever = index.as_retriever(similarity_top_k=top_k)

        # Define the metrics to evaluate
        metrics = ["hit_rate", "mrr", "ap", "ndcg"]
        evaluator = RetrieverEvaluator.from_metric_names(metrics, retriever=retriever)
        return await evaluator.aevaluate_dataset(dataset)

    def evaluate_dataset_list(self) -> pd.DataFrame:
        " Evaluate all datasets in the dataset path list."
        all_results = []
        for dataset_path in self.params.dataset_path_list:
            dataset_type = self._get_dataset_type(dataset_path)
            self.logger.info(f"Evaluating with top_k={self.params.top_k} on {dataset_type} dataset...")
            # Check if the cache exists
            cache_path_filename = check_cache(
                self.params.cache_path, self.params.embed_name, dataset_type, self.params.top_k, self.params.finetune
            )

            if os.path.exists(cache_path_filename):
                self.logger.info(f"Loading cached results from: {cache_path_filename}")
                eval_results = load_cache(cache_path_filename)
            else:
                self.logger.info(f"Evaluating and saving results to cache: {cache_path_filename}")
                eval_results = asyncio.run(
                    self._evaluate_async(
                        dataset_path=dataset_path,
                        top_k=self.params.top_k,
                        finetune=self.params.finetune,
                        path_finetuned=self.params.path_finetuned,
                    )
                )
                save_to_cache(cache_path_filename, eval_results)

            result_df = create_df_results_embedding(name=self.model_name,
                                                    data=dataset_type,
                                                    top=f"top-{self.params.top_k} eval",
                                                    eval_results=eval_results,
                                                    )
            all_results.append(result_df)

        # Combine results from all datasets into a single DataFrame
        df_results_datasets = pd.concat(all_results, axis=0, ignore_index=True)
        self.logger.info(df_results_datasets)
        return df_results_datasets

    def _get_dataset_type(self, dataset_path: str) -> str:
        " Get the dataset type from the dataset path."
        if "train" in dataset_path:
            return "train"
        elif "val" in dataset_path:
            return "val"
        else:
            raise ValueError(f"Invalid dataset path: {dataset_path}")

    def run(self):
        " Run the evaluator."
        self.logger.info("Starting evaluation...")
        df_results = self.evaluate_dataset_list()
        return df_results

    def display_results(self, df_results: pd.DataFrame):
        """Display the results."""
        plot_results_embedding(df_results, finetune=self.params.finetune, params=self.params)
        self.logger.info("Evaluation completed.")


if __name__ == "__main__":

    logger.info("Starting evaluation...")
    params = EvalEmbeddingSettings()
    evaluator = EmbeddingEvaluator(params)
    df_results = evaluator.run()
    evaluator.display_results(df_results)
    logger.info("Evaluation completed.")
