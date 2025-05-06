
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
from evaluate.utils.cache import check_cache, load_cache, save_to_cache
from evaluate.utils.display import display_results_embedding, plot_results_embedding
from config.eval_setting import EvalEmbeddingSettings
from config.logging_setting import setup_logger

params = EvalEmbeddingSettings()
logger = setup_logger(f"logs/evaluate/{params.embed_name}/{'finetune' if params.finetune else 'baseline'}")

nest_asyncio.apply()

async def evaluate(dataset_path: str, 
                    embed_model: str = None, 
                    top_k: int = 1, 
                    finetune:bool = False, 
                    path_finetuned: str = None) -> list:
    """Evaluate the retriever using the given dataset and embedding model."""

    dataset = EmbeddingQAFinetuneDataset.from_json(dataset_path)
    corpus = dataset.corpus
    embed_model = embed_model or Settings.embed_model
    nodes = [TextNode(id_=id_, text=text) for id_, text in corpus.items()]

    if finetune:
        base_embed_model = resolve_embed_model(embed_model)
        embed_model = AdapterEmbeddingModel(
                                        base_embed_model,
                                        adapter_path=path_finetuned
                                        )
    index = VectorStoreIndex(nodes, 
                            embed_model=embed_model, 
                            show_progress=True)

    retriever = index.as_retriever(similarity_top_k=top_k)
    metrics = ["hit_rate", "mrr", "ap", "ndcg"]
    retriever_evaluator = RetrieverEvaluator.from_metric_names(metrics, retriever=retriever)
    eval_results = await retriever_evaluator.aevaluate_dataset(dataset)
    return eval_results



def run_evaluator_top_k(name:str,
                        dataset_path_list:list,
                        top_k:int, 
                        finetune:bool = False, 
                        path_finetuned: str = None,
                        path_to_cache:str = None) -> pd.DataFrame:
    """Run evaluator for all dataset types."""
    all_results = []
    model_name = f"finetune_{name}" if finetune else name
    embed_model = f"local:{name}"
    for dataset_path in dataset_path_list:
        if "train" in dataset_path:
            dataset_type = "train"
        elif "val" in dataset_path:
            dataset_type = "val"
        else:
            raise ValueError(f"Invalid dataset path: {dataset_path}")
        logger.info(f"Evaluating with top_k={top_k} on {dataset_type} dataset...")

        cache_path_filename = check_cache(path_to_cache, name, dataset_type, top_k, finetune)

        if os.path.exists(cache_path_filename):
            logger.info(f"Loading cached results from: {cache_path_filename}")
            eval_results = load_cache(cache_path_filename)
        else:
            logger.info(f"Evaluating and saving results to cache: {cache_path_filename}")
            eval_results = asyncio.run(evaluate(dataset_path=dataset_path,
                                                embed_model=embed_model, 
                                                top_k=top_k, 
                                                finetune=finetune, 
                                                path_finetuned=path_finetuned))
            save_to_cache(cache_path_filename, eval_results)
        
        # Display results
        result_df = display_results_embedding(
                                    name=model_name,
                                    data=dataset_type,
                                    top=f"top-{top_k} eval",
                                    eval_results=eval_results
                                    )
        all_results.append(result_df)
    df_results = pd.concat(all_results, axis=0, ignore_index=True)
    logger.info(df_results)
    return df_results


if __name__ == "__main__":

    logger.info("Starting evaluation...")
    # Run evaluator for all dataset types
    df_results = run_evaluator_top_k(name=params.embed_name, 
                                    dataset_path_list=params.dataset_path_list,
                                    path_to_cache=params.cache_path,
                                    top_k=params.top_k, 
                                    finetune=params.finetune, 
                                    path_finetuned=params.path_finetuned)
    # Display results
    plot_results_embedding(df_results, finetune=params.finetune, params=params)
    logger.info("Evaluation completed.")
    