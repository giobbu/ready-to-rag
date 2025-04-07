
from llama_index.core.schema import TextNode
from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.evaluation import EmbeddingQAFinetuneDataset, RetrieverEvaluator
from llama_index.embeddings.adapter import AdapterEmbeddingModel
from llama_index.core.embeddings.utils import resolve_embed_model
from loguru import logger
import nest_asyncio
import asyncio
import pandas as pd

from config import EvalSettings
params = EvalSettings()

nest_asyncio.apply()
async def evaluate(dataset_path: str, embed_model: str = None, top_k: int = 1, finetune:bool = False, path_finetuned: str = None) -> list:
    """Evaluate the retriever using the given dataset and embedding model."""
    dataset = EmbeddingQAFinetuneDataset.from_json(dataset_path)
    corpus = dataset.corpus
    embed_model = embed_model or Settings.embed_model
    nodes = [TextNode(id_=id_, text=text) for id_, text in corpus.items()]
    if finetune:
        base_embed_model = resolve_embed_model(embed_model)
        embed_model_finetuned = AdapterEmbeddingModel(
                                        base_embed_model,
                                        adapter_path=path_finetuned,
                                    )
        index = VectorStoreIndex(nodes, 
                                embed_model=embed_model_finetuned, 
                                show_progress=True)
    else:
        index = VectorStoreIndex(nodes, 
                                embed_model=embed_model, 
                                show_progress=True)
    retriever = index.as_retriever(similarity_top_k=top_k)
    metrics = ["hit_rate", "mrr", "precision", "recall", "ap", "ndcg"]
    retriever_evaluator = RetrieverEvaluator.from_metric_names(metrics, retriever=retriever)
    eval_results = await retriever_evaluator.aevaluate_dataset(dataset)
    return eval_results

def display_results(name:str, data:str, top: str, eval_results: list) -> pd.DataFrame:
    """Display results from evaluate."""
    metric_dicts = [eval_result.metric_vals_dict for eval_result in eval_results]
    full_df = pd.DataFrame(metric_dicts)
    metrics = ["hit_rate", "mrr", "precision", "recall", "ap", "ndcg"]
    columns = {"embed": [name],
                "data": [data],
                "top": [top], 
                **{k: [full_df[k].mean()] for k in metrics}}
    metric_df = pd.DataFrame(columns)
    return metric_df

def run_evaluator(name:str,
                dataset_type_list:list,
                top_k:int, 
                finetune:bool = False, 
                path_finetuned: str = None) -> pd.DataFrame:
    df_results = pd.DataFrame([])
    for dataset_type in dataset_type_list:
        dataset_path = f"save/gpt/{dataset_type}_dataset_gpt.json"
        bge = f"local:{name}"
        logger.info(f"Evaluating with top_k={top_k} on {dataset_type} dataset...")
        eval_results = asyncio.run(evaluate(dataset_path, bge, top_k=top_k, finetune = finetune, path_finetuned = path_finetuned))
        if df_results.empty:
            df_results = display_results(name=f"finetune_{name}" if finetune else name,
                                        data = dataset_type,
                                        top=f"top-{top_k} eval", 
                                        eval_results=eval_results)
        else:
            df_results = pd.concat([df_results, 
                                    display_results(name=f"finetune_{name}" if finetune else name,
                                                    data = dataset_type,
                                                    top=f"top-{top_k} eval", 
                                                    eval_results=eval_results)], 
                                                    axis=0)
    logger.info(df_results)
    return df_results

if __name__ == "__main__":
    df_results = run_evaluator(name = params.embed_name, 
                                dataset_type_list= params.dataset_type_list, 
                                top_k= params.top_k, 
                                finetune=params.finetune, 
                                path_finetuned=params.path_finetuned)

    # plot stacked bar chart for train and val and for each metric in one plot
    import matplotlib.pyplot as plt
    import seaborn as sns
    df_melt = df_results.melt(id_vars=["embed", "data", "top"],
                            value_vars=["hit_rate", "mrr", "precision", "recall", "ap", "ndcg"],
                            var_name="metric",
                            value_name="value")
    df_melt = df_melt.sort_values(by=["data", "top"])
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    sns.barplot(x="metric", y="value", hue="data", data=df_melt)
    if params.finetune:
        plt.title(f"Evaluation Results for Finetuned {params.embed_name} with top-k {params.top_k}")
    else:
        plt.title(f"Evaluation Results for {params.embed_name} with top-k {params.top_k}")
    plt.xlabel("Metric")
    plt.ylabel("Value")
    plt.legend(title="Data")
    plt.xticks(rotation=45)
    plt.tight_layout()
    # save
    if params.finetune:
        plt.savefig(f"imgs/finetuned_eval_results.png")
    else:
        plt.savefig(f"imgs/baseline_eval_results.png")
    plt.show()
    