from llama_index.core.schema import TextNode
from llama_index.core import Settings
from llama_index.core import VectorStoreIndex
from llama_index.core.evaluation import EmbeddingQAFinetuneDataset
import pandas as pd
from tqdm import tqdm
import numpy as np


def display_results(results_dict, embed_model):
    """Display the evaluation results."""
    hit_rates = []
    mrrs = []
    for results in results_dict:
        hit_rate = results["is_hit"]
        mrr = results["mrr"]
        hit_rates.append(hit_rate)
        mrrs.append(mrr)
    final_results = {"retriever": embed_model, 
                        "hit_rate": np.mean(hit_rates), 
                        "mrr": np.mean(mrrs)}
    return final_results

def evaluate(dataset_path, embed_model, top_k=10):
    """Evaluate the retriever using the given dataset and embedding model."""
    
    dataset = EmbeddingQAFinetuneDataset.from_json(dataset_path)

    corpus = dataset.corpus
    queries = dataset.queries
    relevant_docs = dataset.relevant_docs

    embed_model = embed_model or Settings.embed_model
    nodes = [TextNode(id_=id_, text=text) for id_, text in corpus.items()]
    index = VectorStoreIndex(nodes,
                            embed_model=embed_model, 
                            show_progress=True
                            )
    retriever = index.as_retriever(similarity_top_k=top_k)

    eval_results = []
    for query_id, query in tqdm(queries.items()):
        retrieved_nodes = retriever.retrieve(query)
        retrieved_ids = [node.node.node_id for node in retrieved_nodes]
        expected_id = relevant_docs[query_id][0]

        rank = None
        for idx, id in enumerate(retrieved_ids):
            if id == expected_id:
                rank = idx + 1
                break

        is_hit = rank is not None  # assume 1 relevant doc
        mrr = 0 if rank is None else 1 / rank

        eval_result = {
                        "is_hit": is_hit,
                        "mrr": mrr
                        }
        eval_results.append(eval_result)

    df_results = pd.DataFrame([])
    final_results = display_results(eval_results, embed_model)
    
    # append the results to the dataframe
    if df_results.empty:
        df_results = pd.DataFrame([final_results])
    else:
        df_results = pd.concat([df_results, pd.DataFrame([final_results])], ignore_index=True)

    return df_results


if __name__ == "__main__":
    
    val_dataset_path = "save/gpt/train_dataset_gpt.json"
    bge = "local:BAAI/bge-small-en"
    df_results = evaluate(val_dataset_path, bge)

    print(df_results)
    
    