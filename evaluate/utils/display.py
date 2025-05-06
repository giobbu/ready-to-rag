import pandas as pd

def display_results_embedding(name:str, data:str, top: str, eval_results: list) -> pd.DataFrame:
    """Display results from evaluate."""
    metric_dicts = [eval_result.metric_vals_dict for eval_result in eval_results]
    full_df = pd.DataFrame(metric_dicts)
    metrics = ["hit_rate", "mrr", "ap", "ndcg"]
    columns = {"embed": [name],
                "data": [data],
                "top": [top], 
                **{k: [full_df[k].mean()] for k in metrics}}
    metric_df = pd.DataFrame(columns)
    return metric_df


def plot_results_embedding(df_results: pd.DataFrame, finetune:bool = False, 
                           params=None) -> None:
    """Plot results from evaluate."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    df_melt = df_results.melt(id_vars=["embed", "data", "top"],
                            value_vars=["hit_rate", "mrr", "ap", "ndcg"],
                            var_name="metric",
                            value_name="value")
    df_melt = df_melt.sort_values(by=["data", "top"])
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    sns.barplot(x="metric", y="value", hue="data", data=df_melt)
    if finetune:
        plt.title(f"Evaluation Results for Finetuned {params.embed_name} with top-k {params.top_k}")
    else:
        plt.title(f"Evaluation Results for {params.embed_name} with top-k {params.top_k}")
    plt.xlabel("Metric")
    plt.ylabel("Value")
    plt.legend(title="Data")
    plt.xticks(rotation=45)
    plt.tight_layout()
    # save
    output_file = f"imgs/{'finetuned' if finetune else 'baseline'}_eval_results.png"
    plt.savefig(output_file)
    plt.show()