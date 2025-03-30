# Ready-To-Rag
*To rag or not to rag*

## What is RAG?
RAG (Retrieval-Augmented Generation) is a technique that combines retrieval-based methods with generative models to produce more accurate and contextually relevant responses. It retrieves relevant documents from a large corpus and uses them to generate answers. In this example we used a basic RAG using LlamaIndex and Hugging Face models.


## How to run the RAG

1. Clone the repository:
    ```sh
    git clone https://github.com/giobbu/ready-to-rag.git
    cd ready-to-rag
    ```

2. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

3. Run the script:
    ```sh
    python main.py --query "<query>"
    ```

## Baseline

```sh
nano config.py
```

In the`EmbeddingSettings` dataclass, if `finetuning` is set to `False`, the RAG will not fine-tune the embedding model.

### Example 
- Query:
> What does it mean 'Explainable Anomaly Detection'?

- Response:
> Explainable Anomaly Detection (EAD) is a new approach to anomaly dete-
ction that aims to provide an explanation of the anomaly to the domain experts by providing a visualization of the anomaly's cause. EAD is based on explanation algorithms that provide a detailed explanation of the reason for the anomaly, enabling domain experts to better understand the anomaly and make informed decisions. In this paper, we provide an overview of the current state of EAD and discuss the limitations and challenges of using explanation algorithms in EAD.

## Embedding Finetuning

If `finetuning` is set to `True`, the RAG  fine-tunes a linear adapter on top of the embedding model. 

### Example
- Query:
> What does it mean 'Explainable Anomaly Detection'?

- Response:
> Explainable Anomaly Detection refers to the process of not only identifying anomalies in a system or dataset but also providing clear and understandable explanations for why those anomalies were detected. This approach aims to bridge the gap between detecting outliers and understanding domain-specific anomalies by offering insights into the reasons behind the anomalies. By providing explanations for anomalies, domain experts can better comprehend the detected anomalies, leading to increased trust in the anomaly detection system and facilitating decision-making based on the identified anomalies.

## Evaluation (TODO)

## Hyperparameters Evaluation (TODO)

