# Ready-To-Rag
*To rag or not to rag*

## What is RAG?
RAG (Retrieval-Augmented Generation) is a technique that combines retrieval-based methods with generative models to produce more accurate and contextually relevant responses. It retrieves relevant documents from a large corpus and uses them to generate answers. In this example we used a basic RAG using LlamaIndex and Hugging Face models.


## How to Run the Script
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

## Example

- Query:
> What does it mean - Explainable Anomaly Detection - ?

- Response:
> Explainable Anomaly Detection (EAD) is a process of understanding and explaning anomalies in a machine learning model. EAD helps domain experts to understand and validate the decision-making process of the machine learning model by providing explanations that explain how the model makes predictions. Explaining anomalies is important in many real-world applications, such as in the autonomous car and medical domains, where the lack of understanding and validating the decision-making process of a machine learning system is a disadvantage. EAD can help bridge the gap between detecting outliers and identiﬁcating domain-specific anomalies.
