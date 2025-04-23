from pydantic import BaseModel

class EmbeddingSettings(BaseModel):
    embed_name: str = "BAAI/bge-small-en-v1.5"
    input_dir_train: str = "data/finetune-embed/train"
    input_dir_val: str = "data/finetune-embed/val"
    input_dir_test: str = "data/paper/"
    out_dir_train: str = "save/qa/train_dataset_gpt.json"
    out_dir_val: str = "save/qa/val_dataset_gpt.json"
    run_finetuning: bool = False
    use_finetuned_model: bool = False
    adapter: str = "linear"
    sent_transf_params : dict = {"bias" : True,
                                "epochs" : 10}
    model_output_path: str = "save/embedding/baai/linear_adapter_model_output"