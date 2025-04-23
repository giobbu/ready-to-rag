from pydantic import BaseModel

class StoreIndex(BaseModel):
    vec_store_idx_dir: str = "storage/vector_store_index"
    vec_store_idx_name: str = "vector_store_index"
    summ_idx_dir: str = "storage/summary_index"
    summ_idx_name: str = "summary_index"