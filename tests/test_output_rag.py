import os
from config.setting import Settings
params = Settings()

from dotenv import load_dotenv
load_dotenv()

def test_structured_output(load_rag):
    basic_rag = load_rag
    path_to_rag = os.getenv("PATH_TO_RAG")
    directory_test_document = os.path.join(path_to_rag, params.input_dir_test)
    documents = basic_rag.load_data(input_dir=directory_test_document, required_exts=[".pdf"])
    nodes = basic_rag.parse_documents(documents, chunk_size=params.sentence_splitter_chunk)
    summary_index = basic_rag.create_or_load_summary_idx(nodes=nodes,
                                                        summary_path=params.summ_idx_dir,
                                                        summary_idx=params.summ_idx_name)
    list_tools = [basic_rag.create_summary_query_tool(summary_index)]
    query = "What is the main topic of the document?"
    response = basic_rag.answer(query, list_tools)
    assert response is not None, "Response should not be None"
    assert isinstance(response, dict), "Response should be a dictionary"
    assert "confidence" in response, "Response should contain 'confidence' key"
    assert "response" in response, "Response should contain 'response' key"
    assert "confidence_explanation" in response, "Response should contain 'confidence_explanation' key"