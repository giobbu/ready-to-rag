import os
from config.setting import Settings
params = Settings()

from dotenv import load_dotenv
load_dotenv()

def test_input_data(load_rag):
    " Test if the input data is loaded correctly"
    basic_rag = load_rag
    path_to_rag = os.getenv("PATH_TO_RAG")
    directory_test_document = os.path.join(path_to_rag, params.input_dir_test)
    assert os.path.exists(directory_test_document), f"Directory {directory_test_document} does not exist"
    documents = basic_rag.load_data(input_dir=directory_test_document, required_exts=[".pdf"])
    assert documents is not None, "Documents should not be None"
    assert len(documents) > 0, "Documents should not be empty"

def test_parse_documents(load_rag):
    " Test if the documents are parsed correctly"
    basic_rag = load_rag
    path_to_rag = os.getenv("PATH_TO_RAG")
    directory_test_document = os.path.join(path_to_rag, params.input_dir_test)
    documents = basic_rag.load_data(input_dir=directory_test_document, required_exts=[".pdf"])
    nodes = basic_rag.parse_documents(documents, chunk_size=params.sentence_splitter_chunk)
    assert nodes is not None, "Parsed nodes should not be None"
    assert len(nodes) > 0, "Parsed nodes should not be empty"