from chromadb.utils.embedding_functions.openai_embedding_function import OpenAIEmbeddingFunction
from langchain_community.llms import Ollama
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# DB
from langchain_community.vectorstores import Chroma

import os
import shutil

from langchain_community.embeddings import HuggingFaceEmbeddings

DATA_PATH = "data"
CHROMA_PATH = "chroma"


def main():
    generate_data_store()


def random_generation(prompt = 'Tell me what the time is in Nepal'):
    llm = Ollama(model="llama2")
    result = llm.invoke(prompt)
    print(result)


def generate_data_store():
    documents = load_documents()
    chunks = split_text(documents)
    save_to_chroma(chunks)

def load_documents():
    loader = DirectoryLoader(DATA_PATH, glob="*.md")
    documents = loader.load()
    return documents


def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 500,
        length_function = len,
        add_start_index = True,
    )

    chunks = text_splitter.split_documents(documents)
    return chunks


def save_to_chroma(chunks):

    # Clear out the database first.
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    # Create embeddings
    model_name = "all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(model_name=model_name)

    """
    Vector Embeddings
    List of numbers in python
    Once you have the words mapped to vector embeddings, you can easily use cosine similarity or Euclidean distance. 
    """

    # load it into Chroma
    db = Chroma.from_documents(chunks, embeddings, persist_directory=CHROMA_PATH)

    db.persist()
    print(f"Saved {len(chunks)} chunk to db.")



if __name__ == "__main__":
    main()