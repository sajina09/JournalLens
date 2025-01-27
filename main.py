from langchain_community.llms import Ollama
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document


DATA_PATH = "data"


def main():
    generate_data_store()


def random_generation(prompt = 'Tell me what the time is in Nepal'):
    llm = Ollama(model="llama2")
    result = llm.invoke(prompt)
    print(result)


def generate_data_store():
    documents = load_documents()
    chunks = split_text(documents)
    # save_to_chroma(chunks)

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
    print("Chunks", chunks)

    return chunks


if __name__ == "__main__":
    main()