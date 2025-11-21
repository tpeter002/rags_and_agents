import os
import sys
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.readers.file import PDFReader
from dotenv import load_dotenv

# load api key
load_dotenv("secrets.env")
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    print("Error: GEMINI_API_KEY not found.")
    print("Please create a .env file and set the key.")
    sys.exit(1)

# where documents are stored
DOCUMENTS_DIR = "docs"

def setup_rag_pipeline():
    """
    Sets up the LlamaIndex RAG pipeline: Load -> Index -> Query Engine.
    """
    print("1. data loading")

    if not os.path.exists(DOCUMENTS_DIR):
        os.makedirs(DOCUMENTS_DIR)
        print(f"Created directory: '{DOCUMENTS_DIR}'")
        print(f"Please place your PDF or text files inside the '{DOCUMENTS_DIR}' folder and re-run the script.")
        return None

    # use pypdf extractor, could be extended to basically anything
    file_extractor = {
        ".pdf": PDFReader(),
    }
    documents = SimpleDirectoryReader(
        input_dir=DOCUMENTS_DIR,
        file_extractor=file_extractor
    ).load_data()

    print(f"successfully loaded {len(documents)} documents.")
    print("2. indexing and embeddings")

    llm = Gemini(model="gemini-2.5-flash", api_key=API_KEY)
    
    # use local vectorizer, because that is the simplest
    print("Initializing local BGE embedding model...")
    embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-small-en-v1.5"
    )

    # chunks the documents and uses the local model to generate embeddings
    index = VectorStoreIndex.from_documents(
        documents,
        llm=llm,
        embed_model=embed_model
    )

    print("3. create query engine")
    query_engine = index.as_query_engine(
        response_mode="compact", # concise answers
        llm=llm
    )
    return query_engine

def chat_loop(query_engine):
    """
    Starts an interactive chat loop for the user to query the document index.
    """
    if query_engine is None:
        return

    print("\nStart Chatting with your Documents")
    print("Type your question or 'exit' to quit.")
    print("-" * 50)

    while True:
        prompt = input("\nYou: ")

        if prompt.lower() == 'exit':
            print("Thank you for using the RAG pipeline. Goodbye!")
            break

        if not prompt.strip():
            continue

        try:
            # retrieval uses the local embedding model
            # synthesis uses the cloud gemini
            response = query_engine.query(prompt)

            print(f"\nAI Assistant: {response.response}")

            # print the sources used for grounding
            source_nodes = response.source_nodes
            if source_nodes:
                print("\n[Sources Used for Grounding]:")
                for node in source_nodes:
                    file_name = node.metadata.get('file_name', 'Unknown File')
                    page_label = node.metadata.get('page_label', 'N/A')
                    print(f" - File: {file_name}, Page: {page_label}")
            else:
                print("\n no specific source nodes were retrieved for this query")

        except Exception as e:
            print(f"\nAn error occurred during query: {e}")
            print("Please check your API key for the Gemini LLM.")


if __name__ == "__main__":
    query_engine = setup_rag_pipeline()
    chat_loop(query_engine)