import os
from typing import List, Dict, Any
from dotenv import load_dotenv
from .vectorstore import FaissVectorStore
from langchain_groq import ChatGroq

load_dotenv()


class RAGSearch:
    """Retrieval-Augmented Generation search using FAISS + Groq LLM."""

    def __init__(self, persist_dir: str = "faiss_store", embedding_model: str = "all-MiniLM-L6-v2", llm_model: str = "llama3-8b-8192"):
        self.vectorstore = FaissVectorStore(persist_dir, embedding_model)
        faiss_path = os.path.join(persist_dir, "index.faiss")
        meta_path = os.path.join(persist_dir, "metadata.pkl")
        if not (os.path.exists(faiss_path) and os.path.exists(meta_path)):
            from .data_loader import load_all_documents
            docs = load_all_documents("data")
            self.vectorstore.build_from_documents(docs)

        groq_api_key = os.getenv("GROQ_API_KEY", "")
        self.llm = None
        if groq_api_key:
            try:
                self.llm = ChatGroq(groq_api_key=groq_api_key, model_name=llm_model)
                print(f"[INFO] Groq LLM initialized: {llm_model}")
            except Exception as e:
                print(f"[WARNING] Could not initialize Groq LLM: {e}")
        else:
            print("[INFO] GROQ_API_KEY not set. LLM summarization disabled.")

    def search_and_summarize(self, query: str, top_k: int = 5) -> str:
        """Retrieve relevant chunks and summarize using LLM (if available)."""
        results = self.vectorstore.search(query, top_k=top_k)
        if not results:
            return "No relevant documents found."
        texts = [r.get("text", "") for r in results if r.get("text")]
        context = "\n\n".join(texts)
        if not context:
            return "No relevant context extracted from documents."
        if self.llm:
            try:
                prompt = f"Summarize the following context for the query: '{query}'\n\nContext:\n{context}\n\nSummary:"
                response = self.llm.invoke([prompt])
                return str(response.content)
            except Exception as e:
                print(f"[WARNING] LLM invocation failed: {e}")
        return f"Retrieved Context for query '{query}':\n\n" + context

    def query(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Return raw search results from the vector store."""
        return self.vectorstore.search(query, top_k=top_k)


if __name__ == "__main__":
    rag_search = RAGSearch()
    query = "What is attention mechanism?"
    summary = rag_search.search_and_summarize(query, top_k=3)
    print("Summary:", summary)