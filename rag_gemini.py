import numpy as np
from langchain.embeddings import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from rank_bm25 import BM25Okapi
from typing import List, Tuple

class GeminiRAG:
    def __init__(self, documents: List[str], embedding_model=None, knoeledge_base=None):
        self.documents = documents
        # Use Gemini embedding model via LangChain (GoogleGenerativeAIEmbeddings)
        self.embedding_model = embedding_model or GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        # Semantic Index (FAISS)
        self.faiss_index = FAISS.from_texts(documents, self.embedding_model)
        # BM25 Index
        self.bm25 = BM25Okapi([doc.split() for doc in documents])
          """
        Initialize the GeminiRAG instance.

        Args:
            knowledge_base: Optional knowledge base object to perform RAG.
        """
        self.knowledge_base = knowledge_base
        
    def semantic_search(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        docs_and_scores = self.faiss_index.similarity_search_with_score(query, k=top_k)
        return [(doc.page_content, score) for doc, score in docs_and_scores]

    def bm25_search(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        scores = self.bm25.get_scores(query.split())
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(self.documents[i], scores[i]) for i in top_indices]
    
    def hybrid_search(self, query: str, top_k: int = 5, alpha: float = 0.5) -> List[Tuple[str, float]]:
        sem_results = dict(self.semantic_search(query, top_k=top_k*2))
        bm25_results = dict(self.bm25_search(query, top_k=top_k*2))
        sem_max = max(sem_results.values(), default=1)
        bm25_max = max(bm25_results.values(), default=1)
        sem_results_norm = {k: v / sem_max for k, v in sem_results.items()}
        bm25_results_norm = {k: v / bm25_max for k, v in bm25_results.items()}
        all_docs = set(sem_results) | set(bm25_results)
        hybrid_scores = {
            doc: alpha * sem_results_norm.get(doc, 0) + (1 - alpha) * bm25_results_norm.get(doc, 0)
            for doc in all_docs
        }
        ranked = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return ranked

    def hyde(self, question: str, model=None) -> str:
        """
        Generate a hypothetical answer (HyDE) to the question to use as a retrieval query.
        Uses Gemini via LangChain.
        """
        # Use LangChain's Gemini LLM wrapper (GoogleGenerativeAI)
        from langchain.llms import GoogleGenerativeAI
        llm = model or GoogleGenerativeAI(model="models/gemini-pro", temperature=0)
        hypothetical_answer = llm(question)
        return hypothetical_answer

    def hyde_hybrid_search(self, question: str, top_k: int = 5, alpha: float = 0.5, model=None) -> List[Tuple[str, float]]:
        hypothetical_answer = self.hyde(question, model=model)
        return self.hybrid_search(hypothetical_answer, top_k=top_k, alpha=alpha)

    def chat(self, prompt: str) -> str:
        """
        Generate a response to the given prompt using RAG.

        Args:
            prompt (str): The user input prompt.

        Returns:
            str: The generated response.
        """
        # Example implementation (replace with actual logic)
        if self.knowledge_base:
            context = self.knowledge_base.retrieve(prompt)
            return f"Context: {context}\nResponse: This is a generated answer."
        return "Response: This is a generated answer."
