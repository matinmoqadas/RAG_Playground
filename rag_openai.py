import numpy as np
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from rank_bm25 import BM25Okapi
from typing import List, Tuple

class RAG:
    def __init__(self, documents: List[str], embedding_model=None):
        self.documents = documents
        self.embedding_model = embedding_model or OpenAIEmbeddings()
        # Semantic Index (FAISS)
        self.faiss_index = FAISS.from_texts(documents, self.embedding_model)
        # BM25 Index
        self.bm25 = BM25Okapi([doc.split() for doc in documents])
        
    def semantic_search(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        docs_and_scores = self.faiss_index.similarity_search_with_score(query, k=top_k)
        return [(doc.page_content, score) for doc, score in docs_and_scores]

    def bm25_search(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        scores = self.bm25.get_scores(query.split())
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(self.documents[i], scores[i]) for i in top_indices]
    
    def hybrid_search(self, query: str, top_k: int = 5, alpha: float = 0.5) -> List[Tuple[str, float]]:
        # alpha: weighting between semantic and bm25 (0=only bm25, 1=only semantic)
        sem_results = dict(self.semantic_search(query, top_k=top_k*2))
        bm25_results = dict(self.bm25_search(query, top_k=top_k*2))

        # Normalize scores
        sem_max = max(sem_results.values(), default=1)
        bm25_max = max(bm25_results.values(), default=1)
        sem_results_norm = {k: v / sem_max for k, v in sem_results.items()}
        bm25_results_norm = {k: v / bm25_max for k, v in bm25_results.items()}

        # Union candidate docs
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
        This improves semantic search by retrieving documents similar to the hypothetical answer.
        """
        # Use a small LLM for HyDE. For demo, just echo the question, but you can plug in OpenAI or HuggingFace models.
        # Example with langchain LLM:
        from langchain.llms import OpenAI
        llm = model or OpenAI(temperature=0)
        hypothetical_answer = llm(question)
        return hypothetical_answer

    def hyde_hybrid_search(self, question: str, top_k: int = 5, alpha: float = 0.5, model=None) -> List[Tuple[str, float]]:
        hypothetical_answer = self.hyde(question, model=model)
        return self.hybrid_search(hypothetical_answer, top_k=top_k, alpha=alpha)
