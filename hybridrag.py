import numpy as np
import re
from typing import List, Tuple, Dict, Optional
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from rank_bm25 import BM25Okapi
from adaptivechunking import AdaptiveChunker
from ragmetrics import RAGMetrics



def load_pdf(path: str, chunker: Optional[AdaptiveChunker] = None) -> List[str]:

    try:
        from pypdf import PdfReader
    except ImportError:
        raise ImportError("pypdf is required")

    reader = PdfReader(path)
    pages_text = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages_text.append(text.strip())

    full_text = "\n\n".join(pages_text)

    if chunker is None:
        chunker = AdaptiveChunker()

    return chunker.chunk(full_text)



class HybridRAG:
    def __init__(
        self,
        documents: List[str],
        embedding_model=None,
        chunker: Optional[AdaptiveChunker] = None,
        auto_chunk: bool = False,
    ):
        self.embedding_model = embedding_model or OpenAIEmbeddings()
        self.chunker = chunker or AdaptiveChunker()

        if auto_chunk:
            documents = [
                chunk
                for doc in documents
                for chunk in self.chunker.chunk(doc)
            ]

        self.documents = documents
        self.faiss_index = FAISS.from_texts(documents, self.embedding_model)
        self.bm25 = BM25Okapi([doc.split() for doc in documents])
        self.metrics = RAGMetrics()

    @classmethod
    def from_pdf(
        cls,
        pdf_path: str,
        embedding_model=None,
        chunker: Optional[AdaptiveChunker] = None,
    ) -> "RAG":
        _chunker = chunker or AdaptiveChunker()
        chunks = load_pdf(pdf_path, chunker=_chunker)
        return cls(documents=chunks, embedding_model=embedding_model, chunker=_chunker)

    def semantic_search(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        docs_and_scores = self.faiss_index.similarity_search_with_score(query, k=top_k)
        return [(doc.page_content, score) for doc, score in docs_and_scores]

    def bm25_search(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        scores = self.bm25.get_scores(query.split())
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(self.documents[i], scores[i]) for i in top_indices]

    def hybrid_search(
        self, query: str, top_k: int = 5, alpha: float = 0.5
    ) -> List[Tuple[str, float]]:
        sem_results = dict(self.semantic_search(query, top_k=top_k * 2))
        bm25_results = dict(self.bm25_search(query, top_k=top_k * 2))

        sem_max = max(sem_results.values(), default=1)
        bm25_max = max(bm25_results.values(), default=1)

        sem_norm = {k: v / sem_max for k, v in sem_results.items()}
        bm25_norm = {k: v / bm25_max for k, v in bm25_results.items()}

        all_docs = set(sem_results) | set(bm25_results)
        hybrid_scores = {
            doc: alpha * sem_norm.get(doc, 0) + (1 - alpha) * bm25_norm.get(doc, 0)
            for doc in all_docs
        }
        ranked = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return ranked

    def hyde(self, question: str, model=None) -> str:
        from langchain.llms import OpenAI
        llm = model or OpenAI(temperature=0)
        return llm(question)

    def hyde_hybrid_search(
        self, question: str, top_k: int = 5, alpha: float = 0.5, model=None
    ) -> List[Tuple[str, float]]:
        hypothetical_answer = self.hyde(question, model=model)
        return self.hybrid_search(hypothetical_answer, top_k=top_k, alpha=alpha)

    def evaluate(self, test_cases: List[Dict], k: int = 5) -> Dict[str, float]:
        """Run RAGMetrics.evaluate on this instance."""
        return RAGMetrics.evaluate(self, test_cases=test_cases, k=k)
