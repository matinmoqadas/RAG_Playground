
class RAGMetrics:
    @staticmethod
    def precision_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
        top_k = retrieved[:k]
        hits = sum(1 for d in top_k if d in relevant)
        return hits / k if k else 0.0

    @staticmethod
    def recall_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
        top_k = retrieved[:k]
        hits = sum(1 for d in top_k if d in relevant)
        return hits / len(relevant) if relevant else 0.0

    @staticmethod
    def f1_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
        p = RAGMetrics.precision_at_k(retrieved, relevant, k)
        r = RAGMetrics.recall_at_k(retrieved, relevant, k)
        return 2 * p * r / (p + r) if (p + r) else 0.0

    @staticmethod
    def mrr(retrieved: List[str], relevant: List[str]) -> float:
        for rank, doc in enumerate(retrieved, start=1):
            if doc in relevant:
                return 1.0 / rank
        return 0.0

    @staticmethod
    def ndcg_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
        relevant_set = set(relevant)

        def dcg(docs):
            return sum(
                (1.0 / np.log2(i + 2)) for i, d in enumerate(docs[:k]) if d in relevant_set
            )

        ideal_hits = min(len(relevant_set), k)
        ideal_dcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))
        return dcg(retrieved) / ideal_dcg if ideal_dcg else 0.0

    @staticmethod
    def hit_rate_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
        return float(any(d in relevant for d in retrieved[:k]))

    @staticmethod
    def context_precision(
        query: str,
        retrieved_docs: List[str],
        embedding_model,
    ) -> float:

        query_emb = np.array(embedding_model.embed_query(query))
        doc_embs = np.array(embedding_model.embed_documents(retrieved_docs))
        sims = doc_embs @ query_emb / (
            np.linalg.norm(doc_embs, axis=1) * np.linalg.norm(query_emb) + 1e-9
        )
        return float(np.mean(sims))


    @staticmethod
    def evaluate(
        rag: "HybridRAG",
        test_cases: List[Dict],
        k: int = 5,
    ) -> Dict[str, float]:

        metrics_accum: Dict[str, List[float]] = {
            "precision": [],
            "recall": [],
            "f1": [],
            "mrr": [],
            "ndcg": [],
            "hit_rate": [],
            "context_precision": [],
        }

        for tc in test_cases:
            query = tc["query"]
            relevant = tc.get("relevant_docs", [])

            results = rag.hybrid_search(query, top_k=k)
            retrieved = [doc for doc, _ in results]

            metrics_accum["precision"].append(RAGMetrics.precision_at_k(retrieved, relevant, k))
            metrics_accum["recall"].append(RAGMetrics.recall_at_k(retrieved, relevant, k))
            metrics_accum["f1"].append(RAGMetrics.f1_at_k(retrieved, relevant, k))
            metrics_accum["mrr"].append(RAGMetrics.mrr(retrieved, relevant))
            metrics_accum["ndcg"].append(RAGMetrics.ndcg_at_k(retrieved, relevant, k))
            metrics_accum["hit_rate"].append(RAGMetrics.hit_rate_at_k(retrieved, relevant, k))
            metrics_accum["context_precision"].append(
                RAGMetrics.context_precision(query, retrieved, rag.embedding_model)
            )

        return {metric: float(np.mean(vals)) for metric, vals in metrics_accum.items()}
