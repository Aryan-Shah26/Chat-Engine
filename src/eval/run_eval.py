from src.eval.dataset import load_qa_dataset
from src.eval.faithfulness import faithfulness
from src.eval.metrics import recall_at_k, mrr, context_precision

# Each strategy: name -> fn(query: str) -> list[Document]
Strategies = dict[str, callable]


def run_eval(strategies: Strategies, k: int = 5, dataset_path: str = None) -> dict:
    """
    Runs every strategy over the QA dataset and returns averaged metrics:
    {strategy_name: {"recall@k": float, "mrr": float, "context_precision": float}}
    """
    dataset = load_qa_dataset(dataset_path) if dataset_path else load_qa_dataset()
    results = {name: {"recall@k": 0.0, "mrr": 0.0, "context_precision": 0.0} for name in strategies}

    for entry in dataset:
        question, relevant = entry["question"], entry["relevant"]
        for name, retrieve_fn in strategies.items():
            retrieved = retrieve_fn(question)
            results[name]["recall@k"] += recall_at_k(retrieved, relevant, k)
            results[name]["mrr"] += mrr(retrieved, relevant)
            results[name]["context_precision"] += context_precision(retrieved, relevant, k)

    n = len(dataset)
    for name in results:
        results[name] = {metric: round(val / n, 4) for metric, val in results[name].items()}
    return results


def run_faithfulness_eval(client, strategies: dict, dataset_path: str = None) -> dict:
    """
    End-to-end (generation-level) eval, separate from the retrieval-only
    metrics above. Each strategy: name -> fn(question: str) -> (answer, docs).
    Returns {strategy_name: avg_faithfulness}.
    """
    dataset = load_qa_dataset(dataset_path) if dataset_path else load_qa_dataset()
    totals = {name: 0.0 for name in strategies}
    for entry in dataset:
        for name, fn in strategies.items():
            answer, docs = fn(entry["question"])
            totals[name] += faithfulness(client, answer, docs)
    n = len(dataset)
    return {name: round(v / n, 4) for name, v in totals.items()}


if __name__ == "__main__":
    from src.retrieval.retriever import load_chroma_retriever
    from src.retrieval.hybrid_retriever import load_hybrid_index, hybrid_search

    dense_retriever, _ = load_chroma_retriever(top_k=5)
    _, bm25_retriever, _ = load_hybrid_index(top_k=5)

    strategies = {
        "dense_only": lambda q: dense_retriever.invoke(q)[:5],
        "hybrid_rrf_rerank": lambda q: hybrid_search(q, dense_retriever, bm25_retriever, top_k=5),
    }

    print(run_eval(strategies, k=5))