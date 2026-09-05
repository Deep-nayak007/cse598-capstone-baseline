#!/usr/bin/env python3
"""Minimal reproducible baseline for a course-policy Q&A assistant.

The baseline uses TF-IDF cosine similarity over a small JSON knowledge base.
It is intentionally simple so the retrieval behavior is easy to reproduce.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import textwrap
from collections import Counter
from pathlib import Path


TOKEN_RE = re.compile(r"[a-z0-9]+")
STOP_WORDS = {
    "a", "an", "and", "are", "at", "be", "by", "can", "do", "does",
    "for", "how", "i", "in", "is", "it", "my", "of", "on", "or", "the",
    "this", "to", "what", "when", "where", "with",
}


def tokenize(text: str) -> list[str]:
    return [
        token
        for token in TOKEN_RE.findall(text.lower())
        if token not in STOP_WORDS
    ]


def tf(tokens: list[str]) -> dict[str, float]:
    counts = Counter(tokens)
    total = sum(counts.values()) or 1
    return {term: count / total for term, count in counts.items()}


def cosine(a: dict[str, float], b: dict[str, float]) -> float:
    dot = sum(a.get(term, 0.0) * b.get(term, 0.0) for term in a)
    norm_a = math.sqrt(sum(value * value for value in a.values()))
    norm_b = math.sqrt(sum(value * value for value in b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def build_vectors(records: list[dict[str, str]]) -> tuple[list[dict[str, float]], dict[str, float]]:
    documents = [f"{item['question']} {item['answer']}" for item in records]
    tokenized_docs = [tokenize(doc) for doc in documents]
    doc_count = len(tokenized_docs)
    document_frequency = Counter(term for tokens in tokenized_docs for term in set(tokens))
    idf = {
        term: math.log((doc_count + 1) / (freq + 1)) + 1
        for term, freq in document_frequency.items()
    }
    vectors = []
    for tokens in tokenized_docs:
        vector = {term: weight * idf[term] for term, weight in tf(tokens).items()}
        vectors.append(vector)
    return vectors, idf


def vectorize_query(query: str, idf: dict[str, float]) -> dict[str, float]:
    return {term: weight * idf.get(term, 1.0) for term, weight in tf(tokenize(query)).items()}


def answer_question(query: str, records: list[dict[str, str]], vectors: list[dict[str, float]], idf: dict[str, float]) -> dict[str, object]:
    query_vector = vectorize_query(query, idf)
    scored = [
        (cosine(query_vector, vector), record)
        for vector, record in zip(vectors, records)
    ]
    score, best = max(scored, key=lambda item: item[0])
    if score < 0.10:
        return {
            "question": query,
            "status": "needs_clarification",
            "confidence": round(score, 3),
            "matched_source": None,
            "answer": "I could not find a close enough policy match in the current knowledge base."
        }
    return {
        "question": query,
        "status": "answered",
        "confidence": round(score, 3),
        "matched_source": best["id"],
        "answer": best["answer"]
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kb", default="examples/course_faq.json", help="Path to the JSON knowledge base.")
    parser.add_argument("--input", default="examples/test_questions.json", help="Path to a JSON list of questions.")
    parser.add_argument("--output", default="output/baseline_output.json", help="Path where results will be written.")
    args = parser.parse_args()

    records = json.loads(Path(args.kb).read_text())
    questions = json.loads(Path(args.input).read_text())
    vectors, idf = build_vectors(records)
    results = [answer_question(question, records, vectors, idf) for question in questions]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2) + "\n")

    print(f"Wrote {len(results)} answers to {output_path}")
    for result in results:
        print(f"- [{result['status']}] {result['question']}")
        print(f"  source={result['matched_source']} confidence={result['confidence']}")
        print(textwrap.fill(
            f"answer={result['answer']}",
            width=88,
            initial_indent="  ",
            subsequent_indent="    ",
        ))


if __name__ == "__main__":
    main()
