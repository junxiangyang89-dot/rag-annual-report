import argparse
import json
import time
from collections import Counter
from decimal import Decimal
from pathlib import Path

import httpx

from financial_rag.evidence import Cell, Evidence, metric_of, requested_metrics
from financial_rag.pipeline import Pipeline, outside_scope
from financial_rag.providers import DeepSeek, ProviderError, QwenReranker
from financial_rag.retrieve import Retriever, query_scope

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    return json.loads((ROOT / "evaluation" / name).read_text(encoding="utf-8"))


def documents(rows):
    return {f"{row.stock_code}_{row.report_year}" for row in rows}


def gold_matches(row, gold):
    return (
        row.stock_code == gold["stock_code"] and row.report_year == gold["year"]
        and row.metric == gold["metric"] and row.page in gold["pdf_pages"]
        and any(c.year == gold["year"] and c.normalized() == Decimal(gold["normalized"]) for c in row.cells)
    )


def claim_matches(claim, citation, gold):
    if citation["stock_code"] != gold["stock_code"] or claim["year"] != gold["year"]:
        return False
    cell = Cell.model_validate({key: claim[key] for key in ("column", "year", "raw", "unit")})
    return (
        metric_of(claim["row_label"]) == gold["metric"] and claim["page"] in gold["pdf_pages"]
        and cell.normalized() == Decimal(gold["normalized"])
    )


def evaluate(retriever, reranker=None, generator=None):
    questions = load("questions.json")["questions"]
    labels = load("gold.json")["cases"]
    pipeline = Pipeline(retriever, reranker, generator)
    results = []
    for q in questions:
        start = time.perf_counter()
        question = q["question"]
        gold = labels.get(str(q["id"]), [])
        targets = set(q["target_docs"])
        companies, years = query_scope(question)
        metrics = requested_metrics(question)
        result = {"id": q["id"], "question": question, "type": q["type"], "gold_cells": len(gold)}
        if q["type"] == "should_refuse":
            result["scope_refusal"] = outside_scope(question)
            if generator:
                answer = pipeline.ask(question)
                result.update(response_status=answer["status"], answer=answer["answer"])
        else:
            candidates = pipeline.retrieve(question, companies, years, metrics)
            before = pipeline.contexts(candidates, metrics, companies, years)
            result.update(
                rrf_count=len(candidates), rrf_doc_coverage=targets <= documents(before),
                rrf_gold_hits=sum(any(gold_matches(e, g) for e in before) for g in gold),
            )
            if reranker:
                try:
                    answer = pipeline.ask(question, debug=True)
                except ProviderError as error:
                    result.update(response_status="provider_error", error=str(error))
                else:
                    contexts = answer.get("contexts", [])
                    rows = [Evidence.model_validate(item) for item in contexts]
                    result.update(
                        response_status=answer["status"], answer=answer["answer"],
                        citations=answer["citations"], claims=answer["claims"],
                        rerank_doc_coverage=targets <= documents(rows),
                        rerank_gold_hits=sum(any(gold_matches(e, g) for e in rows) for g in gold),
                    )
                    citations = {citation["id"]: citation for citation in answer["citations"]}
                    result["answer_gold_hits"] = sum(any(
                        claim_matches(c, citations[c["evidence_id"]], g) for c in answer["claims"]
                    ) for g in gold)
        result["seconds"] = round(time.perf_counter() - start, 3)
        print(f"Q{q['id']:02d} {result.get('response_status', 'offline')} "
              f"rrf={result.get('rrf_count', 0)} gold={result.get('answer_gold_hits', '-')} "
              f"{result['seconds']:.2f}s", flush=True)
        results.append(result)

    return {
        "mode": "online" if reranker else "retrieval_only",
        "reranker": reranker.model if reranker else None,
        "generator": generator.model if generator else None,
        "questions": len(questions),
        "indexed_evidence": len(retriever.rows),
        "indexed_rows": sum(row.kind == "row" for row in retriever.rows),
        "gold_cells": sum(r["gold_cells"] for r in results),
        "rrf_doc_coverage": sum(r.get("rrf_doc_coverage", False) for r in results),
        "rrf_gold_hits": sum(r.get("rrf_gold_hits", 0) for r in results),
        "scope_refusals": sum(r.get("scope_refusal", False) for r in results),
        **({
            "rerank_doc_coverage": sum(r.get("rerank_doc_coverage", False) for r in results),
            "rerank_gold_hits": sum(r.get("rerank_gold_hits", 0) for r in results),
            "answer_gold_hits": sum(r.get("answer_gold_hits", 0) for r in results),
            "status_counts": dict(Counter(r["response_status"] for r in results)),
        } if reranker else {}),
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--online", action="store_true")
    parser.add_argument("--check", action="store_true", help="校验五粮液错例和 3 道拒答题")
    parser.add_argument("--out", type=Path, default=ROOT / "report" / "regression.json")
    args = parser.parse_args()
    retriever = Retriever(ROOT)
    if args.online:
        with httpx.Client(timeout=40) as http, httpx.Client(timeout=20, trust_env=False) as rerank_http:
            report = evaluate(retriever, QwenReranker(rerank_http), DeepSeek(http))
    else:
        report = evaluate(retriever)
    args.out.parent.mkdir(exist_ok=True, parents=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "results"}, ensure_ascii=False))
    if report.get("status_counts", {}).get("provider_error"):
        raise SystemExit(1)
    if args.check:
        critical = next(r for r in report["results"] if r["id"] == 10)
        hits = critical["answer_gold_hits"] if args.online else critical["rrf_gold_hits"]
        refusals = [r for r in report["results"] if r["type"] == "should_refuse"]
        correct_refusals = (
            all(r["response_status"] == "refused" for r in refusals)
            if args.online else all(r["scope_refusal"] for r in refusals)
        )
        if hits != critical["gold_cells"] or not correct_refusals:
            raise SystemExit("关键错例或拒答回归失败")


if __name__ == "__main__":
    main()
