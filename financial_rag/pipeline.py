from decimal import Decimal

from .evidence import Evidence, Selection, ambiguous_request, cite, compact, requested_metrics, validate_claim
from .retrieve import CONTEXT_SIZE, query_scope

REFUSAL = "现有年报证据不足，无法核实该问题。"


def outside_scope(question):
    text = compact(question)
    return any(term in text for term in ("个人财富排名", "适合买入", "实时股价", "未来三年的股息")) or (
        "股价" in text and any(term in text for term in ("今天", "目前", "当前"))
    )


def result(status, answer=REFUSAL, claims=None, citations=None):
    return {"status": status, "answer": answer, "claims": claims or [], "citations": citations or []}


def citation(evidence: Evidence):
    return {key: getattr(evidence, key) for key in ("id", "company", "stock_code", "report_year", "source", "page", "section", "row_label")}


def number_text(question, cell):
    if "亿元" in question and cell.unit in {"元", "千元", "万元", "百万元", "亿元"}:
        converted = cell.normalized() / Decimal(100000000)
        return f"{converted.quantize(Decimal('0.0001'))} 亿元（年报原值 {cell.raw} {cell.unit}）"
    return cell.raw if cell.raw.endswith("%") else f"{cell.raw} {cell.unit}"


def selected_answer(question, selection: Selection, contexts, metrics, companies, years):
    candidates = {row.id: row for row in contexts}
    if metrics:
        values = {}
        for claim in selection.claims:
            evidence, cell = validate_claim(claim, candidates, metrics)
            if cell.year != evidence.report_year:
                raise ValueError("所选数值不是该年年报的当年列")
            if (companies and evidence.stock_code not in companies) or (years and cell.year not in years):
                raise ValueError("证据超出问题指定的公司或年份")
            key = (evidence.stock_code, cell.year, evidence.metric)
            if key in values and cell.normalized() != values[key][1].normalized():
                raise ValueError("同一公司、年份和科目存在冲突")
            values.setdefault(key, (evidence, cell))

        if not values:
            return result("insufficient")
        selected_companies = {key[0] for key in values}
        selected_years = {key[1] for key in values}
        required_years = {min(years), max(years)} if "增长了多少倍" in question and years else years
        if len(selected_companies) != (len(companies) or 1) or (
            required_years and not required_years <= selected_years
        ) or (not years and len(selected_years) != 1):
            return result("insufficient")
        if any((company, year, metric) not in values for company in selected_companies
               for year in selected_years for metric in metrics):
            return result("insufficient")

        rows, claims, citations = [], [], {}
        for evidence, cell in values.values():
            citations.setdefault(evidence.id, citation(evidence))
            ref = list(citations).index(evidence.id) + 1
            claims.append(cite(evidence, cell).model_dump())
            rows.append(f"{evidence.company} {cell.year}年 {evidence.row_label}：{number_text(question, cell)} [{ref}]")
            if "同比" in question:
                prior = next((c for c in evidence.cells if int(c.year) == int(cell.year) - 1 and c.unit == cell.unit), None)
                if prior is None or not prior.normalized():
                    return result("insufficient")
                rate = (cell.normalized() - prior.normalized()) / abs(prior.normalized()) * 100
                claims.append(cite(evidence, prior).model_dump())
                rows.append(f"同比 {rate:.2f}% [{ref}]")
        if "研发费用率" in question or "占当年营业收入的比例" in question:
            denominators = []
            for company, year in sorted({key[:2] for key in values}):
                denominator = values.get((company, year, "revenue"))
                expense = values.get((company, year, "rd_expense"))
                if not denominator or not expense or not denominator[1].normalized():
                    return result("insufficient")
                refs = [list(citations).index(row[0].id) + 1 for row in (expense, denominator)]
                rate = expense[1].normalized() / denominator[1].normalized() * 100
                denominators.append((rate, expense[0].company))
                rows.append(f"{expense[0].company} {year}年 研发费用率 {rate:.2f}% [{refs[0]}][{refs[1]}]")
            if "哪家更高" in question and len(denominators) > 1:
                rows.append(f"研发费用率最高：{max(denominators)[1]}。")
        elif "增长了多少倍" in question and len(selected_companies) == 1 and len(selected_years) > 1 and len(metrics) == 1:
            company = next(iter(selected_companies))
            first, last = min(selected_years), max(selected_years)
            base = values[(company, first, metrics[0])][1].normalized()
            if base <= 0:
                return result("insufficient")
            ratio = values[(company, last, metrics[0])][1].normalized() / base
            rows.append(f"{last}年是{first}年的 {ratio:.3f} 倍，净增 {ratio-1:.3f} 倍。")
        elif "哪家" in question and "更高" in question and len(metrics) == 1:
            rates = [(cell.normalized(), row.company) for row, cell in values.values()]
            if len(rates) > 1:
                rows.append(f"该指标最高：{max(rates)[1]}。")
        return result("answered", "\n".join(rows), claims, list(citations.values()))

    quotes, citations = [], {}
    coverage = set()
    for quote in selection.quotes:
        row = candidates.get(quote.evidence_id)
        if not row or compact(quote.excerpt) not in compact(row.text):
            raise ValueError("引用段落与证据不符")
        citations.setdefault(row.id, citation(row))
        coverage.add((row.stock_code, row.report_year))
        quotes.append(f"{row.company} {row.report_year}年：{quote.excerpt} [{list(citations).index(row.id)+1}]")
    if (not quotes or (companies and not companies <= {c for c, _ in coverage})
            or (years and not years <= {y for _, y in coverage})):
        return result("insufficient")
    return result("answered", "\n".join(quotes), citations=list(citations.values()))


class Pipeline:
    def __init__(self, retriever, reranker, generator):
        self.retriever = retriever
        self.reranker = reranker
        self.generator = generator

    def retrieve(self, question, companies, years, metrics):
        # Reserve space per requested company/year before ranking cross-report questions.
        partitions = [(companies, years)]
        if companies and years and (len(companies) > 1 or len(years) > 1):
            partitions = [({company}, {year}) for company in sorted(companies) for year in sorted(years)]
        if metrics and len(metrics) > 1:
            requests = [(c, y, [metric]) for c, y in partitions for metric in metrics]
        else:
            requests = [(c, y, metrics) for c, y in partitions]
        import jieba

        vector = self.retriever.encoder.encode([question])
        sparse = self.retriever.bm25.get_scores(list(jieba.cut(question)))
        rankings = [self.retriever.rank(question, c, y, m, vector=vector, sparse=sparse) for c, y, m in requests]
        merged = {}
        for level in range(10):
            for ranking in rankings:
                if level < len(ranking):
                    row = ranking[level]
                    merged.setdefault(row.id, row)
        return list(merged.values())[:10]

    def contexts(self, ranked, metrics, companies, years):
        coverage = set()
        chosen = []
        limit = min(10, max(CONTEXT_SIZE, len(companies) * len(years) * max(1, len(metrics))))
        for row in ranked:
            key = (row.stock_code, row.report_year, row.metric)
            if key not in coverage:
                chosen.append(row)
                coverage.add(key)
        chosen_ids = {row.id for row in chosen}
        return (chosen + [row for row in ranked if row.id not in chosen_ids])[:limit]

    def ask(self, question, stock_code=None, year=None, *, debug=False):
        companies, years = query_scope(question, stock_code, year)
        if outside_scope(question) or ambiguous_request(question):
            return result("refused")
        metrics = requested_metrics(question)
        candidates = self.retrieve(question, companies, years, metrics)
        if not candidates:
            return result("insufficient")
        ranked = self.reranker.rank(question, candidates)
        contexts = self.contexts(ranked, metrics, companies, years)
        selection = self.generator.select(question, contexts, bool(metrics))
        try:
            answer = selected_answer(question, selection, contexts, metrics, companies, years)
        except ValueError:
            answer = result("unverified")
        if metrics and answer["status"] in {"unverified", "insufficient"}:
            retry = self.generator.select(question, contexts, True, stricter=True)
            try:
                retry_answer = selected_answer(question, retry, contexts, metrics, companies, years)
            except ValueError:
                retry_answer = result("unverified")
            if retry_answer["status"] == "answered":
                answer = retry_answer
        if debug:
            answer["rrf_ids"] = [row.id for row in candidates]
            answer["reranked_ids"] = [row.id for row in ranked]
            answer["contexts"] = [row.model_dump() for row in contexts]
        return answer
