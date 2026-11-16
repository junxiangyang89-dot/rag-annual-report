import json
import unittest
from decimal import Decimal
from types import SimpleNamespace

import httpx

from financial_rag.evidence import Cell, Evidence, Selection, cite, metric_of
from financial_rag.parse import parse_rows
from financial_rag.pipeline import selected_answer
from financial_rag.providers import ProviderError, QwenReranker
from financial_rag.retrieve import query_scope


def sample():
    return Evidence(
        id="profit", stock_code="000858", company="五粮液", report_year="2022",
        source="五粮液年报.pdf", page=60, section="财务报告 > 合并利润表", text="归母净利润",
        kind="row", row_label="1.归属于母公司股东的净利润", metric="parent_profit",
        cells=[
            Cell(column="2022年度", year="2022", raw="26,690,661,397.42", unit="元"),
            Cell(column="2021年度", year="2021", raw="23,377,074,353.40", unit="元"),
        ],
    )


class Tests(unittest.TestCase):
    def test_exact_row_and_value(self):
        row = sample()
        q = "五粮液2022年归母净利润是多少亿元？"
        sel = Selection(claims=[cite(row, row.cells[0])])
        out = selected_answer(q, sel, [row], ["parent_profit"], {"000858"}, {"2022"})
        self.assertEqual(out["status"], "answered")
        self.assertIn("266.9066 亿元", out["answer"])

    def test_reject_wrong_row_even_with_same_amount(self):
        row = sample().model_copy(update={"row_label": "归属于母公司所有者的综合收益总额"})
        self.assertIsNone(metric_of(row.row_label))
        with self.assertRaises(ValueError):
            selected_answer("归母净利润", Selection(claims=[cite(row, row.cells[0])]),
                            [row], ["parent_profit"], {"000858"}, {"2022"})

    def test_reject_prior_year(self):
        row = sample()
        with self.assertRaises(ValueError):
            selected_answer("归母净利润", Selection(claims=[cite(row, row.cells[1])]),
                            [row], ["parent_profit"], {"000858"}, {"2022"})

    def test_reject_changed_citation(self):
        row = sample()
        for key, value in {"page": 61, "section": "其他", "unit": "亿元", "raw": "23"}.items():
            with self.subTest(key=key), self.assertRaises(ValueError):
                changed = cite(row, row.cells[0]).model_copy(update={key: value})
                selected_answer("归母净利润", Selection(claims=[changed]), [row], ["parent_profit"], {"000858"}, {"2022"})

    def test_spans_preserve_column_alignment(self):
        boxes = [(0, 0, 200, 15), (205, 0, 300, 15), (305, 0, 400, 15)]
        header = ["项目", "2022年度", "2021年度"]
        for has_header in (True, False):
            rows = [header, ["归母净利润", "26,690,661,397.42", "23,377,074,353.40"]] if has_header else [
                ["归母净利润", "26,690,661,397.42", "23,377,074,353.40"]
            ]
            table = SimpleNamespace(extract=lambda: rows, rows=[SimpleNamespace(cells=boxes) for _ in rows])
            data = list(parse_rows(table, "2022", [(header[1], "2022", boxes[1]), (header[2], "2021", boxes[2])], "元"))
            self.assertEqual([c.normalized() for c in data[0][3]],
                             [Decimal("26690661397.42"), Decimal("23377074353.40")])

    def test_year_range_and_conflict(self):
        self.assertEqual(query_scope("贵州茅台2021年至2023年营收"), ({"600519"}, {"2021", "2022", "2023"}))
        with self.assertRaises(ValueError):
            query_scope("五粮液2022年归母净利润", "600519", "2022")

    def test_rerank_mapping_and_invalid_index(self):
        rows = [sample(), sample().model_copy(update={"id": "alternative"})]
        for ranks in ([1, 0], [1, 1]):
            def handler(request):
                body = json.loads(request.content)
                self.assertEqual(body["top_n"], 2)
                return httpx.Response(200, json={"results": [
                    {"index": index, "relevance_score": 1-i/2} for i, index in enumerate(ranks)
                ]})
            with httpx.Client(transport=httpx.MockTransport(handler)) as client:
                reranker = QwenReranker.__new__(QwenReranker)
                reranker.http, reranker.model, reranker.url, reranker.key = client, "qwen-test", "https://example.invalid/reranks", "test"
                if len(set(ranks)) != len(ranks):
                    with self.assertRaises(ProviderError):
                        reranker.rank("归母净利润", rows)
                else:
                    self.assertEqual([r.id for r in reranker.rank("归母净利润", rows)], ["alternative", "profit"])


if __name__ == "__main__":
    unittest.main()
