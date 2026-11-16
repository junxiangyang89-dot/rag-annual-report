import json
import os
import re
import threading
from pathlib import Path
from tempfile import NamedTemporaryFile
from zipfile import ZIP_STORED, ZipFile

import faiss
import jieba
import numpy as np
import torch
from rank_bm25 import BM25Okapi
from transformers import AutoModel, AutoTokenizer

from .evidence import Evidence
from .parse import load_evidence

INDEX_VERSION = 1
VECTOR_MODEL = "bge-small-zh-v1.5"
CANDIDATES = 10
CONTEXT_SIZE = 4

COMPANY_ALIASES = {
    "600519": ("贵州茅台", "茅台"),
    "000858": ("五粮液",),
    "601318": ("中国平安", "平安"),
    "300750": ("宁德时代", "宁德"),
    "002415": ("海康威视", "海康"),
}


class Encoder:
    def __init__(self, path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        self.model = AutoModel.from_pretrained(path, local_files_only=True).to(self.device).eval()
        self.lock = threading.Lock()

    @torch.inference_mode()
    def encode(self, texts):
        with self.lock:
            tokens = self.tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to(self.device)
            cls = self.model(**tokens).last_hidden_state[:, 0]
            return torch.nn.functional.normalize(cls).cpu().numpy()


def select_rows(root):
    for evidence in load_evidence(root):
        if evidence.kind == "text" or evidence.cells:
            yield evidence


def build(root: Path):
    rows = list(select_rows(root))
    encoder = Encoder(root / "models" / VECTOR_MODEL)
    index = faiss.IndexFlatIP(encoder.model.config.hidden_size)
    for i in range(0, len(rows), 32):
        index.add(encoder.encode([e.text for e in rows[i:i + 32]]))
        if i % 1600 == 0:
            print(f"索引 {index.ntotal}/{len(rows)}", flush=True)

    target = root / "data/evidence_index.zip"
    with NamedTemporaryFile(dir=target.parent, suffix=".tmp", delete=False) as tmp:
        temp = Path(tmp.name)
    try:
        with ZipFile(temp, "w", ZIP_STORED) as archive:
            archive.writestr("index.faiss", faiss.serialize_index(index).tobytes())
            archive.writestr("evidence.json", json.dumps({
                "version": INDEX_VERSION,
                "model": VECTOR_MODEL,
                "rows": [row.model_dump() for row in rows],
            }, ensure_ascii=False).encode())
        os.replace(temp, target)
    finally:
        temp.unlink(missing_ok=True)
    print(f"已索引 {len(rows)} 条证据，其中 {sum(row.kind == 'row' for row in rows)} 条表格行")


def query_scope(question, stock_code=None, year=None):
    companies = {
        code for code, names in COMPANY_ALIASES.items()
        if code in question or any(name in question for name in names)
    }
    years = set(re.findall(r"(20\d{2})\s*年", question))
    if span := re.search(r"(20\d{2})年?\s*(?:至|到|[-–—])\s*(20\d{2})年", question):
        first, last = map(int, span.groups())
        if first <= last <= first + 10:
            years.update(str(y) for y in range(first, last + 1))
    if stock_code:
        if companies and companies != {stock_code}:
            raise ValueError("股票代码过滤与问题中的公司冲突")
        companies = {stock_code}
    if year:
        if years and years != {year}:
            raise ValueError("报告年份过滤与问题中的年份冲突")
        years = {year}
    return companies, years


class Retriever:
    def __init__(self, root):
        with ZipFile(root / "data/evidence_index.zip") as archive:
            saved = json.loads(archive.read("evidence.json"))
            self.index = faiss.deserialize_index(np.frombuffer(archive.read("index.faiss"), dtype=np.uint8))
        if (saved["version"] != INDEX_VERSION or saved["model"] != VECTOR_MODEL
                or self.index.ntotal != len(saved["rows"])):
            raise ValueError("索引格式或元数据不匹配，请重新建库")
        self.rows = [Evidence.model_validate(row) for row in saved["rows"]]
        self.encoder = Encoder(root / "models" / VECTOR_MODEL)
        # Retain the existing BM25 implementation. Apply the same scope on both branches.
        self.bm25 = BM25Okapi([list(jieba.cut(row.text)) for row in self.rows])

    def rank(self, question, companies, years, metrics, *, vector=None, sparse=None):
        indices = np.asarray([
            i for i, e in enumerate(self.rows)
            if (not companies or e.stock_code in companies)
            and (not years or e.report_year in years)
            and (not metrics or (e.metric in metrics and any(
                c.year == e.report_year for c in e.cells
            )))
        ], dtype=np.int64)
        if not len(indices):
            return []
        if vector is None:
            vector = self.encoder.encode([question])
        if sparse is None:
            sparse = self.bm25.get_scores(list(jieba.cut(question)))
        # Filter before dense top-k so a small report is not lost behind other issuers.
        params = faiss.SearchParameters()
        indices = np.ascontiguousarray(indices)
        params.sel = faiss.IDSelectorBatch(indices)
        _, dense = self.index.search(vector, CANDIDATES, params=params)
        lexical = sorted(indices, key=lambda i: (-sparse[i], int(i)))[:CANDIDATES]
        fused = {}
        for ranking in (dense[0], [i for i in lexical if sparse[i] > 0]):
            for rank, i in enumerate(ranking, 1):
                if i >= 0:
                    fused[i] = fused.get(i, 0.0) + 1 / (60 + rank)
        ordered = sorted(fused, key=lambda i: (-fused[i], int(i)))
        return [self.rows[i] for i in ordered[:CANDIDATES]]
