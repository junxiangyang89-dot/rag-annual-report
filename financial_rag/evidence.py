import re
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

YEAR = re.compile(r"(20\d{2})\s*(?:年(?:度)?)?")
UNIT = r"亿元|百万元|千元|万元|元[／/]股|元|人民币百万元|%"
NUMBER = re.compile(r"[-−－]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?%?")
SCALES = {"元": 1, "千元": 1000, "万元": 10000, "百万元": 1000000, "亿元": 100000000}
METRICS = {
    "parent_profit": r"(?:归属(?:于)?(?:上市公司|母公司|本公司)(?:的)?(?:普通股)?(?:股东|所有者|权益持有人)(?:的)?净利润|归母净利润)",
    "total_revenue": r"营业总收入",
    "revenue": r"营业收入",
    "basic_eps": r"基本每股收益",
    "rd_expense": r"研发费用",
    "rd_investment": r"研发投入(?:金额|合计)?",
    "insurance_revenue": r"保险业务收入(?:总额)?",
    "gross_margin": r"(?:销售)?毛利率",
}


def compact(text: str) -> str:
    return re.sub(r"\s+", "", text)


def raw_label(text: str) -> str:
    label = re.sub(r"^[一二三四五六七八九十\d]+[、.．）)]\s*|^(?:其中|加|减)[：:]", "", compact(text))
    label = re.sub(r"[（(]净亏损以[^）)]*号填列[）)]$", "", label)
    return re.sub(rf"[（(](?:人民币)?(?:{UNIT})[）)]$", "", label)


def metric_of(label: str) -> str | None:
    key = raw_label(label)
    return next((name for name, pattern in METRICS.items() if re.fullmatch(pattern, key)), None)


def requested_metrics(question: str) -> list[str]:
    text = compact(question)
    if re.search(r"扣非|扣除非经常性损益", text):
        return []
    found = [metric for metric, pattern in METRICS.items() if re.search(pattern, text)]
    if "营收" in text and not {"revenue", "total_revenue"}.intersection(found):
        found.append("revenue")
    if ("研发费用率" in text or "占当年营业收入的比例" in text) and "revenue" not in found:
        found.append("revenue")
    return found


def ambiguous_request(question: str) -> bool:
    return bool(re.search(r"扣非|扣除非经常性损益|母公司单体|母公司利润表", compact(question)))


class LockedModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class Cell(LockedModel):
    column: str
    year: str = Field(pattern=r"^20\d{2}$")
    raw: str
    unit: str

    def normalized(self) -> Decimal:
        return Decimal(self.raw.replace(",", "").replace("−", "-").replace("－", "-").rstrip("%")) * SCALES.get(self.unit, 1)


class Evidence(LockedModel):
    id: str
    stock_code: str
    company: str
    report_year: str
    source: str
    page: int = Field(ge=1)
    section: str
    text: str
    kind: Literal["row", "text"]
    row_label: str = ""
    metric: str | None = None
    cells: list[Cell] = Field(default_factory=list)


class Claim(LockedModel):
    evidence_id: str
    section: str
    page: int
    row_label: str
    column: str
    year: str
    raw: str
    unit: str


class Selection(LockedModel):
    claims: list[Claim] = Field(max_length=16)
    quotes: list["SelectedQuote"] = Field(default_factory=list, max_length=8)


class SelectedQuote(LockedModel):
    evidence_id: str
    excerpt: str = Field(min_length=12, max_length=500)


Selection.model_rebuild()


def cite(evidence: Evidence, cell: Cell) -> Claim:
    return Claim(
        evidence_id=evidence.id, section=evidence.section, page=evidence.page,
        row_label=evidence.row_label, column=cell.column, year=cell.year, raw=cell.raw, unit=cell.unit,
    )


def validate_claim(claim: Claim, candidates: dict[str, Evidence], metrics: list[str]) -> tuple[Evidence, Cell]:
    evidence = candidates.get(claim.evidence_id)
    if (evidence is None or evidence.kind != "row" or evidence.metric not in metrics
            or metric_of(evidence.row_label) != evidence.metric
            or (evidence.section, evidence.page, evidence.row_label) != (claim.section, claim.page, claim.row_label)):
        raise ValueError("章节、页码或财务科目与证据不符")
    units = {"basic_eps": {"元/股"}, "gross_margin": {"%"}}.get(evidence.metric, set(SCALES))
    matches = [cell for cell in evidence.cells if cite(evidence, cell) == claim and cell.unit in units]
    if len(matches) != 1:
        raise ValueError("列名、年份、单位或数值与证据不符")
    return evidence, matches[0]
