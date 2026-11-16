import json
import re
from pathlib import Path

import pdfplumber

from .evidence import Cell, Evidence, NUMBER, UNIT, compact, metric_of

TITLE = re.compile(r"^(?:[一二三四五六七八九十\d]+[、.．）)])?\s*((?:合并|母公司)?(?:资产负债表|利润表|现金流量表|所有者权益变动表)|(?:近三年)?主要会计数据[和及]财务指标|研发投入|费用)(?:[（(]续[）)])?$")
SECTION = re.compile(r"^(第[一二三四五六七八九十]+节)[\s　]*(.{2,24})$")
UNIT_LINE = re.compile(rf"单位[：:](?:人民币)?({UNIT})")
HEADER = re.compile(r"^(?:(20\d{2})年(?:度)?|本期金额|上期金额|本年金额|上年金额|本年发生额|上年发生额|本报告期|上年同期|本年|上年|上上年)(?:[（(]已重述[）)])?$")
NO_CONTINUATION = re.compile(r"分季度|第一季度|第二季度|第三季度|第四季度|^\d+[、.]")
RAW_WORD = re.compile(r"^[\d,.−－()+%-]+$")


def year_of(header, report_year):
    match = HEADER.fullmatch(header)
    if not match:
        return None
    if match[1]:
        return match[1]
    if header == "上上年":
        return str(int(report_year) - 2)
    return str(int(report_year) - int(header.startswith(("上期", "上年"))))


def positioned_rows(table):
    return [
        [(compact(value), bbox) for value, bbox in zip(values, row.cells) if value and compact(value) and bbox]
        for values, row in zip(table.extract(), table.rows)
    ]


def compatible(left, right):
    return all(abs(a - b) <= 5 for a, b in zip(left, right))


def parse_rows(table, report_year, carried, table_unit):
    rows = positioned_rows(table)
    header_at = next((i for i, row in enumerate(rows[:9])
                      if any(HEADER.fullmatch(value) for value, _ in row)
                      and not any(NUMBER.fullmatch(value) for value, _ in row)), None)
    if header_at is not None:
        header = [(value, year_of(value, report_year), bbox) for value, bbox in rows[header_at]]
        columns = [(name, year, bbox) for name, year, bbox in header if year]
        first = header_at + 1
    else:
        columns = carried
        first = 0
        if any(NO_CONTINUATION.search(value) for row in rows[:1] for value, _ in row):
            columns = []

    for row_index, row in enumerate(rows[first:], start=first):
        if not row or NUMBER.fullmatch(row[0][0]) or HEADER.fullmatch(row[0][0]):
            continue
        label = row[0][0]
        row_unit = re.search(rf"[（(](?:人民币)?({UNIT})[）)]$", label)
        unit = (row_unit[1] if row_unit else table_unit).replace("／", "/")
        cells = []
        numeric = [(value, box) for value, box in row[1:] if NUMBER.fullmatch(value)]
        for column, year, box in columns:
            center = (box[0] + box[2]) / 2
            matches = [(value, rect) for value, rect in numeric if rect[0] <= center < rect[2]]
            if len(matches) == 1:
                value, rect = matches[0]
                hits = sum(rect[0] <= (h[0] + h[2]) / 2 < rect[2] for _, _, h in columns)
                cell_unit = "%" if value.endswith("%") else unit
                if hits == 1 and cell_unit:
                    cells.append(Cell(column=column, year=year, raw=value, unit=cell_unit))
        if len({c.year for c in cells}) != len(cells):
            cells = []
        display = []
        for value, rect in row[1:]:
            shared = [name for name, _, box in columns if rect[0] <= (box[0] + box[2]) / 2 < rect[2]]
            display.append(f"{shared[0]} {value}" if len(shared) == 1 else value)
        if display:
            yield row_index, label, display, cells, columns
    # Keep observed headers even when the table ends before the first data row.
    yield -1, "", [], [], columns


def positioned_statement_rows(page, start, end, report_year, unit):
    """Read borderless statements using the x coordinates of printed year anchors."""
    headers = page.crop((0, start, page.width, end)).extract_text_lines()
    for i, line in enumerate(headers[:12]):
        years = [(year_of(w["text"], report_year), w) for w in page.crop(
            (0, line["top"], page.width, line["bottom"])
        ).extract_words()]
        years = [(year, w) for year, w in years if year]
        if len(years) != 2 or not ("附注" in line["text"] or "项目" in line["text"]):
            continue
        for row_index, row in enumerate(headers[i+1:], start=i+1):
            words = page.crop((0, row["top"], page.width, row["bottom"])).extract_words()
            left = words[0]["x0"] if words else 0
            label = compact("".join(w["text"] for w in words if w["x0"] < left + 170 and not RAW_WORD.fullmatch(w["text"])))
            metric = metric_of(label)
            if not metric:
                continue
            numbers = [w for w in words if NUMBER.fullmatch(w["text"]) and w["x0"] > left + 150]
            if len(numbers) < 2:
                continue
            numbers = numbers[-2:]
            if any(abs((word["x0"]+word["x1"])/2 - (head["x0"]+head["x1"])/2) > 42
                   for word, (_, head) in zip(numbers, years)):
                continue
            cells = [Cell(column=head["text"], year=year, raw=word["text"], unit=unit)
                     for word, (year, head) in zip(numbers, years)]
            yield row_index, label, metric, cells
        return


def extract_report(path, meta):
    section = ""
    table_title = ""
    table_unit = ""
    carried = []
    previous_bbox = None
    previous_page = 0
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            lines = page.extract_text_lines()
            tables = page.find_tables()
            for line in lines:
                if match := SECTION.fullmatch(line["text"].strip()):
                    if not re.search(r"\.{2,}|…|\d+$", match[2]):
                        if line["top"] < page.height * .45:
                            section = " ".join(match.groups())
            statement = next((i for i, line in enumerate(lines) if compact(line["text"]) == "合并利润表"), None)
            if not tables and statement is not None:
                heads = [compact(line["text"]) for line in lines[statement:statement+5]]
                if any("人民币百万元" in head for head in heads):
                    location = "财务报告 > 合并利润表"
                    start = lines[statement]["top"]
                    for row_index, label, metric, cells in positioned_statement_rows(page, start, page.height, meta["year"], "百万元"):
                        yield Evidence(
                            id=f"{meta['stock_code']}_{meta['year']}_p{page.page_number}_borderless_r{row_index}",
                            stock_code=meta["stock_code"], company=meta["company_name"], report_year=meta["year"],
                            source=path.name, page=page.page_number, section=location, kind="row",
                            text=f"[{meta['company_name']} {meta['year']}年年报 · {location}] {label}：" +
                            "；".join(f"{cell.column} {cell.raw}{cell.unit}" for cell in cells),
                            row_label=label, metric=metric, cells=cells,
                        )
            used_line = 0
            for table_index, table in enumerate(tables):
                above = []
                while used_line < len(lines) and lines[used_line]["bottom"] <= table.bbox[1] + 1:
                    above.append(lines[used_line]["text"])
                    used_line += 1
                new_title = False
                for text in above:
                    normalized = compact(text)
                    if match := SECTION.fullmatch(text.strip()):
                        section = " ".join(match.groups())
                    if match := TITLE.fullmatch(normalized):
                        table_title = match[1]
                        table_unit, carried = "", []
                        new_title = True
                    if match := UNIT_LINE.search(normalized):
                        table_unit = match[1]
                bbox = (table.bbox[0], table.bbox[2])
                can_carry = (
                    previous_bbox is not None and compatible(previous_bbox, bbox)
                    and previous_page == page.page_number - 1 and table_index == 0
                    and table.bbox[1] < page.height * 0.25 and not new_title
                    and not any(NO_CONTINUATION.search(compact(s)) for s in above)
                )
                if not new_title and not can_carry and not any(UNIT_LINE.search(compact(s)) for s in above):
                    if not (previous_page == page.page_number and table_title == "研发投入"):
                        table_title, table_unit = "", ""
                inherited = carried if can_carry else []
                # Some report tables put the summary heading above a short disclaimer.
                if not table_title:
                    recent = [compact(line) for line in above[-8:]]
                    table_title = next((m[1] for line in reversed(recent) if (m := TITLE.fullmatch(line))), "")
                if table_title.startswith("合并"):
                    location = "财务报告 > " + table_title
                else:
                    location = " > ".join(part for part in (section, table_title) if part) or "年报表格"
                for row_index, label, values, cells, columns in parse_rows(table, meta["year"], inherited, table_unit):
                    carried = columns
                    if row_index == -1:
                        continue
                    qualified = [
                        f"{cell.year}年（{cell.column}）：{cell.raw}"
                        + (cell.unit if not cell.raw.endswith("%") else "")
                        for cell in cells
                    ]
                    text = f"[{meta['company_name']} {meta['year']}年年报 · {location}] {label}：" + "；".join(qualified or values)
                    # Parent-only statements cannot support consolidated metrics.
                    trusted = (
                        table_title == "合并利润表"
                        or "主要会计数据" in table_title
                        or (table_title == "费用" and metric_of(label) == "rd_expense")
                        or (table_title == "研发投入" and metric_of(label) == "rd_investment")
                    )
                    yield Evidence(
                        id=f"{meta['stock_code']}_{meta['year']}_p{page.page_number}_t{table_index}_r{row_index}",
                        stock_code=meta["stock_code"], company=meta["company_name"], report_year=meta["year"],
                        source=path.name, page=page.page_number, section=location, kind="row",
                        text=text, row_label=label, metric=metric_of(label) if trusted else None,
                        cells=cells if trusted else [],
                    )
                previous_bbox, previous_page = bbox, page.page_number


def load_evidence(root: Path):
    manifest = json.loads((root / "data/manifest.json").read_text(encoding="utf-8"))
    reports = {(m["stock_code"], m["year"]): m for m in manifest}
    chunks = root / "data/legacy_chunks.json"
    if not chunks.is_file():
        chunks = root / "data/chunks/all_semantic.json"
    for chunk in json.loads(chunks.read_text(encoding="utf-8")):
        m = chunk["metadata"]
        report = reports[(m["stock_code"], m["year"])]
        if m["page_num"] < 1:
            continue
        yield Evidence(
            id=chunk["chunk_id"], stock_code=m["stock_code"], company=report["company_name"],
            report_year=m["year"], source=report["filename"], page=m["page_num"],
            section=m["section"] or "年报正文", kind="text", text=chunk["content"],
        )
    for meta in manifest:
        yield from extract_report(root / "data/raw_pdf" / meta["filename"], meta)
