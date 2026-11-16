# 面向复杂财务报表的混合检索与高精度引用溯源系统

用 5 家 A 股公司 2021–2023 年的 15 份年报做问答。财务表格按科目和年份列入库，答案附原始值、单位及 PDF 页码。

- pdfplumber 对齐多列表格的横坐标，对符合条件的续页继承表头和单位。
- 本地 BGE-small + FAISS 与 BM25 双路召回，经 RRF 融合后取最多 10 个候选，交给阿里云 `qwen3.7-text-rerank` 重排。
- DeepSeek 用 JSON Mode 选择证据，Pydantic 和后端逐项校验公司、年份、科目、页码、数值与单位。缺少关键证据时返回证据不足。

## 本地运行

使用 Python 3.12。在仓库根目录运行以下命令。年报按[语料清单](data/manifest.json)从巨潮资讯网下载，BGE 权重从 Hugging Face 下载，索引在本地生成。

```bash
python -m pip install -r requirements.txt
python -m financial_rag.download_corpus
python src/parse_pdf.py
python src/chunk_documents.py
python -c "from huggingface_hub import snapshot_download; snapshot_download('BAAI/bge-small-zh-v1.5', local_dir='models/bge-small-zh-v1.5', allow_patterns=['*.json', 'model.safetensors', 'vocab.txt'])"
python -m financial_rag build
```

如下载失败，可按清单将 PDF 保存到 `data/raw_pdf/`，文件名与清单一致。

启动前设置以下环境变量。请从各自的服务后台获取 API 密钥和接入地址。

| 变量 | 用途 |
| --- | --- |
| `DEEPSEEK_API_KEY` | DeepSeek API 密钥 |
| `QWEN_RERANK_API_KEY` | 阿里云重排 API 密钥 |
| `QWEN_RERANK_MODEL` | 阿里云账号可用的重排模型，回归使用 `qwen3.7-text-rerank` |
| `QWEN_RERANK_BASE_URL` | 阿里云重排服务的 HTTPS 接入地址，代码会补全 `/reranks` |

重排的三项配置也可放在仓库外的文本文件中（`apikey`、`model`、`openai兼容`，每行以冒号分隔键和值），通过 `QWEN_RERANK_CONFIG` 指定路径。`DEEPSEEK_MODEL` 可选，默认 `deepseek-flash`。

```bash
python -m financial_rag web
```

打开 [本地问答](http://127.0.0.1:8014/) 或 [API 文档](http://127.0.0.1:8014/docs)。也可以在终端试问：

```bash
python -m financial_rag query "五粮液2022年归母净利润是多少亿元？" --debug
```

## 回归与历史版本

20 道回归题中，22 个黄金数值全部通过数值和引用页码校验，3 道预期拒答题全部正确拒答。17 道其余题中有 14 道返回带引用的答复。详见[回归报告](report/回归报告.md)和[黄金标注](evaluation/gold.json)。旧版 [RAGAS 报告](report/RAGAS评估报告.md)为历史基线，不能与此次回归直接比较。

```bash
python -B -m unittest discover -s tests -v
python -m evaluation.regression --check --out data/retrieval.json
python -m evaluation.regression --online --check --out data/regression.json
```

单元测试无需索引或 API 密钥。后两条命令需先建库，`--online` 还需上述 API 配置。旧版使用仓库的 `main` 分支。源码使用 [MIT 许可证](LICENSE)。
