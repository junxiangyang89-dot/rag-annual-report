# 上市公司年报问答

用 A 股上市公司年报做问答，回答附公司、年份和页码。默认语料是贵州茅台、五粮液、中国平安、宁德时代和海康威视的 2021–2023 年年报。

原生版用 FAISS 和 BM25 混合检索，再用 RRF 合并结果。LangChain 版只用 FAISS，便于对照。两版都可以用本地 BGE 编码，由 DeepSeek 生成答案。

## 本地运行

使用 Python 3.12，在仓库根目录依次运行以下命令。脚本会下载公开年报和 BGE 权重，并在本地生成索引。Windows 上建议使用纯英文目录。

```bash
python -m pip install -r requirements.txt
python src/download_reports.py
python src/parse_pdf.py
python src/chunk_documents.py
python src_langchain/download_model.py
python run_deepseek.py build-index
```

设置环境变量 `DEEPSEEK_API_KEY`，运行 `python run_deepseek.py web`，打开 [本地页面](http://127.0.0.1:8000/)。页面可查看检索、融合、引用和回答。

也可以把已有年报按 `股票代码_年份_标题.pdf` 命名，放进 `data/raw_pdf/`，从解析步骤开始。

运行 LangChain 版时，先执行 `python src_langchain/build_index_lc.py`，再执行 `python run_deepseek.py langchain`。

## 评估

历史 RAGAS 对比使用同一套 20 道题，其中 17 道进入评分。原生版忠实度为 0.809，LangChain 版为 0.601。完整结果和逐题分析见 [评估报告](report/RAGAS评估报告.md)。

源码使用 [MIT 许可证](LICENSE)。年报来自[巨潮资讯网](https://www.cninfo.com.cn/)，模型来自 [BAAI](https://huggingface.co/BAAI/bge-small-zh-v1.5)。
