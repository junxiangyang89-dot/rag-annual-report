"""Run the existing RAG entry points with DeepSeek and local BGE embeddings."""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile
from types import SimpleNamespace

import faiss
import torch
from langchain_core.embeddings import Embeddings
from openai import APIConnectionError, APIStatusError, OpenAI
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parent
BASE_URL = "https://api.deepseek.com"
CHAT_MODEL = "deepseek-flash"
CHAT_OPTIONS = {"thinking": {"type": "disabled"}}
BGE_MODEL = "bge-small-zh-v1.5"
CHUNKS_PATH = Path("data/chunks/all_semantic.json")
BGE_INDEX_DIR = Path("vectorstore/faiss_bge_small_zh_v1_5")


def read_key():
    key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not key:
        raise SystemExit("请设置 DEEPSEEK_API_KEY")
    return key


class BGEEmbeddings(Embeddings):
    def __init__(self):
        model_path = ROOT / "models" / BGE_MODEL
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModel.from_pretrained(model_path, local_files_only=True)
        self.model.to(self.device).eval()
        self.dimension = self.model.config.hidden_size

    @torch.inference_mode()
    def encode(self, texts):
        tokens = self.tokenizer(
            [text.replace("\n", " ") for text in texts],
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(self.device)
        cls = self.model(**tokens).last_hidden_state[:, 0]
        return torch.nn.functional.normalize(cls).cpu().numpy()

    def embed_documents(self, texts):
        return self.encode(texts).tolist() if texts else []

    def embed_query(self, text):
        return self.embed_documents([text])[0]


def write_atomic(path, data):
    with NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as output:
        temporary_path = Path(output.name)
        try:
            output.write(data)
            output.close()
            temporary_path.replace(path)
        finally:
            output.close()
            temporary_path.unlink(missing_ok=True)


def build_index(embeddings, chunk_data, index_path, metadata_path):
    from src.build_index import faiss_metadata

    records = faiss_metadata(json.loads(chunk_data))
    if not records:
        raise SystemExit("分块文件为空，请检查年报解析和分块结果")
    index = faiss.IndexFlatIP(embeddings.dimension)
    batch_size = 32
    print(f"使用 {embeddings.device} 编码 {len(records)} 条年报分块", flush=True)
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        index.add(embeddings.encode([record["content"] for record in batch]))
        if start % (batch_size * 50) == 0:
            print(f"已编码 {index.ntotal}/{len(records)}", flush=True)

    index_path.parent.mkdir(parents=True, exist_ok=True)
    write_atomic(metadata_path, json.dumps(records, ensure_ascii=False).encode("utf-8"))
    write_atomic(index_path, faiss.serialize_index(index).tobytes())
    print(f"BGE 索引已写入 {index_path}，共 {index.ntotal} 条向量")


def configure_native(module, embeddings, key, index_path, metadata_path):
    api = OpenAI(api_key=key, base_url=BASE_URL, timeout=60.0, max_retries=1)

    def complete(*, model, **request):
        # The original query rewriter supplies its own hard-coded model name.
        try:
            return api.chat.completions.create(
                model=CHAT_MODEL, extra_body=CHAT_OPTIONS, **request
            )
        except APIStatusError as error:
            raise RuntimeError(f"DeepSeek 请求失败（HTTP {error.status_code}）") from None
        except APIConnectionError:
            raise RuntimeError("无法连接 DeepSeek 官方接口") from None

    def embed(*, model, input, dimensions):
        vectors = embeddings.embed_documents(input)
        return SimpleNamespace(data=[SimpleNamespace(embedding=vector) for vector in vectors])

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=complete)),
        embeddings=SimpleNamespace(create=embed),
    )
    module.get_client = lambda: client
    module.LLM_MODEL = CHAT_MODEL
    module.EMBED_MODEL = BGE_MODEL
    module.EMBED_DIM = embeddings.dimension
    module.INDEX_PATH = index_path
    module.META_PATH = metadata_path


def configure_langchain(module, embeddings, key):
    from langchain_openai import ChatOpenAI

    module.LLM_MODEL = CHAT_MODEL
    module.VECTORSTORE_DIR = Path("vectorstore/faiss_lc")
    module.get_embeddings = lambda: embeddings
    module.get_llm = lambda: ChatOpenAI(
        model=CHAT_MODEL,
        api_key=key,
        base_url=BASE_URL,
        extra_body=CHAT_OPTIONS,
        temperature=0.1,
        timeout=60.0,
        max_retries=1,
    )


def configure_web(pipeline_module):
    from fastapi.responses import HTMLResponse
    from src import serve

    serve._load_pipeline_module = lambda: pipeline_module
    serve.app.description = f"FAISS + BM25 混合检索，使用本地 BGE 和 {CHAT_MODEL}"

    # Use the existing UI with labels that match the configured providers.
    html = (serve._STATIC_DIR / "index.html").read_text(encoding="utf-8")
    html = html.replace("text-embedding-v3", BGE_MODEL).replace("DashScope qwen-plus", CHAT_MODEL)
    routes = serve.app.router.routes
    routes.remove(next(route for route in routes if route.path == "/"))
    serve.app.add_api_route("/", lambda: HTMLResponse(html), include_in_schema=False)
    return serve.app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["build-index", "web", "native", "langchain"])
    parser.add_argument("--port", type=int, default=8000, help="本地网页端口")
    args, forwarded = parser.parse_known_args()
    if forwarded and args.mode in {"build-index", "web"}:
        parser.error(f"无法识别的参数 {' '.join(forwarded)}")

    os.chdir(ROOT)
    key = read_key() if args.mode != "build-index" else None
    if args.mode != "langchain":
        if not CHUNKS_PATH.is_file():
            raise SystemExit(f"缺少 {CHUNKS_PATH}，请先运行 src/chunk_documents.py")
        chunk_data = CHUNKS_PATH.read_bytes()
        digest = hashlib.sha256(chunk_data).hexdigest()
        index_path = BGE_INDEX_DIR / f"{digest}.bin"
        metadata_path = index_path.with_suffix(".json")
        if args.mode != "build-index" and not (index_path.is_file() and metadata_path.is_file()):
            raise SystemExit("请先运行 run_deepseek.py build-index")
    embeddings = BGEEmbeddings()
    sys.argv[1:] = forwarded

    if args.mode == "langchain":
        from src_langchain import rag_chain_lc

        configure_langchain(rag_chain_lc, embeddings, key)
        rag_chain_lc.main()
        return

    if args.mode == "build-index":
        build_index(embeddings, chunk_data, index_path, metadata_path)
        return

    from src import rag_pipeline

    configure_native(rag_pipeline, embeddings, key, index_path, metadata_path)
    if args.mode == "native":
        if "--no-rerank" not in forwarded:
            sys.argv.append("--no-rerank")
        rag_pipeline.main()
        return

    import uvicorn

    uvicorn.run(configure_web(rag_pipeline), host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    try:
        main()
    except APIStatusError as error:
        raise SystemExit(f"DeepSeek 请求失败（HTTP {error.status_code}）") from None
    except APIConnectionError:
        raise SystemExit("无法连接 DeepSeek 官方接口") from None
