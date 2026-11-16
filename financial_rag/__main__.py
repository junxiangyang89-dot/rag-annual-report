import argparse
import json
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
TITLE = "面向复杂财务报表的混合检索与高精度引用溯源系统"


def new_pipeline(http, rerank_http):
    from .pipeline import Pipeline
    from .providers import DeepSeek, QwenReranker
    from .retrieve import Retriever

    return Pipeline(Retriever(ROOT), QwenReranker(rerank_http), DeepSeek(http))


@asynccontextmanager
async def lifespan(app):
    with httpx.Client(timeout=40) as http, httpx.Client(timeout=20, trust_env=False) as rerank_http:
        app.state.pipeline = new_pipeline(http, rerank_http)
        yield


app = FastAPI(title=TITLE, lifespan=lifespan)


class Question(BaseModel):
    question: str = Field(min_length=2, max_length=400)
    stock_code: str | None = Field(default=None, pattern=r"^\d{6}$")
    year: str | None = Field(default=None, pattern=r"^20\d{2}$")


@app.get("/", response_class=HTMLResponse)
def home():
    return (ROOT / "static/index.html").read_text(encoding="utf-8")


@app.get("/health")
def health():
    return {"status": "ok", "pipeline_ready": True}


def run_query(request: Question, debug: bool):
    from .providers import ProviderError

    try:
        return app.state.pipeline.ask(**request.model_dump(), debug=debug)
    except ProviderError as error:
        raise HTTPException(502, str(error)) from None
    except ValueError as error:
        raise HTTPException(422, str(error)) from None


@app.post("/query")
def ask(request: Question):
    return run_query(request, debug=False)


@app.post("/query/debug")
def debug_query(request: Question):
    return run_query(request, debug=True)


def main():
    parser = argparse.ArgumentParser(description=TITLE)
    modes = parser.add_subparsers(dest="mode", required=True)
    modes.add_parser("build")
    web = modes.add_parser("web")
    web.add_argument("--port", type=int, default=8014)
    cli = modes.add_parser("query")
    cli.add_argument("question")
    cli.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    if args.mode == "build":
        from .retrieve import build

        build(ROOT)
    elif args.mode == "web":
        import uvicorn

        uvicorn.run(app, host="127.0.0.1", port=args.port)
    else:
        from .providers import ProviderError

        with httpx.Client(timeout=40) as http, httpx.Client(timeout=20, trust_env=False) as rerank_http:
            try:
                answer = new_pipeline(http, rerank_http).ask(args.question, debug=args.debug)
            except (ProviderError, ValueError) as error:
                parser.exit(1, f"{error}\n")
            print(json.dumps(answer, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
