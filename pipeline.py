#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["openai>=1.40", "pydantic>=2", "langdetect", "pypdf"]
# ///
"""Multi-step document processing pipeline.

Raw OpenAI SDK + Pydantic only (no frameworks).

    ingest -> extract -> classify (router) -> summarise (evaluator loop) -> emit

Usage:
    uv run pipeline.py docs/contract.txt
    uv run pipeline.py docs/ --report
    uv run pipeline.py docs/ --mock        # offline, no API key needed
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel

# ----------------------------------------------------------------- config ---
MODEL = os.getenv("PIPELINE_MODEL", "gpt-4o-mini")
PRICE_IN_PER_M = float(os.getenv("PRICE_IN_PER_M", "0.15"))    # USD / 1M input tokens
PRICE_OUT_PER_M = float(os.getenv("PRICE_OUT_PER_M", "0.60"))  # USD / 1M output tokens
MAX_CONCURRENT = 5
MAX_ITERATIONS = 3
PASS_SCORE = 7
MAX_CHARS = 60_000
OUTPUT_DIR = Path("output")
COST_LOG = Path("pipeline_costs.jsonl")
SUPPORTED = {".txt", ".pdf"}
MOCK = False  # set by --mock

DocType = Literal["legal", "financial", "technical", "general"]


# ----------------------------------------------------------------- models ---
class Entities(BaseModel):
    people: list[str]
    organisations: list[str]
    dates: list[str]
    monetary_amounts: list[str]


class DocumentIntelligence(BaseModel):
    """Pipeline state: created by ingest, filled in by each later step."""
    filename: str
    word_count: int
    language: str
    document_type: str = ""
    entities: Entities = Entities(people=[], organisations=[], dates=[], monetary_amounts=[])
    summary: str = ""
    summary_score: int = 0
    key_points: list[str] = []
    risk_flags: list[str] = []
    processing_time_s: float = 0.0


class Classification(BaseModel):  # router output
    document_type: DocType


class Summary(BaseModel):  # generator output
    summary: str
    key_points: list[str]
    risk_flags: list[str]


class Evaluation(BaseModel):  # evaluator output
    score: int
    feedback: str


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls: int = 0

    def add(self, prompt: int, completion: int) -> None:
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        self.calls += 1

    @property
    def cost_usd(self) -> float:
        return (self.prompt_tokens * PRICE_IN_PER_M + self.completion_tokens * PRICE_OUT_PER_M) / 1e6


@dataclass
class Result:
    filename: str
    seconds: float
    intel: Optional[DocumentIntelligence] = None
    error: Optional[str] = None
    skipped: bool = False


# -------------------------------------------------------------------- LLM ---
_client = None


def _get_client():
    global _client
    if _client is None:
        from openai import AsyncOpenAI

        _client = AsyncOpenAI()
    return _client


async def call_llm(system: str, user: str, schema: type[BaseModel], usage: Usage):
    """One structured-output call. Returns a parsed `schema` instance."""
    if MOCK:
        return await _mock_llm(system, user, schema, usage)
    resp = await _get_client().chat.completions.parse(
        model=MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        response_format=schema,
    )
    usage.add(resp.usage.prompt_tokens, resp.usage.completion_tokens)
    msg = resp.choices[0].message
    if msg.parsed is None:
        raise ValueError(f"model returned no structured output: {msg.refusal or 'unknown reason'}")
    return msg.parsed


# ------------------------------------------------------------------ steps ---
def ingest(path: Path) -> tuple[DocumentIntelligence, str]:
    """Step 1 (pure code): read file, count words, detect language."""
    from langdetect import DetectorFactory, detect
    from langdetect.lang_detect_exception import LangDetectException

    DetectorFactory.seed = 0
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        from pypdf import PdfReader

        text = "\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
    elif suffix == ".txt":
        text = path.read_text(encoding="utf-8", errors="replace")
    else:
        raise ValueError(f"unsupported file type: {suffix}")
    text = text.strip()
    if not text:
        raise ValueError("document is empty")
    try:
        language = detect(text)
    except LangDetectException:
        language = "unknown"
    return DocumentIntelligence(filename=path.name, word_count=len(text.split()), language=language), text


async def extract(text: str, usage: Usage) -> Entities:
    """Step 2 (LLM): named entities as structured output."""
    system = (
        "Extract named entities from the document. Return people, organisations, dates, and "
        "monetary amounts exactly as written. Use empty lists when none are present. Do not invent entities."
    )
    return await call_llm(system, text[:MAX_CHARS], Entities, usage)


async def classify(text: str, usage: Usage) -> str:
    """Step 3 (LLM router): pick the document type."""
    system = (
        "Classify the document into exactly one category: "
        "'legal' (contracts, terms, policies, court filings), "
        "'financial' (reports, statements, budgets, forecasts), "
        "'technical' (documentation, runbooks, incident or engineering notes), "
        "'general' (everything else, e.g. emails, memos, articles)."
    )
    result = await call_llm(system, text[:4000], Classification, usage)
    return result.document_type


_SUMMARY_FORMAT = (
    " Return: a concise summary (3-5 sentences), 3-6 key points, and risk_flags "
    "(a list of concrete risks you can point to in the text; empty list if none)."
)
SUMMARISERS: dict[str, str] = {
    "legal": "You are a legal analyst. Summarise obligations, parties, term, termination, liability, "
    "and unusual clauses." + _SUMMARY_FORMAT,
    "financial": "You are a financial analyst. Summarise headline figures, trends, drivers, and outlook; "
    "keep numbers exact." + _SUMMARY_FORMAT,
    "technical": "You are a senior engineer. Summarise the system, the problem or procedure, and the "
    "required actions." + _SUMMARY_FORMAT,
    "general": "You are a careful editor. Summarise the main message, requests, and next steps." + _SUMMARY_FORMAT,
}
EVALUATOR_SYSTEM = (
    "You grade summaries. Score 1-10 for accuracy against the source, coverage of the important points, "
    "and concision. Penalise invented facts heavily. Give one or two sentences of specific, actionable "
    "feedback on what to fix."
)


async def summarise(text: str, doc_type: str, usage: Usage) -> tuple[Summary, int]:
    """Step 4 (specialist LLM + evaluator-optimizer loop, max 3 rounds, pass at >= 7)."""
    system = SUMMARISERS[doc_type]
    best: Optional[tuple[Summary, int]] = None
    feedback = ""
    for attempt in range(1, MAX_ITERATIONS + 1):
        prompt = f"Document:\n{text[:MAX_CHARS]}"
        if feedback:
            prompt += f"\n\nPrevious feedback to address:\n{feedback}"
        draft = await call_llm(system, prompt, Summary, usage)

        points = "\n".join(f"- {p}" for p in draft.key_points)
        review = await call_llm(
            EVALUATOR_SYSTEM,
            f"Source:\n{text[:12_000]}\n\nSummary:\n{draft.summary}\n\nKey points:\n{points}",
            Evaluation,
            usage,
        )
        score = max(1, min(10, review.score))
        print(f"  [evaluator] attempt {attempt}/{MAX_ITERATIONS}: {score}/10", file=sys.stderr)
        if best is None or score > best[1]:
            best = (draft, score)
        if score >= PASS_SCORE:
            break
        feedback = review.feedback
    assert best is not None
    return best


def emit(intel: DocumentIntelligence, out_dir: Path, stem: str) -> Path:
    """Step 5: write the intelligence package."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{stem}.json"
    out.write_text(intel.model_dump_json(indent=2), encoding="utf-8")
    return out


# --------------------------------------------------------------- pipeline ---
def log_cost(filename: str, usage: Usage, seconds: float, status: str) -> None:
    entry = {
        "file": filename,
        "status": status,
        "model": "mock" if MOCK else MODEL,
        "llm_calls": usage.calls,
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "cost_usd": round(usage.cost_usd, 6),
        "seconds": round(seconds, 2),
    }
    with COST_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


async def process_document(path: Path, sem: asyncio.Semaphore, out_dir: Path) -> Result:
    """Run steps 1-5 for one document. Never raises: failures become <stem>.error.json."""
    async with sem:
        start = time.perf_counter()
        usage = Usage()
        try:
            intel, text = ingest(path)
            intel.entities = await extract(text, usage)
            intel.document_type = await classify(text, usage)
            draft, score = await summarise(text, intel.document_type, usage)
            intel.summary = draft.summary
            intel.key_points = draft.key_points
            intel.risk_flags = draft.risk_flags
            intel.summary_score = score
            intel.processing_time_s = round(time.perf_counter() - start, 2)
            emit(intel, out_dir, path.stem)
            log_cost(path.name, usage, intel.processing_time_s, "ok")
            return Result(path.name, intel.processing_time_s, intel=intel)
        except Exception as exc:  # noqa: BLE001 - isolate per-document failures
            elapsed = time.perf_counter() - start
            out_dir.mkdir(parents=True, exist_ok=True)
            err = {"filename": path.name, "error": f"{type(exc).__name__}: {exc}", "seconds": round(elapsed, 2)}
            (out_dir / f"{path.stem}.error.json").write_text(json.dumps(err, indent=2), encoding="utf-8")
            log_cost(path.name, usage, elapsed, "error")
            return Result(path.name, elapsed, error=err["error"])


async def run_batch(paths: list[Path], out_dir: Path, skip_existing: bool) -> list[Result]:
    sem = asyncio.Semaphore(MAX_CONCURRENT)
    todo, results = [], []
    for p in paths:
        if skip_existing and (out_dir / f"{p.stem}.json").exists():
            results.append(Result(p.name, 0.0, skipped=True))
        else:
            todo.append(p)
    results += await asyncio.gather(*(process_document(p, sem, out_dir) for p in todo))
    return results


# ----------------------------------------------------------------- output ---
def print_report(results: list[Result], wall: float) -> None:
    rows = []
    for r in results:
        if r.skipped:
            rows.append((r.filename, "skipped", "-", "-", "-"))
        elif r.intel:
            i = r.intel
            rows.append((i.filename, i.document_type, f"{i.word_count:,}", f"{i.summary_score}/10", f"{r.seconds:.1f}s"))
        else:
            rows.append((r.filename, "ERROR", "-", "-", f"{r.seconds:.1f}s"))
    head = ("File", "Type", "Words", "Score", "Time")
    widths = [max(len(str(x[c])) for x in [head, *rows]) + 2 for c in range(5)]

    def line(l, m, r_):
        return l + m.join("═" * w for w in widths) + r_

    def row(cells):
        out = []
        for c, (cell, w) in enumerate(zip(cells, widths)):
            out.append((" " + str(cell)).ljust(w) if c < 2 else (str(cell) + " ").rjust(w))
        return "║" + "║".join(out) + "║"

    print(line("╔", "╦", "╗"))
    print(row(head))
    print(line("╠", "╬", "╣"))
    for x in rows:
        print(row(x))
    print(line("╚", "╩", "╝"))
    sequential = sum(r.seconds for r in results)
    print(f"Total: {len(results)} docs | {wall:.1f}s parallel (vs ~{sequential:.1f}s sequential)")
    failed = [r for r in results if r.error]
    if failed:
        print(f"{len(failed)} failed - see {OUTPUT_DIR}/*.error.json")


# ------------------------------------------------------------------- mock ---
async def _mock_llm(system: str, user: str, schema: type[BaseModel], usage: Usage):
    """Offline stand-in so the plumbing can be tested without an API key."""
    await asyncio.sleep(0.4)
    usage.add(len(system + user) // 4, 60)
    low = user.lower()
    if schema is Entities:
        return Entities(
            people=["Alice Example"], organisations=["Acme Corp"], dates=["2026-01-01"], monetary_amounts=["$1,000"]
        )
    if schema is Classification:
        keywords = {
            "legal": ("agreement", "hereby", "liability", "clause", "party"),
            "financial": ("revenue", "quarter", "profit", "margin", "forecast"),
            "technical": ("api", "server", "deploy", "latency", "runbook"),
        }
        scores = {k: sum(w in low for w in v) for k, v in keywords.items()}
        best = max(scores, key=scores.get)
        return Classification(document_type=best if scores[best] >= 2 else "general")
    if schema is Summary:
        revised = " [revised]" if "previous feedback" in low else ""
        return Summary(summary=f"Mock summary.{revised}", key_points=["Point A", "Point B"], risk_flags=[])
    if schema is Evaluation:  # first draft fails, the revised one passes
        return Evaluation(score=8 if "[revised]" in low else 6, feedback="Mock: be more specific.")
    raise TypeError(schema)


# -------------------------------------------------------------------- cli ---
def collect(target: Path) -> list[Path]:
    if target.is_dir():
        return sorted(p for p in target.iterdir() if p.suffix.lower() in SUPPORTED)
    return [target]


def main() -> int:
    global MOCK, OUTPUT_DIR
    ap = argparse.ArgumentParser(description="Document intelligence pipeline")
    ap.add_argument("path", type=Path, help="a .txt/.pdf file or a folder of them")
    ap.add_argument("--report", action="store_true", help="print a summary table instead of JSON")
    ap.add_argument("--skip-existing", action="store_true", help="skip files that already have an output JSON")
    ap.add_argument("--out", type=Path, default=OUTPUT_DIR, help="output directory (default: output/)")
    ap.add_argument("--mock", action="store_true", help="use a fake LLM (offline testing)")
    args = ap.parse_args()

    MOCK, OUTPUT_DIR = args.mock, args.out
    if not args.path.exists():
        print(f"error: {args.path} does not exist", file=sys.stderr)
        return 2
    if not MOCK and not os.getenv("OPENAI_API_KEY"):
        print("error: set OPENAI_API_KEY (or pass --mock)", file=sys.stderr)
        return 2
    paths = collect(args.path)
    if not paths:
        print(f"error: no .txt or .pdf files in {args.path}", file=sys.stderr)
        return 2

    start = time.perf_counter()
    results = asyncio.run(run_batch(paths, OUTPUT_DIR, args.skip_existing))
    wall = time.perf_counter() - start

    if args.report:
        print_report(results, wall)
    elif len(results) == 1 and results[0].intel:
        print(results[0].intel.model_dump_json(indent=2))
    else:
        for r in results:
            status = "skipped" if r.skipped else (f"ERROR {r.error}" if r.error else f"ok ({r.seconds:.1f}s)")
            print(f"{r.filename}: {status}")
        print(f"Total: {len(results)} docs in {wall:.1f}s")
    return 1 if any(r.error for r in results) else 0


if __name__ == "__main__":
    sys.exit(main())
