# Document Intelligence Pipeline

A lightweight, framework-free pipeline: raw OpenAI SDK + Pydantic, one file.

```
ingest ─▶ extract ─▶ classify (router) ─▶ summarise (evaluator loop ≤3, pass ≥7) ─▶ emit
```

Documents run in parallel (`asyncio.gather`, semaphore of 5). Each document ends up as `output/<name>.json`.

## Run

```bash
export OPENAI_API_KEY=sk-...

uv run pipeline.py docs/contract.txt      # one document, prints JSON
uv run pipeline.py docs/                  # whole folder in parallel
uv run pipeline.py docs/ --report         # summary table
uv run pipeline.py docs/ --skip-existing  # idempotent re-runs
uv run pipeline.py docs/ --mock           # offline, fake LLM, no key needed
```

`uv` reads the dependencies from the header of `pipeline.py`. Without uv:
`pip install openai pydantic langdetect pypdf && python pipeline.py docs/`.

## Files written

| Path | Contents |
|---|---|
| `output/<name>.json` | `DocumentIntelligence` package |
| `output/<name>.error.json` | error details if that document failed (batch keeps going) |
| `pipeline_costs.jsonl` | one line per document: tokens, USD cost, seconds |

## Configuration

Environment variables: `PIPELINE_MODEL` (default `gpt-4o-mini`), `PRICE_IN_PER_M` and `PRICE_OUT_PER_M`
(USD per 1M tokens, used for the cost log; update them if you change the model).

Constants at the top of `pipeline.py`: `MAX_CONCURRENT`, `MAX_ITERATIONS`, `PASS_SCORE`.

## Notes

- `DocumentIntelligence` fields filled by later steps have defaults, so the object can exist from Step 1.
- Router categories are `legal`, `financial`, `technical`, `general`; each has its own summariser prompt.
- To test the evaluator retry for real, weaken a prompt in `SUMMARISERS` and watch the `[evaluator]` lines on stderr.
