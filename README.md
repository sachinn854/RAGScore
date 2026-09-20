# RAGScore

A from-scratch **RAG evaluation harness**. Give it a document corpus and a set of
questions with ground-truth answers; it scores a RAG pipeline's answers on:

- **Retrieval:** precision@k, recall@k, hit@k, MRR
- **Faithfulness:** are the answer's claims supported by the retrieved context?
- **Answer relevance:** does the answer address the question?
- **Answer correctness:** how close is the answer to the ideal answer?

Built without LangChain / LlamaIndex — every piece (chunking, retrieval, prompt
building, the judge loop, the metrics) is hand-written to keep the evaluation logic
transparent. See [`architecture.md`](architecture.md) for the full data flow and
file layout.

## Setup
```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
copy .env.example .env
```

Fill in `.env`:
- `OPENROUTER_API_KEY` — for answer generation + the judge ([openrouter.ai](https://openrouter.ai))
- `GEMINI_API_KEY` — for embeddings, free tier, no card ([aistudio.google.com](https://aistudio.google.com) → Get API key)

## Usage
```bash
# 1. build the vector index from data/corpus/
python -m ragscore.ingest

# 2. ask one question (sanity check)
python -m ragscore.rag -q "What is d_k in the base Transformer model?"

# 3. run the full eval set (data/eval_set.jsonl) through every metric
python -m ragscore.runner

# 4. print a terminal report from the latest run
python -m ragscore.report

# 5. ablation: sweep chunk_size and top_k, compare
python -m ragscore.experiments
```

Every module is standalone runnable with `--help` (e.g. `python -m ragscore.ingest --help`)
and every knob (`--chunk-size`, `--top-k`, `--gen-model`, ...) can be overridden from the CLI.

## Results

Corpus: 7 short articles on the Transformer architecture (~4,000 words).
Eval set: 20 hand-written questions — 15 single-doc, 3 multi-hop, 2 unanswerable
(including one adversarial near-miss designed to tempt hallucination).

### Baseline (chunk_size=512, chunk_overlap=64, top_k=5)

| metric | score |
|---|---|
| precision@k | 0.760 |
| recall@k | 0.900 |
| hit@k | 0.900 |
| MRR | 0.875 |
| faithfulness | 0.950 |
| answer_relevance | 1.000 |
| answer_correctness | 0.898 |

Full 20-question run: **150s, $0.0065**.

### Experiment 1 — chunk_size: 256 vs 512

| chunk_size | precision@k | faithfulness | answer_correctness |
|---|---|---|---|
| 256 | **0.780** | 0.990 | 0.832 |
| 512 | 0.760 | 0.950 | **0.898** |

Smaller chunks (256) retrieve more precisely — less irrelevant text per chunk — but
**correctness drops** (0.832 vs 0.898): each chunk carries less surrounding context,
so the generator has less to work with even when it retrieves the right chunk.
Precision and answer quality pulled in opposite directions here.

### Experiment 2 — chunk_size: 128 vs 512

| chunk_size | faithfulness | answer_correctness |
|---|---|---|
| 128 | **0.990** | 0.858 |
| 512 | 0.950 | **0.898** |

Counterintuitive: the *smallest* chunks got the *highest* faithfulness. With very
small, focused chunks the model has less room to over-claim beyond what a chunk
literally says — but again at a cost to overall correctness, for the same reason
as above.

### Experiment 3 — top_k: 3 vs 8

| top_k | precision@k | recall@k | hit@k | MRR |
|---|---|---|---|---|
| 3 | **0.817** | 0.900 | 0.900 | 0.875 |
| 8 | 0.613 | 0.900 | 0.900 | 0.875 |

Classic precision/recall trade-off: precision falls sharply as `top_k` grows
(more slots, same number of relevant docs, more diluted), while recall, hit@k,
and MRR stay completely flat. The right document was already found within the
top 3 for nearly every question — pulling more chunks only added noise, it never
helped find something that wasn't already there.

**Takeaway:** for this corpus, `chunk_size=512, top_k=5` (the default) is the most
balanced setting — not the best on any single metric, but the only one that doesn't
trade away answer correctness for a retrieval-metric win.

## What I learned

- **Retrieval and generation fail independently.** The clearest example: a question
  asking for a fact stated only for a *different* language pair than asked. Retrieval
  correctly found the relevant chunk (score 0.76), but generation hallucinated by
  applying the nearby fact anyway — faithfulness and answer_correctness both caught
  it (0.00 and 0.29) even though retrieval looked fine. One metric alone would have
  hidden the failure.
- **LLM-as-judge needs a narrow rubric.** Splitting "does the answer address the
  question" (relevance) from "are the claims true" (faithfulness) from "does it match
  the reference" (correctness) mattered — a single vague "is this a good answer?"
  prompt would have blurred exactly the failure modes above.
- **Caching isn't optional at this scale.** Every metric call is an LLM call; a full
  20-question run makes ~120 of them. Content-hash caching made iterating on prompts
  and re-running experiments nearly free after the first pass.
- **Free-tier rate limits are a real constraint, not an edge case.** Gemini's
  embeddings free tier caps requests per minute; a burst re-ingest (during the
  chunk_size ablation) needed patient exponential backoff, not just a couple of
  retries, to reliably get through.
