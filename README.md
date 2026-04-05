# hp-rag: Harry Potter RAG Experiment

Local LLM (Nemotron 9B) + SQLite-based RAG for answering multi-hop questions about Harry Potter and the Philosopher's Stone.

## The Problem

A simple question like **"What gift did Dumbledore give Harry for Christmas?"** is deceptively hard for RAG systems. The answer (the invisibility cloak) requires evidence scattered across multiple passages:

| Chunk | Chapter | Evidence | Key challenge |
|-------|---------|----------|---------------|
| 260 | 12 | Harry unwraps "something fluid and silvery gray" — Ron identifies it as an invisibility cloak | No mention of "Dumbledore", "Christmas", or "gift" |
| 261 | 12 | Anonymous note: "Your father left this in my possession before he died" | Sender unknown |
| 391 | 17 | Ron: "D'you think he meant you to do it? Sending you your father's cloak and everything?" | Finally connects cloak → Dumbledore, but no "Christmas" or "gift" |

No single chunk contains all three concepts (Dumbledore + Christmas + gift/cloak). Standard FTS or keyword search fails because the query terms and the answer terms have almost zero lexical overlap.

## What didn't work

| Method | Chunk 260 | Chunk 261 | Chunk 391 | Problem |
|--------|-----------|-----------|-----------|---------|
| FTS (BM25) | miss | miss | miss | Query terms don't co-occur with answer terms |
| Character trigrams | rank 315 | rank 12 | rank 10 | 260 has no query-term overlap |
| Co-occurrence graph hop | rank 90 | rank 58 | rank 35 | Generic nouns (feet, bed) dominate bridges |
| IDF-weighted bridge concepts | — | — | — | "cloak" at rank 73; character names drown signal |

## What worked: Trigram + Window Expansion

A two-step approach with **zero DB modification**:

### Step 1: Character trigram retrieval (top-15)

Break the question into character trigrams, score all chunks by overlap ratio. This retrieves chunks 261 (rank 12) and 391 (rank 10) — passages that share enough character sequences with the query.

### Step 2: Window expansion (±1 adjacent chunk)

For each retrieved chunk, also include the immediately preceding and following chunk from the same chapter. Since chunks 260 and 261 are sequential (chapter 12, seq 10 and 11), chunk 260 is automatically pulled in as a neighbor of 261.

**Result: all 3 target chunks retrieved.** 38 chunks total (~9,400 words) passed as context.

### Step 3: Nemotron answers correctly

```
ANSWER:
Dumbledore gave Harry an invisibility cloak for Christmas.

Evidence from the text:
- In Chapter 12, passage 10, Harry unwraps a parcel to find a silvery cloth
  that makes him invisible. Ron identifies it as an invisibility cloak.
- A note fell out: "Your father left this in my possession before he died.
  It is time it was returned to you. Use it well. A Very Merry Christmas to you."
- In Chapter 17, Ron confirms: "D'you think he meant you to do it?
  Sending you your father's cloak and everything?"
```

## Architecture

```
Question
  │
  ├─ char trigrams → score all 400 chunks → top-15
  │
  ├─ ±1 window expansion → ~38 chunks
  │
  └─ Nemotron 9B (vLLM, localhost:8000)
       ├─ system: "Answer based ONLY on provided text"
       ├─ user: 38 chunks + question
       └─ output: reasoning + answer with citations
```

## Database Schema

SQLite (`hp_logic.db`) — Harry Potter and the Philosopher's Stone, chunked at 250 words.

| Table | Rows | Description |
|-------|------|-------------|
| `chapters` | 17 | Chapter titles |
| `chunks` | 400 | 250-word text chunks with chapter/seq |
| `chunks_fts` | — | FTS5 virtual table (porter tokenizer) |
| `np_index` | 5,291 | Noun phrase index with frequencies |
| `np_chunk_map` | 16,442 | Noun phrase ↔ chunk mapping |

## Token Budget (Nemotron 9B, Thinking enabled)

| Component | Tokens |
|-----------|--------|
| Context (38 chunks, English) | ~13,700 |
| Reasoning (thinking) | ~500 |
| Answer | ~200 |
| **Total** | **~14,200** |

For Japanese text (~10,000 chars), expect 1.5–2.0x token inflation due to LLaMA tokenizer inefficiency. `max_tokens=4096` minimum, `8192` recommended.

## Files

| File | Description |
|------|-------------|
| `build_db.py` | Build SQLite from CSV |
| `build_word_index.py` | Build noun phrase index + FTS |
| `extract_proper_nouns.py` | Extract proper nouns from chunks |
| `generate_nemotron_prompts.py` | Generate batch prompts for Nemotron |
| `run_nemotron.py` | Execute prompts against vLLM |
| `harry_potter_books.csv` | Source text |
| `hp_logic.db` | Pre-built SQLite database |
| `proper_nouns.json` | Extracted proper nouns (430 entries) |

## Key Takeaway

> Multi-hop RAG doesn't need embeddings, vector DBs, or query decomposition agents.
> Character trigrams + adjacent chunk window + a capable local LLM can solve it
> with just SQLite and ~50 lines of retrieval code.

## Requirements

- Python 3.12+
- vLLM with Nemotron 9B (or compatible OpenAI-API server on localhost:8000)
- SQLite3 (built-in)

## License

MIT
