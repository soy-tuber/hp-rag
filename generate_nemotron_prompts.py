"""
Nemotron用プロンプト生成スクリプト
- proper_nouns.json の各固有名詞に対して
- 出現チャンクから文脈を取得
- Nemotronに投げるプロンプトをバッチ化してJSON出力

使い方:
  1. このスクリプトを実行 → nemotron_tasks.jsonl が生成される
  2. Desktop PCにhp_logic.dbとnemotron_tasks.jsonlをコピー
  3. run_nemotron.py を実行（ollama/vllm等に対応）
"""

import sqlite3
import json

DB_PATH = "/home/soy/hp-rag/hp_logic.db"
NOUNS_PATH = "/home/soy/hp-rag/proper_nouns.json"
TASKS_PATH = "/home/soy/hp-rag/nemotron_tasks.jsonl"

# 1名詞あたり最大チャンク数（コンテキスト制限対策）
MAX_CHUNKS_PER_NOUN = 8

SYSTEM_PROMPT = """\
You are a text analyst. You will be given excerpts from a novel and a proper noun that appears in them.
Your task is to describe what this proper noun refers to, based ONLY on the provided text excerpts.
Do NOT use any external knowledge about this novel or its characters.
Only state what can be directly inferred from the given text.

Respond in this exact JSON format:
{
  "phrase": "<the proper noun>",
  "category": "<person|place|creature|object|organization|event|spell|other>",
  "description": "<1-3 sentences describing what this is, based only on the text>",
  "related_to": ["<other proper nouns mentioned alongside this one>"]
}"""


def load_chunks_for_noun(conn, phrase, chapters):
    """固有名詞が出現するチャンクを取得（頻度の高い章を優先）"""
    c = conn.cursor()
    # FTS5で検索、章情報付き
    # phraseが複数語の場合はクォートで囲む
    search_term = f'"{phrase}"' if " " in phrase else phrase
    try:
        c.execute("""
            SELECT ck.chunk_id, ck.chapter_num, ck.seq, ck.text
            FROM chunks_fts
            JOIN chunks ck ON chunks_fts.rowid = ck.chunk_id
            WHERE chunks_fts MATCH ?
            ORDER BY ck.chapter_num, ck.seq
            LIMIT ?
        """, (search_term, MAX_CHUNKS_PER_NOUN))
        return c.fetchall()
    except Exception:
        return []


def build_prompt(phrase, chunks):
    """1つの固有名詞に対するユーザープロンプトを構築"""
    excerpts = []
    for chunk_id, ch, seq, text in chunks:
        excerpts.append(f"[Chapter {ch}, passage {seq}]\n{text}")

    excerpts_text = "\n\n---\n\n".join(excerpts)

    return f"""Proper noun to analyze: "{phrase}"

Text excerpts where it appears:

{excerpts_text}

Based ONLY on these excerpts, describe what "{phrase}" refers to. Respond in JSON format."""


def main():
    with open(NOUNS_PATH, encoding="utf-8") as f:
        nouns = json.load(f)

    conn = sqlite3.connect(DB_PATH)
    tasks = []
    skipped = 0

    for noun in nouns:
        phrase = noun["phrase"]
        chapters = noun["chapters"]
        chunks = load_chunks_for_noun(conn, phrase, chapters)

        if not chunks:
            skipped += 1
            continue

        user_prompt = build_prompt(phrase, chunks)
        tasks.append({
            "phrase": phrase,
            "frequency": noun["frequency"],
            "chapters": chapters,
            "system_prompt": SYSTEM_PROMPT,
            "user_prompt": user_prompt,
        })

    conn.close()

    # JSONL形式で出力（1行1タスク）
    with open(TASKS_PATH, "w", encoding="utf-8") as f:
        for task in tasks:
            f.write(json.dumps(task, ensure_ascii=False) + "\n")

    print(f"Generated: {TASKS_PATH}")
    print(f"Tasks: {len(tasks)}")
    print(f"Skipped (no chunks found): {skipped}")
    print(f"\nSample prompt for '{tasks[0]['phrase']}':")
    print(f"  System: {tasks[0]['system_prompt'][:80]}...")
    print(f"  User:   {tasks[0]['user_prompt'][:200]}...")


if __name__ == "__main__":
    main()
