"""
Phase 1: Harry Potter Book 1 → SQLite RAG Database
- CSV → 章結合 → 固定ワード数チャンク分割 → SQLite (FTS5)
"""

import csv
import sqlite3
import os

# --- 設定 ---
CSV_PATH = os.path.join(os.path.dirname(__file__), "harry_potter_books.csv")
DB_PATH = os.path.join(os.path.dirname(__file__), "hp_logic.db")
BOOK_FILTER = "Book 1: Philosopher's Stone"
CHUNK_SIZE = 250       # ワード数
CHUNK_OVERLAP = 50     # オーバーラップワード数

CHAPTER_TITLES = {
    1: "The Boy Who Lived",
    2: "The Vanishing Glass",
    3: "The Letters from No One",
    4: "The Keeper of the Keys",
    5: "Diagon Alley",
    6: "The Journey from Platform Nine and Three-Quarters",
    7: "The Sorting Hat",
    8: "The Potions Master",
    9: "The Midnight Duel",
    10: "Hallowe'en",
    11: "Quidditch",
    12: "The Mirror of Erised",
    13: "Nicolas Flamel",
    14: "Norbert the Norwegian Ridgeback",
    15: "The Forbidden Forest",
    16: "Through the Trapdoor",
    17: "The Man with Two Faces",
}

# --- Step 1: CSVからBook1の章テキストを読み込む ---
def load_chapters(csv_path):
    chapters = {}
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["book"] == BOOK_FILTER:
                ch = int(row["chapter"].split("-")[1])
                chapters.setdefault(ch, []).append(row["text"])
    # 各行を結合して章ごとの連続テキストにする
    return {ch: " ".join(lines) for ch, lines in chapters.items()}


# --- Step 2: 固定ワード数でチャンク分割（オーバーラップ付き） ---
def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start = end - overlap  # オーバーラップ分戻す
    return chunks


# --- Step 3: SQLite DB構築 ---
def build_database(chapters):
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # テーブル作成
    c.executescript("""
        CREATE TABLE chapters (
            chapter_num   INTEGER PRIMARY KEY,
            chapter_title TEXT NOT NULL
        );

        CREATE TABLE chunks (
            chunk_id    INTEGER PRIMARY KEY AUTOINCREMENT,
            chapter_num INTEGER NOT NULL REFERENCES chapters(chapter_num),
            seq         INTEGER NOT NULL,
            word_count  INTEGER NOT NULL,
            text        TEXT NOT NULL
        );

        CREATE VIRTUAL TABLE chunks_fts USING fts5(
            text,
            content=chunks,
            content_rowid=chunk_id,
            tokenize='porter'
        );

        -- Phase 2 で Nemotron が埋める
        CREATE TABLE concepts (
            concept_id  INTEGER PRIMARY KEY AUTOINCREMENT,
            name        TEXT UNIQUE NOT NULL,
            description TEXT
        );

        CREATE TABLE chunk_concepts (
            chunk_id   INTEGER NOT NULL REFERENCES chunks(chunk_id),
            concept_id INTEGER NOT NULL REFERENCES concepts(concept_id),
            PRIMARY KEY (chunk_id, concept_id)
        );

        CREATE INDEX idx_chunks_chapter ON chunks(chapter_num);
    """)

    # 章マスタ挿入
    for ch_num in sorted(chapters.keys()):
        title = CHAPTER_TITLES.get(ch_num, f"Chapter {ch_num}")
        c.execute("INSERT INTO chapters VALUES (?, ?)", (ch_num, title))

    # チャンク挿入 + FTS同期
    total_chunks = 0
    for ch_num in sorted(chapters.keys()):
        ch_chunks = chunk_text(chapters[ch_num])
        for seq, chunk_text_data in enumerate(ch_chunks, 1):
            wc = len(chunk_text_data.split())
            c.execute(
                "INSERT INTO chunks (chapter_num, seq, word_count, text) VALUES (?, ?, ?, ?)",
                (ch_num, seq, wc, chunk_text_data),
            )
            chunk_id = c.lastrowid
            # FTS5 content-sync
            c.execute(
                "INSERT INTO chunks_fts (rowid, text) VALUES (?, ?)",
                (chunk_id, chunk_text_data),
            )
            total_chunks += 1

    conn.commit()

    # 統計表示
    print(f"Database: {DB_PATH}")
    print(f"Chapters: {len(chapters)}")
    print(f"Chunks:   {total_chunks}")
    print()
    c.execute("""
        SELECT ch.chapter_num, ch.chapter_title, COUNT(*) as n,
               SUM(ck.word_count) as words
        FROM chunks ck JOIN chapters ch ON ck.chapter_num = ch.chapter_num
        GROUP BY ch.chapter_num ORDER BY ch.chapter_num
    """)
    for row in c.fetchall():
        print(f"  Ch {row[0]:2d}: {row[2]:3d} chunks, {row[3]:5d} words | {row[1]}")

    # FTS5テスト
    print("\n--- FTS5 test: 'invisible cloak' ---")
    c.execute("""
        SELECT ck.chunk_id, ck.chapter_num, ck.seq,
               snippet(chunks_fts, 0, '>>>', '<<<', '...', 20) as snip
        FROM chunks_fts
        JOIN chunks ck ON chunks_fts.rowid = ck.chunk_id
        WHERE chunks_fts MATCH 'invisible cloak'
        LIMIT 5
    """)
    for row in c.fetchall():
        print(f"  [chunk {row[0]}, ch{row[1]} seq{row[2]}] {row[3]}")

    conn.close()


if __name__ == "__main__":
    chapters = load_chapters(CSV_PATH)
    print(f"Loaded {len(chapters)} chapters from CSV\n")
    build_database(chapters)
