"""
名詞句（Noun Phrase）頻出インデックス生成
- SQLiteのchunksから全テキスト取得
- NLTKでPOSタグ付け → RegexpParserで名詞句を抽出
  パターン: (形容詞|固有名詞)* + 名詞+
- 頻出順でnp_indexテーブルに格納
"""

import sqlite3
import re
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

DB_PATH = "/home/soy/hp-rag/hp_logic.db"
STOP_WORDS = set(stopwords.words("english"))

# 名詞句の文法パターン
# JJ=形容詞, NNP/NNPS=固有名詞, NN/NNS=一般名詞
NP_GRAMMAR = r"""
    NP: {<JJ|NNP|NNPS>*<NN|NNS|NNP|NNPS>+}
"""
NP_PARSER = nltk.RegexpParser(NP_GRAMMAR)


def normalize_np(words):
    """名詞句の正規化（小文字化、ストップワード除去）"""
    filtered = [w for w in words if w.lower() not in STOP_WORDS and len(w) >= 2]
    if not filtered:
        return None
    return " ".join(w.lower() for w in filtered)


def extract_noun_phrases():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT chunk_id, chapter_num, text FROM chunks ORDER BY chunk_id")
    rows = c.fetchall()
    conn.close()

    global_counter = Counter()
    chunk_nps = {}  # chunk_id → list of noun phrases

    for chunk_id, chapter_num, text in rows:
        clean = re.sub(r"[`\u2018\u2019\u201c\u201d]", "'", text)
        tokens = word_tokenize(clean)
        tagged = nltk.pos_tag(tokens)
        tree = NP_PARSER.parse(tagged)

        nps = []
        for subtree in tree:
            if isinstance(subtree, nltk.Tree) and subtree.label() == "NP":
                words = [word for word, tag in subtree.leaves()]
                np = normalize_np(words)
                if np:
                    nps.append(np)

        global_counter.update(nps)
        chunk_nps[chunk_id] = nps

    return global_counter, chunk_nps


def build_index(global_counter, chunk_nps):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.executescript("""
        DROP TABLE IF EXISTS np_chunk_map;
        DROP TABLE IF EXISTS np_index;

        CREATE TABLE np_index (
            np_id      INTEGER PRIMARY KEY AUTOINCREMENT,
            phrase     TEXT UNIQUE NOT NULL,
            total_freq INTEGER NOT NULL
        );

        CREATE TABLE np_chunk_map (
            np_id    INTEGER NOT NULL REFERENCES np_index(np_id),
            chunk_id INTEGER NOT NULL REFERENCES chunks(chunk_id),
            freq     INTEGER NOT NULL,
            PRIMARY KEY (np_id, chunk_id)
        );

        CREATE INDEX idx_npm_chunk ON np_chunk_map(chunk_id);
    """)

    # np_index 挿入（頻出順）
    np_to_id = {}
    for phrase, freq in global_counter.most_common():
        c.execute("INSERT INTO np_index (phrase, total_freq) VALUES (?, ?)", (phrase, freq))
        np_to_id[phrase] = c.lastrowid

    # np_chunk_map 挿入
    for chunk_id, nps in chunk_nps.items():
        local = Counter(nps)
        for phrase, freq in local.items():
            nid = np_to_id.get(phrase)
            if nid:
                c.execute(
                    "INSERT INTO np_chunk_map (np_id, chunk_id, freq) VALUES (?, ?, ?)",
                    (nid, chunk_id, freq),
                )

    conn.commit()

    # 統計表示
    c.execute("SELECT COUNT(*) FROM np_index")
    total = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM np_chunk_map")
    total_maps = c.fetchone()[0]

    print(f"Unique noun phrases: {total}")
    print(f"NP-chunk mappings:   {total_maps}")
    print(f"\nTop 40 noun phrases:")
    c.execute("SELECT phrase, total_freq FROM np_index ORDER BY total_freq DESC LIMIT 40")
    for p, f in c.fetchall():
        print(f"  {f:4d}  {p}")

    conn.close()


if __name__ == "__main__":
    print("Extracting noun phrases from chunks...")
    gc, cn = extract_noun_phrases()
    print(f"Found {len(gc)} unique noun phrases\n")
    build_index(gc, cn)
