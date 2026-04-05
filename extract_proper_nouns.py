"""
固有名詞抽出 & クリーニング → カテゴリ分類してJSON出力
- NLTKのPOSタグでNNP/NNPSを含む名詞句を抽出
- ノイズ除去（1語の一般動詞/形容詞/感嘆詞等）
- 頻度2以上をJSON出力
"""

import sqlite3
import re
import json
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize

DB_PATH = "/home/soy/hp-rag/hp_logic.db"
OUT_PATH = "/home/soy/hp-rag/proper_nouns.json"

NP_GRAMMAR = r"NP: {<JJ|NNP|NNPS>*<NN|NNS|NNP|NNPS>+}"
PARSER = nltk.RegexpParser(NP_GRAMMAR)

# 1語でNNPに誤タグされやすいノイズ語（感嘆詞・動詞・副詞・形容詞等）
NOISE_WORDS = {
    "come", "sorry", "was", "how", "where", "did", "are", "don", "got",
    "well", "hey", "right", "please", "hurry", "look", "shut", "inside",
    "someone", "anyone", "sir", "mom", "dad", "mummy", "mommy", "bye",
    "brought", "disappeared", "vanished", "said", "too", "him", "much",
    "just", "half", "deep", "anyway", "okay", "clearly", "lucky",
    "nevertheless", "could", "yeh", "nah", "blimey", "poor", "bring",
    "stay", "excuse", "bright", "trouble", "smoke", "danger", "white",
    "fine", "lot", "lots", "bit", "life", "ages", "head", "you",
    "sobbed", "barked", "gasped", "cried", "moaned", "snapped", "croaked",
    "growled", "sneered", "demanded", "roared", "breathed", "shouted",
    "nudged", "sat", "woke", "told", "thought", "shook", "sniffed",
    "a", "the", "id", "las", "got", "yeh", "jus", "fer", "bin",
}

# 明らかにフレーズの一部が壊れているパターン
NOISE_PATTERNS = [
    r"^[\.\,\!\?]+",          # 句読点始まり
    r"^\-",                   # ハイフン始まり
    r"^'",                    # クオート始まり
    r"sobbed\s", r"barked\s", r"gasped\s", r"cried\s",
    r"moaned\s", r"snapped\s", r"croaked\s", r"growled\s",
    r"sneered\s", r"demanded\s", r"roared\s", r"breathed\s",
    r"shouted\s", r"nudged\s",
    r"\bsat\b.*\bHagrid\b",
    r"\bwhile\b",
    r"\bsecond\s+Harry\b",
    r"\bmoment\b.*\b(Uncle|Neville)\b",
    r"\btold\s+Harry\b",
    r"\bteam\s+Gryffindor\b",
    r"\bonly\s+Stone\b",
    r"\bsure\s+Professor\b",
    r"\bpast\s+Fluffy\b",
    r"\bsomething\s+Dumbledore\b",
    r"\beverything\s+Hagrid\b",
    r"\bfirst\b.*\blesson\b",
    r"\blast\b.*\bmatch\b",
    r"\bown\b.*\bteam\b",
    r"\bgood\b.*\bmark\b",
    r"\babysmal\b",
    r"\bexcellent\b.*\bChaser\b",
    r"\bfaithful\b",
    r"\blucky\s+Harry\b",
    r"\bpoor\s+little\b",
    r"\bstupid\b",
    r"\bhard-faced\b",
    r"\bbeady-eyed\b",
    r"\binteresting\b",
    r"\bforgive\b",
    r"\bcousin\b",
    r"\blet\b.*\bprofessor\b",
    r"\byou\b.*\bprofessor\b",
    r"\bbut\b.*\bprofessor\b",
    r"\bjoke\b.*\bprofessor\b",
    r"\bReckon\b",
    r"INSULT",
    r"SEIZE",
    r"\bBOOM\b",
    r"\bBudge\b",
]


def is_noise(phrase):
    low = phrase.lower().strip()
    # 1語でノイズリストに含まれる
    if " " not in low and low.rstrip(".") in NOISE_WORDS:
        return True
    # 末尾ピリオド付き1語（Harry. Ron. 等）→ ピリオド除去版が残るので除外
    if low.endswith(".") and " " not in low:
        return True
    # ノイズパターンにマッチ
    for pat in NOISE_PATTERNS:
        if re.search(pat, phrase, re.IGNORECASE):
            return True
    # 全大文字で5語以上（叫び文等）
    if phrase == phrase.upper() and len(phrase.split()) >= 3:
        return True
    return False


def extract():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT chunk_id, text FROM chunks")
    rows = c.fetchall()
    conn.close()

    counter = Counter()
    # どの章に出現するか記録
    chunk_chapters = {}
    conn2 = sqlite3.connect(DB_PATH)
    c2 = conn2.cursor()
    c2.execute("SELECT chunk_id, chapter_num FROM chunks")
    for cid, ch in c2.fetchall():
        chunk_chapters[cid] = ch
    conn2.close()

    phrase_chapters = {}  # phrase → set of chapter_nums

    for chunk_id, text in rows:
        clean = re.sub(r"[\u2018\u2019\u201c\u201d]", "'", text)
        tokens = word_tokenize(clean)
        tagged = nltk.pos_tag(tokens)
        tree = PARSER.parse(tagged)

        for subtree in tree:
            if isinstance(subtree, nltk.Tree) and subtree.label() == "NP":
                leaves = subtree.leaves()
                has_proper = any(t in ("NNP", "NNPS") for _, t in leaves)
                if has_proper:
                    phrase = " ".join(w for w, t in leaves)
                    if not is_noise(phrase):
                        counter[phrase] += 1
                        phrase_chapters.setdefault(phrase, set()).add(chunk_chapters[chunk_id])

    # 頻度2以上
    results = []
    for phrase, freq in sorted(counter.items(), key=lambda x: -x[1]):
        if freq < 2:
            break
        chapters = sorted(phrase_chapters[phrase])
        results.append({
            "phrase": phrase,
            "frequency": freq,
            "chapters": chapters,
        })

    return results


def save(results):
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Output: {OUT_PATH}")
    print(f"Total proper noun phrases: {len(results)}")
    print(f"\nTop 50:")
    for r in results[:50]:
        chs = ",".join(str(c) for c in r["chapters"])
        print(f"  {r['frequency']:4d}  {r['phrase']:<35s}  ch:[{chs}]")


if __name__ == "__main__":
    print("Extracting proper noun phrases...")
    results = extract()
    save(results)
