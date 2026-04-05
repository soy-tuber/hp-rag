"""
Nemotron実行スクリプト（Desktop PC用 / vLLM）
- nemotron_tasks.jsonl を読み込み
- vLLM の OpenAI互換API に1件ずつ投げて結果を回収
- nemotron_results.json に出力

前提: vLLM がポート8000で起動済み

使い方:
  python run_nemotron.py [--model MODEL_NAME] [--base-url URL]

デフォルト:
  --model   nemotron      (vLLM起動時のモデル名に合わせて変更)
  --base-url http://localhost:8000
"""

import json
import argparse
import urllib.request
import urllib.error
import sys
import time

TASKS_PATH = "nemotron_tasks.jsonl"
RESULTS_PATH = "nemotron_results.json"


def call_vllm(base_url, model, system_prompt, user_prompt):
    """vLLM の OpenAI互換 /v1/chat/completions を呼ぶ"""
    url = f"{base_url}/v1/chat/completions"
    payload = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.1,
        "max_tokens": 512,
    }).encode("utf-8")

    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    resp = urllib.request.urlopen(req, timeout=120)
    data = json.loads(resp.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"]


def parse_json_response(text):
    """レスポンスからJSON部分を抽出"""
    # ```json ... ``` ブロック対応
    if "```" in text:
        start = text.find("```")
        end = text.rfind("```")
        if start != end:
            inner = text[start:end].split("\n", 1)
            if len(inner) > 1:
                text = inner[1]

    # { } を探す
    brace_start = text.find("{")
    brace_end = text.rfind("}")
    if brace_start != -1 and brace_end != -1:
        try:
            return json.loads(text[brace_start:brace_end + 1])
        except json.JSONDecodeError:
            pass
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="nemotron", help="ollama model name")
    parser.add_argument("--base-url", default="http://localhost:8000", help="vLLM base URL")
    parser.add_argument("--resume", action="store_true", help="skip already processed phrases")
    args = parser.parse_args()

    # タスク読み込み
    tasks = []
    with open(TASKS_PATH, encoding="utf-8") as f:
        for line in f:
            tasks.append(json.loads(line))

    # 既存結果の読み込み（resume用）
    existing = {}
    if args.resume:
        try:
            with open(RESULTS_PATH, encoding="utf-8") as f:
                for r in json.load(f):
                    existing[r.get("phrase", "")] = r
            print(f"Resuming: {len(existing)} already done")
        except (FileNotFoundError, json.JSONDecodeError):
            pass

    results = list(existing.values())
    total = len(tasks)
    errors = 0

    for i, task in enumerate(tasks):
        phrase = task["phrase"]

        if phrase in existing:
            continue

        print(f"[{i+1}/{total}] {phrase} (freq={task['frequency']})...", end=" ", flush=True)

        try:
            raw = call_vllm(args.base_url, args.model, task["system_prompt"], task["user_prompt"])
            parsed = parse_json_response(raw)

            if parsed:
                parsed["frequency"] = task["frequency"]
                parsed["chapters"] = task["chapters"]
                results.append(parsed)
                cat = parsed.get("category", "?")
                print(f"OK [{cat}]")
            else:
                # パース失敗でも生テキストを保存
                results.append({
                    "phrase": phrase,
                    "category": "parse_error",
                    "description": raw[:500],
                    "related_to": [],
                    "frequency": task["frequency"],
                    "chapters": task["chapters"],
                })
                print(f"PARSE_ERROR")
                errors += 1

        except (urllib.error.URLError, Exception) as e:
            print(f"ERROR: {e}")
            errors += 1
            continue

        # 10件ごとに中間保存
        if (i + 1) % 10 == 0:
            with open(RESULTS_PATH, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

    # 最終保存
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nDone: {len(results)} results, {errors} errors")
    print(f"Output: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
