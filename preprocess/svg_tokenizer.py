import os, glob, json
import pandas as pd
from preprocess.svg_parser import svg_to_path_list, path_to_tokens

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_DIR = os.path.join(BASE_DIR, "dataset")
RESULT_DIR = os.path.join(BASE_DIR, "results")
MIN_FREQ  = 1
SPECIALS  = ["<pad>", "<unk>", "<bos>", "<eos>"]

def svg_tokenizer(input_dir=INPUT_DIR, min_freq=MIN_FREQ):
    print("🔧 Preprocessing started...")
    svg_files = glob.glob(os.path.join(input_dir, "**", "*.svg"), recursive=True)

    data = []
    processed = 0

    for fp in svg_files:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                svg_txt = f.read()
            paths = svg_to_path_list(svg_txt)
            tokens = [tok for d in paths for tok in path_to_tokens(d)]
            data.append({
                "file": os.path.basename(fp),
                "paths": tokens
            })
            processed += 1
        except Exception as e:
            print(f"[WARN] {fp}: {e}")

    print(f"✔ 완료! {processed}/{len(svg_files)}개 파일 처리됨")

    # → DataFrame 생성
    df = pd.DataFrame(data)

    # → paths 컬럼 펼치기 + 토큰 카운트
    token_counts = df.explode("paths")["paths"].value_counts()

    # → token2id, id2token 생성
    token2id = {tok: idx for idx, tok in enumerate(SPECIALS)}
    next_id = len(SPECIALS)

    for token, count in token_counts.items():
        if count < min_freq:
            continue
        if token not in token2id:
            token2id[token] = next_id
            next_id += 1

    id2token = {v: k for k, v in token2id.items()}

    # ───── JSON 저장 ─────────────────────────────
    os.makedirs(RESULT_DIR, exist_ok=True)

    df_path = os.path.join(RESULT_DIR, "svg_tokens.jsonl")
    with open(df_path, "w", encoding="utf-8") as f:
        for row in data:
            json.dump(row, f, ensure_ascii=False)
            f.write("\n")

    with open(os.path.join(RESULT_DIR, "token2id.json"), "w", encoding="utf-8") as f:
        json.dump(token2id, f, ensure_ascii=False, indent=2)

    with open(os.path.join(RESULT_DIR, "id2token.json"), "w", encoding="utf-8") as f:
        json.dump(id2token, f, ensure_ascii=False, indent=2)

    print(f"📦 저장 완료 → {RESULT_DIR}")
