import os
import json
import sentencepiece as spm

def load_vocab_and_tokenizer(token2id_path, sp_model_path):
    token2id = json.load(open(token2id_path, encoding="utf-8"))
    sp = spm.SentencePieceProcessor(model_file=sp_model_path)
    return token2id, sp

def convert_and_split_data(
    svg_tokens_jsonl,
    token2id,
    sp,
    output_jsonl="./results/train_pairs.jsonl",
    split_dir="./results",
    train_ratio=1,
    val_ratio=0.2
):
    svg_bos = token2id["<bos>"]
    svg_eos = token2id["<eos>"]
    svg_unk = token2id["<unk>"]
    txt_bos, txt_eos = sp.bos_id(), sp.eos_id()

    def svg_tokens_to_ids(tokens):
        return [svg_bos] + [token2id.get(t, svg_unk) for t in tokens] + [svg_eos]

    def text_to_ids(text):
        return [txt_bos] + sp.encode(text, out_type=int) + [txt_eos]

    os.makedirs(os.path.dirname(output_jsonl), exist_ok=True)

    processed = []
    with open(svg_tokens_jsonl, encoding="utf-8") as fin:
        for line in fin:
            obj = json.loads(line)
            fname = obj["file"]
            svg_ids = svg_tokens_to_ids(obj["paths"])
            prompt = fname.replace(".svg", "").replace("_", " ") + " icon"
            text_ids = text_to_ids(prompt)

            processed.append({
                "text": prompt,
                "text_ids": text_ids,
                "svg_ids": svg_ids
            })

    with open(output_jsonl, "w", encoding="utf-8") as fout:
        for row in processed:
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Split
    n = len(processed)
    train_end = int(train_ratio * n)
    val_start = int((0.8) * n)
    val_end = int((train_ratio + val_ratio) * n)

    os.makedirs(split_dir, exist_ok=True)
    with open(os.path.join(split_dir, "train.jsonl"), "w", encoding="utf-8") as f:
        f.write("\n".join(json.dumps(r, ensure_ascii=False) for r in processed[:train_end]))
    with open(os.path.join(split_dir, "val.jsonl"), "w", encoding="utf-8") as f:
        f.write("\n".join(json.dumps(r, ensure_ascii=False) for r in processed[val_start:val_end]))
    # with open(os.path.join(split_dir, "test.jsonl"), "w", encoding="utf-8") as f:
    #     f.write("\n".join(json.dumps(r, ensure_ascii=False) for r in processed[val_end:]))

    print(f"✔ 총 {n}개 샘플 중 → train/val/test = {train_end}/{val_end-train_end}/{n-val_end}")
