import torch
import sentencepiece as spm
import json
import os
from train.model import TextToSVG

BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR      = os.path.join(BASE_DIR, "results")
SAVE_PATH = os.path.join(RESULT_DIR, "model", "model.pt")
MODEL_PATH = os.path.join(RESULT_DIR, "model", "model.pt")

TOKEN2ID_JSON   = os.path.join(RESULT_DIR, "token2id.json")
SP_MODEL        = os.path.join(RESULT_DIR, "text_bpe.model")
MAX_LEN         = 512
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"

sp = spm.SentencePieceProcessor(model_file=SP_MODEL)
token2id = json.load(open(TOKEN2ID_JSON, encoding="utf-8"))
id2token = {v: k for k, v in token2id.items()}

VOCAB_TEXT = sp.get_piece_size()
VOCAB_SVG  = len(token2id)

BOS_ID = token2id["<bos>"]
EOS_ID = token2id["<eos>"]
PAD_ID = token2id["<pad>"]
UNK_ID = token2id["<unk>"]

model = TextToSVG(VOCAB_TEXT, VOCAB_SVG).to(DEVICE)
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

@torch.no_grad()
def infer(prompt: str):
    text_ids = [sp.bos_id()] + sp.encode(prompt, out_type=int) + [sp.eos_id()]
    text_ids = torch.tensor(text_ids, device=DEVICE).unsqueeze(0)

    out = torch.tensor([[BOS_ID]], device=DEVICE)

    for _ in range(MAX_LEN):
        logits = model(text_ids, out)
        next_id = logits[:, -1].argmax(-1, keepdim=True)
        out = torch.cat([out, next_id], dim=1)
        if next_id.item() == EOS_ID:
            break

    svg_ids = out.squeeze().tolist()[1:-1]
    tokens = [id2token.get(i, "<unk>") for i in svg_ids]
    svg_path = decode_svg_ids(tokens)
    return svg_path

def decode_svg_ids(tokens: list[str]) -> str:
    """
    의미 단위 토큰 리스트 → SVG path 문자열
    예: ["M_3_3", "L_4_4", "Z"] → "M 3 3 L 4 4 Z"
    """
    d_tokens = []
    for tok in tokens:
        if tok in {"PATH_START", "PATH_END", "<bos>", "<eos>", "<pad>", "<unk>"}:
            continue
        d_tokens.append(tok.replace("_", " "))
    return " ".join(d_tokens)

if __name__ == "__main__":
    while True:
        prompt = input("🔍 Prompt 입력 ('exit' 입력 시 종료): ")
        if prompt.strip().lower() == "exit":
            break
        svg_d = infer(prompt.strip())
        svg_xml = (
            '<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg" '
            'stroke="black" fill="none" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
            f'<path d="{svg_d}"/></svg>'
        )
        print(f"🖋 {prompt.strip()} to Svg:\n{svg_xml}")   
        with open(f"${prompt.strip()}.svg", "w", encoding="utf-8") as f:
            f.write(svg_xml)
