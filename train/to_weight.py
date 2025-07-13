from collections import Counter
import torch, os, json
from config import RESULT_DIR, DEVICE
from train.svg_dataset import TextToSVGDataset

TRAIN_JSONL = os.path.join(RESULT_DIR, "train.jsonl")
TOKEN2ID_JSON    =  os.path.join(RESULT_DIR, "token2id.json")

def to_weight():
    token2id = json.load(open(TOKEN2ID_JSON, encoding="utf-8"))

    train_set = TextToSVGDataset(TRAIN_JSONL, max_len=512)
    # ── train_set 이미 생성된 시점 ─────────────────────────────
    freq = Counter()

    for row in train_set.rows:           # raw JSON 딕셔너리
        freq.update(row["svg_ids"])      # 리스트 그대로 카운트

    vocab_size = len(token2id)
    weight = torch.ones(vocab_size, dtype=torch.float32)

    # 2-1. 커맨드 토큰 목록
    cmds = ['M','L','C','Q','A','Z','H','V','PATH_START','PATH_END']

    # 2-2. 커맨드에 4배 가중치
    for cmd in cmds:
        if cmd in token2id:
            weight[token2id[cmd]] = 4.0

    return weight.to(DEVICE)
