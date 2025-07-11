import json, torch
from torch.utils.data import Dataset

class TextToSVGDataset(Dataset):
    """
    JSONL 파일을 읽어 (text_ids, svg_ids) 텐서로 반환.
    max_len을 넘어가면 잘라내고, 패딩은 collate_fn에서 처리.
    """
    def __init__(self, jsonl_path: str, max_len: int = 512):
        self.rows = [json.loads(line) for line in open(jsonl_path, encoding="utf-8")]
        self.max_len = max_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]

        # (1) 시퀀스 자르기
        txt = row["text_ids"][: self.max_len]
        svg = row["svg_ids"][: self.max_len]

        return {
            "text": torch.tensor(txt, dtype=torch.long),
            "svg":  torch.tensor(svg, dtype=torch.long)
        }


def collate_fn(batch: list[dict]):
    """
    batch = [{"text": tensor, "svg": tensor}, ...]
    반환: dict {
        text(ids padded)   : (B, T_text),
        svg_in(ids padded) : (B, T_svg-1)  # 디코더 입력
        svg_tgt(ids padded): (B, T_svg-1)  # 예측 대상
        text_mask          : (B, T_text)   # 1=실제, 0=pad
        svg_mask           : (B, T_svg-1)
    }
    """
    pad_id = 0

    # ── 길이 계산
    t_max = max(len(x["text"]) for x in batch)
    s_max = max(len(x["svg"])  for x in batch)

    # ── 패딩 헬퍼
    def pad(seq, max_len):
        if len(seq) < max_len:
            return torch.cat([seq, seq.new_full((max_len - len(seq),), pad_id)])
        return seq

    # ── 스택
    text = torch.stack([pad(b["text"], t_max) for b in batch])
    svg  = torch.stack([pad(b["svg"],  s_max) for b in batch])

    # ── 디코더: shift-right
    svg_in  = svg[:, :-1]           # <bos> ... last-1
    svg_tgt = svg[:, 1:]            #            first+1 ... <eos>

    # ── 마스크 (pad==0 이면 0)
    text_mask = (text != pad_id).long()
    svg_mask  = (svg_tgt != pad_id).long()

    return {
        "text": text,
        "text_mask": text_mask,
        "svg_in": svg_in,
        "svg_tgt": svg_tgt,
        "svg_mask": svg_mask
    }
