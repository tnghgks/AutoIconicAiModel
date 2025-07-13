import torch, torch.nn as nn

# ─────────── 1. 하이퍼파라미터 ───────────
BATCH      = 64
D_MODEL    = 256
N_HEAD     = 4
N_LAYERS   = 3
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

class TextToSVG(nn.Module):
    def __init__(self, VOCAB_TEXT, VOCAB_SVG):
        super().__init__()
        self.txt_emb = nn.Embedding(VOCAB_TEXT, D_MODEL, padding_idx=0)
        self.svg_emb = nn.Embedding(VOCAB_SVG,  D_MODEL, padding_idx=0)
        self.tf = nn.Transformer(
            dropout=0.3,
            d_model=D_MODEL, nhead=N_HEAD,
            num_encoder_layers=N_LAYERS,
            num_decoder_layers=N_LAYERS,
            batch_first=True,
        )
        self.out = nn.Linear(D_MODEL, VOCAB_SVG)

    def generate_square_subsequent_mask(self, sz: int, device):
        mask = torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)
        return mask.to(device)

    def forward(self, txt, svg_in):
        enc = self.txt_emb(txt)      # (B, Tₜₑₓₜ, D)
        dec = self.svg_emb(svg_in)   # (B, Tₛᵥ₉-1, D)

        tgt_mask = self.generate_square_subsequent_mask(svg_in.size(1), svg_in.device)
        h = self.tf(enc, dec, tgt_mask=tgt_mask)   # ← 마스크 전달
        return self.out(h)           # (B, Tₛᵥ₉-1, V_svg)
    
