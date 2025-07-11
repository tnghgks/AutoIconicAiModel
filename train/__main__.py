
from torch.utils.data import DataLoader
from train.svg_dataset import TextToSVGDataset, collate_fn
from torch.utils.data import Dataset
import sentencepiece as spm
import json, torch, os
from train.text_to_svg import TextToSVG
from config import RESULT_DIR
from train.to_weight import to_weight
from config import DEVICE

# ─────────── 1. 하이퍼파라미터 ───────────
BATCH           = 64
D_MODEL         = 256
N_HEAD          = 8
N_LAYERS        = 4
LR              = 1e-4
EPOCHS          = 100   
MAX_LEN         = 512
SP_MODEL        = os.path.join(RESULT_DIR, "text_bpe.model")
TRAIN_JSON      = os.path.join(RESULT_DIR, "train.jsonl")
VAL_JSONL       = os.path.join(RESULT_DIR, "val.jsonl")
TOKEN2ID_JSON   = os.path.join(RESULT_DIR, "token2id.json")
SAVE_PATH = os.path.join(RESULT_DIR, "model", "model.pt")
BEST_PATH = os.path.join(RESULT_DIR, "model", "best_model.pt")
os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)


def main():
    train_set = TextToSVGDataset(TRAIN_JSON, MAX_LEN)
    val_set   = TextToSVGDataset(VAL_JSONL,   MAX_LEN)

    train_loader = DataLoader(
        train_set,
        batch_size=BATCH,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_set,
        batch_size=BATCH,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True
    )
    sp = spm.SentencePieceProcessor(model_file=SP_MODEL)
    token2id = json.load(open(TOKEN2ID_JSON, encoding="utf-8"))

    VOCAB_TEXT = sp.get_piece_size()
    VOCAB_SVG  = len(token2id)

    model = TextToSVG(VOCAB_TEXT,VOCAB_SVG).to(DEVICE)
    opt   = torch.optim.AdamW(model.parameters(), lr=LR)
    loss_fn = torch.nn.CrossEntropyLoss(
        weight=to_weight(),
        ignore_index=0,
        label_smoothing=0.1 
    )

    best_val_loss = float("inf") 

    for epoch in range(1, EPOCHS+1):
        model.train()
        tr_loss = 0
        for batch in train_loader:
            for k in batch: batch[k] = batch[k].to(DEVICE)

            logits = model(batch["text"], batch["svg_in"])
            loss = loss_fn(
                logits.view(-1, VOCAB_SVG),
                batch["svg_tgt"].reshape(-1)
            )

            opt.zero_grad()
            loss.backward()
            opt.step()
            tr_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                for k in batch: batch[k] = batch[k].to(DEVICE)
                logits = model(batch["text"], batch["svg_in"])
                val_loss += loss_fn(
                    logits.view(-1, VOCAB_SVG),
                    batch["svg_tgt"].reshape(-1)
                ).item()

        tr_loss_avg = tr_loss / len(train_loader)
        val_loss_avg = val_loss / len(val_loader)

        print(f"Epoch {epoch:>2} | train {tr_loss_avg:.4f} | val {val_loss_avg:.4f}")
        torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "val_loss": val_loss_avg
            }, SAVE_PATH)
        
        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            torch.save(model.state_dict(), BEST_PATH)
            print(f"🏅 Best 모델 저장 → {BEST_PATH}")
        
if __name__ == "__main__":
    main()
    print("SVG Tokenizer completed.")