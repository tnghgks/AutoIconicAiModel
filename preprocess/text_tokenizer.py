import os
import sentencepiece as spm

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_DIR = os.path.join(BASE_DIR, "dataset")
RESULT_DIR = os.path.join(BASE_DIR, "results")
PROMPT_PATH = os.path.join(RESULT_DIR, "prompts.txt")
TEXT_BPE_PATH = os.path.join(RESULT_DIR, "text_bpe")

def text_tokenizer():
    prompts = []
    for filename in os.listdir(INPUT_DIR):
        if filename.endswith(".svg"):
            name = filename.replace(".svg", "").replace("_", " ")
            prompts.append(name + " icon")

    # 저장
    with open(PROMPT_PATH, "w", encoding="utf-8") as f:
        for line in prompts:
            f.write(line.strip() + "\n")

    spm.SentencePieceTrainer.train(
        input=PROMPT_PATH,
        model_prefix=TEXT_BPE_PATH,
        vocab_size=1000,
        model_type='bpe',
        bos_id=0, eos_id=1, unk_id=2, pad_id=3,
        user_defined_symbols=["icon", "button", "symbol"]
)