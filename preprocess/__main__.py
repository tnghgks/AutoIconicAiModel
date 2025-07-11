from preprocess.svg_tokenizer import svg_tokenizer
from preprocess.text_tokenizer import text_tokenizer
from preprocess.build_trainset import load_vocab_and_tokenizer, convert_and_split_data
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_DIR = os.path.join(BASE_DIR, "dataset")
RESULT_DIR = os.path.join(BASE_DIR, "results")
TOKEN2ID_JSON = os.path.join(RESULT_DIR, "token2id.json")
SP_MODEL      = os.path.join(RESULT_DIR, "text_bpe.model")
SVG_TOKENS_JSONL = os.path.join(RESULT_DIR, "svg_tokens.jsonl")

def main():
    svg_tokenizer()
    text_tokenizer()
    token2id, sp = load_vocab_and_tokenizer(TOKEN2ID_JSON, SP_MODEL)
    convert_and_split_data(SVG_TOKENS_JSONL, token2id, sp)

if __name__ == "__main__":
    main()
    print("SVG Tokenizer completed.")