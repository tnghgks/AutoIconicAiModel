import os, torch

BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR      = os.path.join(BASE_DIR, "results")
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
