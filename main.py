import argparse
import subprocess

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["preprocess", "train", "infer"])
    parser.add_argument("--prompt", type=str, help="prompt for inference")
    args = parser.parse_args()

    if args.mode == "preprocess":
        subprocess.run(["python", "-m", "preprocess"])
        print("🚀 Starting training...")

    elif args.mode == "train":
        subprocess.run(["python", "-m", "train"])
        print("🚀 Starting training...")
    elif args.mode == "infer":
        subprocess.run(["python", "-m", "inference"])


if __name__ == "__main__":
    main()