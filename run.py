import argparse
import subprocess

def main():
    parser = argparse.ArgumentParser(description="asdf")
    parser.add_argument("mode", choices=["preprocess", "train", "chat"], help="The mode to be execute.")

    args = parser.parse_args()
    if args.mode == "preprocess":
        subprocess.run(["python", "src/preprocess.py"], check=True)
    elif args.mode == 'train':
        subprocess.run(["python", "src/train.py"], check=True)
    elif args.mode == 'chat':
        subprocess.run(["python", "src/chat.py"], check=True)

if __name__ == "__main__":
    main()
