import argparse

from src import chat, preprocess, train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["preprocess", "train", "chat"], help="The mode to be execute.")
    parser.add_argument("--update", action="store_true", help="Flag when model shall be updated based on current parameters")
    args = parser.parse_args()

    if args.mode == "preprocess":
        preprocess.make_train_test()
    elif args.mode == "train":
        train.model_training(args.update)
    elif args.mode == "chat":
        chat.conversation()

if __name__ == "__main__":
    main()
