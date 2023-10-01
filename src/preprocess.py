from collections import Counter
import torch

from src.utils import encode


def drop_datetime(txt: str) -> str:
    """Strip away the datetime information of each message."""
    
    txt = txt.split("\n")
    txt_clean = [i[21:] for i in txt]
    txt_joined = "\n".join(txt_clean)
    return txt_joined


def get_infrequent_chars(txt: str, min_count: int) -> list:
    """Identify characters that appear less than a minimum count."""

    chars_counts = Counter(txt)
    chars_remove = [k for k,v in chars_counts.items() if v< min_count]
    return chars_remove


def drop_chars(txt: str, drop: list) -> str:
    """Drop a list of characters from string."""

    return txt.translate(str.maketrans("", "", "".join(drop)))


def get_vocab(text: str) -> list:
    """Returns a sorted list of all available characters in the corpus."""

    return sorted(list(set("".join(text))))


if __name__ == "__main__":

    # load corpus of whatsapp chat messages
    text = open("../assets/chat.txt", "r").read()
    text = drop_datetime(text)

    # shrink vocabulary by eliminating rare characters
    infreq_chars = get_infrequent_chars(text, min_count=500)
    text = drop_chars(text, infreq_chars)

    # write vocabulary of corpus to file
    vocab = get_vocab(text)
    open("../assets/vocab.txt", "w").write("".join(vocab))

    # encode characters into a tensor of integers
    data = encode(text, vocab)

    # split up the data into train and validation set
    n = int(0.9*len(data))
    train_data = data[:n]
    valid_data = data[n:]

    # export tensors
    torch.save(train_data, "../assets/train.pt")
    torch.save(valid_data, "../assets/valid.pt")
