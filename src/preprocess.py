import json
import re
from collections import Counter
from typing import List, Tuple

import torch
from config import end_token
from utils import encode, tag, tokenizer, get_vocab


def fix_linebreaks(txt: str) -> Tuple[str, List[str]]:
    """
    Line breaks are set before each timestamp, where '<END>' tokens are placed. While senders are also
    tagged as individual tokens, a list of unique senders is collected.
    """
    txt_list = []
    sender_list = []
    date_fmt = "\[\d{2}\.\d{2}\.\d{2}, \d{2}:\d{2}:\d{2}\] "

    txt = txt.replace("\n", "")
    txt = re.split(date_fmt, txt)[1:]
    
    for i in txt:
        sender, msg = i.split(": ", 1)
        sender_list.append(sender)
        txt_list.append(tag(sender)+msg+end_token)

    txt = "".join(txt_list)
    sender_list = tag(list(set(sender_list)))
    return txt, sender_list


def drop_chars(txt: str, drop: List[str]) -> str:
    """Drop a list of characters from string."""

    return txt.translate(str.maketrans("", "", "".join(drop)))


def get_infrequent_chars(txt: str, min_count: int) -> List[str]:
    """Identify characters that appear less than a minimum count."""

    chars_counts = Counter(txt)
    chars_remove = [k for k,v in chars_counts.items() if v< min_count]
    return chars_remove


if __name__ == "__main__":

    # read corpus of whatsapp chat messages
    with open("../assets/chat.txt", "r") as f:
        text = f.read()

    # shrink vocabulary by eliminating rare characters
    infreq_chars = get_infrequent_chars(text, min_count=100)
    infreq_chars += ["~\u202f"]
    text = drop_chars(text, infreq_chars)

    # split text with end tokens
    text, senders = fix_linebreaks(text)

    # convert text from string to list (each element is a token)
    tokens = tokenizer(text, senders)

    # get vocabulary of corpus to file
    vocab = get_vocab(tokens)

    # encode characters into a tensor of integers
    data = encode(tokens, vocab)

    # split up the data into train and validation set
    n = int(0.9*len(data))
    train_data = data[:n]
    valid_data = data[n:]

    # export tensors
    torch.save(train_data, "../assets/train.pt")
    torch.save(valid_data, "../assets/valid.pt")

    with open("../assets/vocab.txt", "w", encoding="utf-8") as f:
        f.write(json.dumps(vocab))

    with open("../assets/senders.txt", "w", encoding="utf-8") as f:
        f.write(json.dumps(senders))

    print("SUCCESS")