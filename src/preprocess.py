from collections import Counter
import torch


def drop_datetime(txt: str) -> str:
    """Strip away the datetime information of each message."""
    
    txt = txt.split("\n")
    txt_clean = [i[21:] for i in txt]
    txt_joined = "\n".join(txt_clean)
    return txt_joined


def get_infrequent_chars(txt: str, min_count: int) -> list:

    chars_counts = Counter(txt)
    chars_remove = [k for k,v in chars_counts.items() if v< min_count]
    return chars_remove


def drop_chars(txt: str, drop: list) -> str:
    return txt.translate(str.maketrans("", "", "".join(drop)))


def get_vocab(text: str) -> list:
    """Returns a sorted list of all available characters in the corpus."""

    return sorted(list(set("".join(text))))


def encode(s: str, vocab: list) -> torch.tensor:
    """Encode a string into a tensor of integers, given a fixed vocabulary."""

    map = {s:i for i,s in enumerate(vocab)}
    enc = [map[c] for c in s]
    enc = torch.tensor(enc, dtype=torch.long)
    return enc


def decode(tensor: torch.tensor, vocab: list) -> str:
    """Decode a tensor of integers, back into a string."""

    map_enc = {s:i for i,s in enumerate(vocab)}
    map_dec = {i:s for s,i in map_enc.items()}
    dec = [map_dec[i.item()] for i in tensor]
    dec = "".join(dec)
    return dec


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
