import numpy as np


class Tokenizer:
    def __init__(self, corpus: str = None):
        self.vocab = {}
        self.id_to_token = {}
        if corpus:
            self.build_vocab(corpus)

    def build_vocab(self, corpus: str):
        unique_chars = sorted(set(corpus))
        self.vocab = {ch: i for i, ch in enumerate(unique_chars)}
        self.id_to_token = {i: ch for ch, i in self.vocab.items()}

    def encode(self, text: str) -> list[int]:
        return [self.vocab[ch] for ch in text if ch in self.vocab]

    def decode(self, ids: list[int]) -> str:
        return ''.join(self.id_to_token[i] for i in ids if i in self.id_to_token)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)


if __name__ == "__main__":
    import os
    corpus_path = os.path.join(os.path.dirname(__file__), "corpus.txt")
    with open(corpus_path, "r") as f:
        corpus = f.read()

    tokenizer = Tokenizer(corpus)
    print(f"Vocab size: {tokenizer.vocab_size}")

    sample = corpus[:100]
    encoded = tokenizer.encode(sample)
    decoded = tokenizer.decode(encoded)

    print(f"\nOriginal ({len(sample)} chars):\n{sample}")
    print(f"\nEncoded ({len(encoded)} tokens):\n{encoded}")
    print(f"\nDecoded:\n{decoded}")
    print(f"\nRoundtrip match: {sample == decoded}")