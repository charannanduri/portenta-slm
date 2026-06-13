"""
Character-level tokenizer, built from scratch.

A tokenizer answers two questions:
  encode(text)  -> list of integer IDs   (so the model can do math on text)
  decode(ids)   -> text                   (so we can read the model's output)

For char-level, the "vocabulary" is just the sorted set of unique
characters that appear in our training data. Each character maps to one
integer ID (its index in that sorted list). That's the whole trick.
"""


class CharTokenizer:
    def __init__(self, text: str):
        # The vocabulary: every unique character, in a stable sorted order.
        # Sorting matters so the same text always produces the same IDs.
        chars = sorted(set(text))
        self.vocab = chars
        self.vocab_size = len(chars)

        # Two lookup tables: char->id (for encoding) and id->char (for decoding).
        self.stoi = {ch: i for i, ch in enumerate(chars)}  # "string to int"
        self.itos = {i: ch for i, ch in enumerate(chars)}  # "int to string"

    def encode(self, text: str) -> list[int]:
        return [self.stoi[ch] for ch in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[i] for i in ids)


if __name__ == "__main__":
    # Quick self-test so we can SEE what the tokenizer does.
    with open("data/tinyshakespeare.txt", "r") as f:
        text = f.read()

    tok = CharTokenizer(text)
    print(f"vocab_size = {tok.vocab_size}")
    print(f"vocabulary = {tok.vocab}")

    sample = "Hi there!"
    ids = tok.encode(sample)
    print(f"\nencode({sample!r}) = {ids}")
    print(f"decode(...)       = {tok.decode(ids)!r}")

    # The most important sanity check in all of tokenization:
    # decoding what we encoded must give back the original text exactly.
    assert tok.decode(tok.encode(text)) == text, "round-trip failed!"
    print("\nround-trip over the full dataset: OK")
