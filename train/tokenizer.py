"""
The tokenizer: turns text into numbers and back.

The computer can only do math on numbers, not letters. So we give every
character its own number. "a" might be 39, "b" might be 40, space might be 1.
  encode("hi") -> [47, 48]     text in, numbers out
  decode([47, 48]) -> "hi"     numbers in, text out

The "vocabulary" is just the list of every different character that shows up in
the training text, sorted. A character's number is simply where it sits in that
list. That's the whole idea.
"""


class CharTokenizer:
    def __init__(self, text: str):
        # grab every different character in the text, put them in a fixed order.
        # sorting matters so the same text always gives the same numbers.
        chars = sorted(set(text))
        self.vocab = chars
        self.vocab_size = len(chars)

        # two little lookup tables: one for char -> number, one for number -> char
        self.stoi = {ch: i for i, ch in enumerate(chars)}  # char to number
        self.itos = {i: ch for i, ch in enumerate(chars)}  # number to char

    def encode(self, text: str) -> list[int]:
        return [self.stoi[ch] for ch in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[i] for i in ids)


if __name__ == "__main__":
    # quick test so we can actually see what it does
    with open("data/tinyshakespeare.txt", "r") as f:
        text = f.read()

    tok = CharTokenizer(text)
    print(f"vocab_size = {tok.vocab_size}")
    print(f"vocabulary = {tok.vocab}")

    sample = "Hi there!"
    ids = tok.encode(sample)
    print(f"\nencode({sample!r}) = {ids}")
    print(f"decode(...)       = {tok.decode(ids)!r}")

    # the big check: if we turn text into numbers and back, we should get the
    # exact same text. if this ever fails, everything after it is broken.
    assert tok.decode(tok.encode(text)) == text, "round-trip failed!"
    print("\nround-trip over the full dataset: OK")
