"""Write test/tokenizer.fixtures.json from Python's tiktoken (the model's tokenizer).

    ../.venv/bin/python scripts/make_tokenizer_fixtures.py

The web test compares js-tiktoken's ids with these strings, so the JS tokenizer is
checked against the implementation the model was trained with, not against itself.
"""

import json
from pathlib import Path

import tiktoken

STRINGS = [
    "",
    "Once upon a time",
    "Once upon a time,",
    " there was a little girl",
    "there was a little girl ",
    "  two  spaces  ",
    "\n\n",
    "\n\nOnce",
    "\tTabbed\t",
    "It's a test: can't, won't, we'll, I'm, you've, he'd",
    "Numbers: 0 1 42 1000 3.14 -7",
    "Hello, world! (punctuation) ... --- ###",
    "café, naïve, résumé",
    "東京 is a city in Japan",
    "Привет, мир",
    "مرحبا بالعالم",
    "emoji: 😀 🌍 🚀👩‍👩‍👧‍👦",
    "combining: e\u0301 a\u0308",
    "zero width: \u200b\u200b",
    "non-breaking\u00a0space",
    "line\r\nwindows\r\n",
    "quotes: “smart” ‘quotes’ — dash – en …",
    "The little dog ran. The little dog jumped. The little dog slept.",
    "Once upon a time there was a tiny dragon who could not fly.",
    "a" * 200,
    "    ",
    "word\nword",
    "\\backslash /slash",
    "𝕳𝖊𝖑𝖑𝖔",
    "nul\u0000char",
    "One day, Timmy found a shiny red ball. He played with it until dinner.",
]


def main():
    enc = tiktoken.get_encoding("gpt2")
    out = [
        {
            "text": s,
            "tokens": enc.encode_ordinary(s),
            "decoded": enc.decode(enc.encode_ordinary(s)),
        }
        for s in STRINGS
    ]
    path = Path(__file__).resolve().parent.parent / "test" / "tokenizer.fixtures.json"
    path.write_text(json.dumps(out, ensure_ascii=True, indent=1) + "\n")
    print(f"Wrote {path} ({len(out)} strings)")


if __name__ == "__main__":
    main()
