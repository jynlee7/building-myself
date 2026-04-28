# AGENTS.md

## Running the tokenizer

```bash
python tokenizer.py
```

This runs the tokenizer on `corpus.txt` and prints vocab size, a sample encode/decode, and verifies roundtrip.

## Notes

- Simple character-level tokenizer (one token = one unique character)
- No external dependencies required (numpy imported but not used in main path)
- Corpus contains chat/message data used for testing encode/decode roundtrip