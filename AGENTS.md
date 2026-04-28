# AGENTS.md

## Running the tokenizer

```bash
python tokenizer.py
```

## Running training (PyTorch on GPU required)

```bash
python train_pytorch.py
```

## Notes

- Character-level tokenizer from `corpus.txt` (778 vocab)
- PyTorch model on GPU for training (NumPy CPU too slow)
- Training generates `training_metrics.json` with loss/perplexity per epoch
- Generation via `model.py` or `train_pytorch.py`

## Blog

See `BLOG.md` for the full project documentation.