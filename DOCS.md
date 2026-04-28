"""
BLOG.md - Full Project Documentation
=================================
Building an LLM from Scratch: My Journey

This file documents the complete project from idea to trained model.

SECTIONS:
--------
1. Motivation - Why build an LLM from scratch
2. The Problem - Why pure NumPy failed on CPU
3. The Solution - Porting to PyTorch
4. Architecture - Model components
5. Training - Metrics and results
6. Generation - Output samples
7. What I Learned - Key takeaways
8. Next Steps - Future improvements

FILES:
------
- tokenizer.py      - Character-level tokenizer (778 vocab)
- model.py        - PyTorch transformer architecture
- train_pytorch.py - Training script with metrics logging
- training_metrics.json - Training results (epochs 1-5)
- training_data.txt - Cleaned iMessages corpus
- BLOG.md        - This documentation
- AGENTS.md      - Agent instructions

KEY FACTS:
--------
- Dataset: My iMessages (cleaned before training)
- Vocab size: 778 unique characters
- Training samples: 711,900 pairs
- Model params: ~200K
- Training time: ~3.5 min on GPU
- Final perplexity: 11.44 (epoch 5)

METRICS (training_metrics.json):
--------------------------
Epoch | Loss   | Perplexity | Time
------|-------|----------|-------
  1   | 2.51  | 12.31    | 41s
  2   | 2.46  | 11.66    | 82s
  3   | 2.44  | 11.52    | 124s
  4   | 2.45  | 11.60    | 164s
  5   | 2.44  | 11.44    | 205s

ROADBLOCKS ENCOUNTERED:
----------------
1. CPU too slow for NumPy (~hours vs. minutes on GPU)
2. Manual backprop only updated final layer
3. Memory allocation overhead in training loop
4. Had to clean corpus ( URLs, explicit content)

LESSONS LEARNED:
-------------
1. PyTorch/TensorFlow exist for good reason (GPU + autograd)
2. Full backprop through attention is complex
3. Clean data before training
4. Small models CAN learn (given enough epochs)

TO RUN:
------
# On GPU (Colab)
git clone https://github.com/jynlee7/building-myself
cd building-myself
python train_pytorch.py  # Runs on CUDA if available

# Generate text
from model import *
model = TransformerLanguageModel(vocab_size=778, d_model=128)
generated = generate_text(model, tokenizer, "Hey", max_new_tokens=30)
print(generated)
"""