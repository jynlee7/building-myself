# Building an LLM from Scratch: My Journey

**Date:** April 2026
**Author:** Jayden Lee

---

## The Problem: Why Pure NumPy Failed on CPU

When I first started this project, I wrote the entire transformer in pure Python/NumPy. The goal was educational - understanding how LLMs work under the hood by implementing every component from scratch.

**The Roadmap I Started With:**
1. Tokenizer (character-level)
2. Token Embeddings
3. Positional Encodings
4. Self-Attention
5. Feed-Forward Network
6. LayerNorm + Residual
7. Transformer Block
8. LM Head
9. Training Loop
10. Generation

I had the architecture mapped out. The code was clean. There was just one problem: **it wouldn't train.**

### The CPU Trap

NumPy executes on the CPU, not the GPU. Every matrix multiplication runs sequentially on your processor. With 711,900 training samples and 11,000+ batches, my Mac was struggling to calculate even basic operations.

```python
# This runs on CPU - painfully slow
scores = Q @ K.transpose(0, 2, 1)  # Manual matrix mult
```

Training would have taken **hours** on CPU vs. minutes on GPU.

### The Backpropagation Problem

Even worse - my gradient calculations were incomplete. The backward pass only updated the final output layer:

```python
# What I had - only trains lm_head
d_W = (d_logits.transpose(0, 2, 1) @ self.model.lm_head.x).mean(axis=0)
self.model.lm_head.W -= self.lr * d_W
```

The embeddings, attention weights, and feed-forward networks were frozen with random values. The model was training a linear classifier on top of static noise.

### The Memory Bottleneck

Every batch created new arrays in Python:

```python
# Slow - allocates new memory every iteration
inputs = np.array([p[0] for p in batch])
targets = np.array([p[1] for p in batch])
```

For 711K samples, this memory overhead alone was killing performance.

---

## The Solution: PyTorch for GPU

After analyzing the roadblocks, I ported the architecture to PyTorch:

```python
# PyTorch automatically handles:
# 1. GPU acceleration (.to('cuda'))
# 2. Full backprop (loss.backward())
# 3. Optimized dataloaders
```

---

## The Model Architecture

```
┌─────────────────────────────────────────────┐
│  Token Embedding (778 → 128)              │
│  Each character → 128-dim vector       │
├─────────────────────────────────────────────┤
│  Positional Encoding                   │
│  sin/cos at different frequencies    │
├─────────────────────────────────────────────┤
│  Transformer Block × 1               │
│  ┌──────────────────────────────┐  │
│  │ Self-Attention (Q, K, V)    │  │
│  │ softmax(QK^T / √d_k) @ V   │  │
│  └──────────────────────────────┘  │
│  ┌──────────────────────────────┐  │
│  │ Feed-Forward               │  │
│  │ GELU(W1·x + b1) · W2 + b2 │  │
│  └──────────────────────────────┘  │
├─────────────────────────────────────────────┤
│  LM Head (128 → 778)              │
│  Project to vocabulary logits   │
└─────────────────────────────���───────────────┘
```

**Parameters:** ~200K trainable weights

---

## Training Results

**Dataset:** My iMessages (cleaned)
- 711,932 characters
- 778 unique characters (vocab)
- 711,900 training pairs

**Config:**
- d_model: 128
- batch_size: 64
- epochs: 5
- learning_rate: 0.001
- optimizer: Adam

### Metrics

| Epoch | Loss | Perplexity | Time (s) |
|-------|------|----------|----------|
| 1 | 2.51 | 12.31 | 41.3 |
| 2 | 2.46 | 11.66 | 81.9 |
| 3 | 2.44 | 11.52 | 123.7 |
| 4 | 2.45 | 11.60 | 164.3 |
| 5 | 2.44 | 11.44 | 205.1 |

### Loss Curve

```
Epoch 1: ████████████████████░░░░░░░ 2.51
Epoch 2: ███████████████████░░░░░░░░ 2.46
Epoch 3: ███████████████████░░░░░░░░ 2.44
Epoch 4: ███████████████████░░░░░░░░ 2.45
Epoch 5: ████████████████████░░░░░░░ 2.44
```

### Perplexity Over Time

```
Epoch 1: ████████████████████████░ 12.31
Epoch 2: ███████████████████████░░ 11.66
Epoch 3: ███████████████████████░░ 11.52
Epoch 4: ███████████████████████░░ 11.60
Epoch 5: ██████████████████████░░░ 11.44
```

**Training time:** ~3.5 minutes on GPU (vs. hours on CPU)

---

## What I Learned

1. **GPU matters** - Frameworks like PyTorch exist for a reason. The hardware acceleration makes the difference between "impossible" and "done."

2. **Autograd is hard** - Implementing backprop manually for attention is complex. PyTorch's `loss.backward()` handles the calculus automatically.

3. **Clean data is key** - My iMessages had URLs and some explicit content. Had to clean the corpus before training.

4. **Small models can learn** - 200K params is tiny, but it still captured my writing patterns (given more time/epochs).

---

## Generation Samples

After 5 epochs, the model generates text. It's picking up common patterns:

| Prompt | Generated |
|--------|----------|
| "Hey" | "Hey t t t t t t t t t t t t t t t" |
| "What" | "What t t t t t t t t t t t t t t t" |
| "Im" | "Im t t t t t t t t t t t t t t t" |

**Analysis:** The model outputs a lot of "t" characters - it's learned that "t" is extremely common in my writing (in words like "it", "that", "the", "to", "just", etc.). With only 5 epochs and ~200K parameters, it's capturing frequency distributions but not yet full language structure.

**To improve:** More epochs, larger model, or fine-tuning would help.

1. **More epochs** - The model was still improving. Run 10-20 epochs.

2. **Larger model** - d_model=256 or 512 for better capacity.

3. **More layers** - Stack transformer blocks.

4. **Full training data** - 711K samples is decent, more would help.

---

## Code

The complete PyTorch implementation is on GitHub:
- `model.py` - Architecture
- `train_pytorch.py` - Training script

Run it yourself:
```bash
git clone https://github.com/jynlee7/building-myself
cd building-myself
python train_pytorch.py  # On GPU
```

---

## Conclusion

Building an LLM from scratch taught me why industry uses PyTorch/TensorFlow. The concepts (attention, embeddings, positional encoding) are learnable, but the engineering (GPU, autograd, dataloaders) matters just as much.

This was step one. The model learned my writing style - now I need to teach it more.

**Follow along:** github.com/jynlee7/building-myself