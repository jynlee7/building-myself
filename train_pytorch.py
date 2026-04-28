"""
Transformer Language Model - PyTorch Version
For GPU training on Google Colab
"""
import torch
import torch.nn as nn
import time
import json
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

    def encode(self, text: str):
        return [self.vocab[ch] for ch in text if ch in self.vocab]

    def decode(self, ids):
        return ''.join(self.id_to_token[i] for i in ids if i in self.id_to_token)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)


class Embedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, token_ids):
        return self.embedding(token_ids)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        i = torch.arange(0, d_model, 2)
        div_term = torch.pow(10000.0, 2 * i / d_model)
        pe[:, 0::2] = torch.sin(position / div_term)
        pe[:, 1::2] = torch.cos(position / div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        seq_len = x.shape[1]
        return x + self.pe[:, :seq_len]


class SelfAttention(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)

    def forward(self, x):
        return self.W_Q(x), self.W_K(x), self.W_V(x)


def attention(Q, K, V):
    d_k = Q.shape[-1]
    scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k))
    attn_weights = torch.softmax(scores, dim=-1)
    return torch.matmul(attn_weights, V), attn_weights


class FeedForward(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        d_ff = 4 * d_model
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.fc2(torch.nn.functional.gelu(self.fc1(x)))


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = SelfAttention(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model)

    def forward(self, x):
        Q, K, V = self.attn(x)
        attn_out, _ = attention(Q, K, V)
        x = self.ln1(x + attn_out)
        ffn_out = self.ffn(x)
        x = self.ln2(x + ffn_out)
        return x


class LMHead(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return self.linear(x)


class TransformerLanguageModel(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 128, max_len: int = 512):
        super().__init__()
        self.d_model = d_model
        self.embedding = Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.transformer = TransformerBlock(d_model)
        self.lm_head = LMHead(d_model, vocab_size)

    def forward(self, token_ids):
        x = self.embedding(token_ids) * np.sqrt(self.d_model)
        x = self.pos_encoding(x)
        x = self.transformer(x)
        return self.lm_head(x)

    def predict(self, token_ids):
        logits = self.forward(token_ids)
        return logits[:, -1, :]

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())


def create_training_pairs(text, tokenizer, seq_len=32):
    token_ids = tokenizer.encode(text)
    pairs = []
    for i in range(len(token_ids) - seq_len):
        input_seq = token_ids[i:i + seq_len]
        target_seq = token_ids[i + 1:i + seq_len + 1]
        pairs.append((input_seq, target_seq))
    return pairs


def train_model(model, training_pairs, epochs=5, batch_size=64, lr=0.001, device='cuda'):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    metrics = {"epoch": [], "loss": [], "perplexity": [], "time": [], "samples": []}
    n_samples = len(training_pairs)
    print(f"Training on {n_samples} samples, batch_size={batch_size}, epochs={epochs}")
    
    start_time = time.time()
    
    for epoch in range(epochs):
        np.random.shuffle(training_pairs)
        epoch_loss = 0
        n_batches = 0
        
        for i in range(0, n_samples, batch_size):
            batch = training_pairs[i:i + batch_size]
            if len(batch) < batch_size // 2:
                continue
            
            inputs = torch.tensor([p[0] for p in batch], dtype=torch.long).to(device)
            targets = torch.tensor([p[1] for p in batch], dtype=torch.long).to(device)
            
            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        avg_loss = epoch_loss / max(n_batches, 1)
        ppl = np.exp(avg_loss)
        elapsed = time.time() - start_time
        
        metrics["epoch"].append(epoch + 1)
        metrics["loss"].append(float(avg_loss))
        metrics["perplexity"].append(float(ppl))
        metrics["time"].append(elapsed)
        metrics["samples"].append(n_samples * (epoch + 1))
        
        print(f"Epoch {epoch + 1}/{epochs} | Loss: {avg_loss:.4f} | Perplexity: {ppl:.2f} | Time: {elapsed:.1f}s")
    
    return metrics


def generate_text(model, tokenizer, prompt, max_new_tokens=50, device='cuda'):
    model.eval()
    token_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long).to(device)
    generated = list(token_ids[0].cpu().numpy())

    with torch.no_grad():
        for _ in range(max_new_tokens):
            if len(generated) > 128:
                generated = generated[-128:]
            input_ids = torch.tensor([generated], dtype=torch.long).to(device)
            logits = model.predict(input_ids)
            next_token = torch.argmax(logits, dim=-1).item()

            if next_token == 0 and len(generated) > 10:
                break
            generated.append(next_token)

    return tokenizer.decode(generated)


if __name__ == "__main__":
    corpus_path = "training_data.txt"
    with open(corpus_path, "r") as f:
        corpus = f.read()
    
    print(f"Loaded corpus: {len(corpus)} characters")
    
    tokenizer = Tokenizer(corpus)
    print(f"Vocab size: {tokenizer.vocab_size}")
    
    seq_len = 32
    training_pairs = create_training_pairs(corpus, tokenizer, seq_len)
    print(f"Training pairs: {len(training_pairs)}")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    d_model = 128
    model = TransformerLanguageModel(vocab_size=tokenizer.vocab_size, d_model=d_model)
    print(f"Model: d_model={d_model}, params={model.num_params:,}")
    
    print("\nStarting training...")
    metrics = train_model(model, training_pairs, epochs=5, batch_size=64, device=device)
    
    with open("training_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to training_metrics.json")
    
    test_prompts = ["Hey", "What", "Im"]
    print("\nGeneration samples:")
    for prompt in test_prompts:
        generated = generate_text(model, tokenizer, prompt, max_new_tokens=30, device=device)
        print(f"  '{prompt}' -> '{generated}'")