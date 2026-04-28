"""
API Server for Text Generation
Run this on Render (Python/Flask service)
"""
from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

app = Flask(__name__)

# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ============== MODEL ARCHITECTURE ==============

class Embedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, token_ids):
        return self.embedding(token_ids)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
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
    def __init__(self, d_model):
        super().__init__()
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)

    def forward(self, x):
        return self.W_Q(x), self.W_K(x), self.W_V(x)


def attention(Q, K, V):
    d_k = Q.shape[-1]
    scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=Q.dtype))
    attn_weights = F.softmax(scores, dim=-1)
    return torch.matmul(attn_weights, V), attn_weights


class FeedForward(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        d_ff = 4 * d_model
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class TransformerBlock(nn.Module):
    def __init__(self, d_model):
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
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return self.linear(x)


class TransformerLanguageModel(nn.Module):
    def __init__(self, vocab_size, d_model=128, max_len=512):
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


# ============== TOKENIZER ==============

def build_vocab(corpus):
    unique_chars = sorted(set(corpus))
    return {ch: i for i, ch in enumerate(unique_chars)}, {i: ch for ch, i in zip(unique_chars, range(len(unique_chars)))}


def encode(text, vocab):
    return [vocab[ch] for ch in text if ch in vocab]


def decode(ids, id_to_token):
    return ''.join(id_to_token[i] for i in ids if i in id_to_token)


# ============== LOAD MODEL ==============

print("Loading model...")

# Check for saved model weights
if os.path.exists('model_weights.pt'):
    print("Loading from model_weights.pt...")
    checkpoint = torch.load('model_weights.pt', map_location=device)
    VOCAB = checkpoint['vocab']
    ID_TO_TOKEN = checkpoint['id_to_token']
    VOCAB_SIZE = len(VOCAB)
    D_MODEL = checkpoint.get('d_model', 128)
    
    model = TransformerLanguageModel(VOCAB_SIZE, D_MODEL)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    print(f"Model loaded from checkpoint")
else:
    print("Using fresh model (no weights saved)")
    # Load corpus from file
    try:
        with open('training_data.txt', 'r') as f:
            CORPUS = f.read()
    except:
        CORPUS = """Hey Whats up Im down to go"""
    
    VOCAB, ID_TO_TOKEN = build_vocab(CORPUS)
    VOCAB_SIZE = len(VOCAB)
    D_MODEL = 128

    model = TransformerLanguageModel(VOCAB_SIZE, D_MODEL)
    model = model.to(device)
    model.eval()

print(f"Model on {device}")
print(f"Vocab size: {VOCAB_SIZE}")


# ============== API ROUTES ==============

@app.route('/generate', methods=['POST'])
def generate():
    data = request.get_json()
    prompt = data.get('prompt', '')
    max_tokens = data.get('max_tokens', 50)
    temperature = data.get('temperature', 1.0)
    
    # Encode prompt
    token_ids = encode(prompt, VOCAB)
    if not token_ids:
        return jsonify({'error': 'No valid tokens in prompt'}), 400
    
    generated = list(token_ids)
    
    with torch.no_grad():
        for _ in range(max_tokens):
            if len(generated) > 128:
                generated = generated[-128:]
            
            input_ids = torch.tensor([generated], dtype=torch.long).to(device)
            logits = model.predict(input_ids)
            
            # Temperature scaling
            if temperature != 1.0:
                logits = logits / temperature
            
            next_token = torch.argmax(logits, dim=-1).item()
            
            if next_token == 0 and len(generated) > 10:
                break
            
            generated.append(next_token)
    
    result = decode(generated, ID_TO_TOKEN)
    
    return jsonify({
        'prompt': prompt,
        'generated': result,
        'max_tokens': max_tokens,
        'temperature': temperature
    })


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'device': device})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))