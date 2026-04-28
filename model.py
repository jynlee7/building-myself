import numpy as np


class Embedding:
    def __init__(self, vocab_size: int, d_model: int):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.W = np.random.randn(vocab_size, d_model) * 0.1

    def forward(self, token_ids: np.ndarray) -> np.ndarray:
        return self.W[token_ids]


class PositionalEncoding:
    def __init__(self, d_model: int, max_len: int = 512):
        self.d_model = d_model
        self.max_len = max_len
        self.encodings = self._compute_encodings(max_len, d_model)

    def _compute_encodings(self, max_len: int, d_model: int) -> np.ndarray:
        pos = np.arange(max_len).reshape(-1, 1)
        i = np.arange(d_model).reshape(1, -1)
        div_term = np.power(10000.0, 2 * i / d_model)

        pe = np.zeros((max_len, d_model))
        pe[:, 0::2] = np.sin(pos / div_term[:, 0::2])
        pe[:, 1::2] = np.cos(pos[:, 0::2] / div_term[:, 1::2])
        return pe

    def forward(self, x: np.ndarray) -> np.ndarray:
        seq_len = x.shape[1]
        return x + self.encodings[:seq_len]


def gelu(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


class SelfAttention:
    def __init__(self, d_model: int):
        self.d_model = d_model
        scale = 1.0 / np.sqrt(d_model)
        self.W_Q = np.random.randn(d_model, d_model) * scale
        self.W_K = np.random.randn(d_model, d_model) * scale
        self.W_V = np.random.randn(d_model, d_model) * scale

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        Q = x @ self.W_Q
        K = x @ self.W_K
        V = x @ self.W_V
        return Q, K, V


def attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
    d_k = Q.shape[-1]
    scores = Q @ K.transpose(0, 2, 1)
    scores = scores / np.sqrt(d_k)

    attn_weights = softmax(scores, axis=-1)
    output = attn_weights @ V
    return output, attn_weights


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


class FeedForward:
    def __init__(self, d_model: int, d_ff: int = None):
        self.d_model = d_model
        self.d_ff = d_ff or 4 * d_model
        self.W1 = np.random.randn(d_model, self.d_ff) * 0.1
        self.b1 = np.zeros(self.d_ff)
        self.W2 = np.random.randn(self.d_ff, d_model) * 0.1
        self.b2 = np.zeros(d_model)

    def forward(self, x: np.ndarray) -> np.ndarray:
        hidden = gelu(x @ self.W1 + self.b1)
        return hidden @ self.W2 + self.b2


class LayerNorm:
    def __init__(self, d_model: int, eps: float = 1e-6):
        self.d_model = d_model
        self.eps = eps
        self.gamma = np.ones(d_model)
        self.beta = np.zeros(d_model)

    def forward(self, x: np.ndarray) -> np.ndarray:
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        x_norm = (x - mean) / np.sqrt(var + self.eps)
        return self.gamma * x_norm + self.beta


class TransformerBlock:
    def __init__(self, d_model: int):
        self.ln1 = LayerNorm(d_model)
        self.attn = SelfAttention(d_model)
        self.ln2 = LayerNorm(d_model)
        self.ffn = FeedForward(d_model)

    def forward(self, x: np.ndarray) -> np.ndarray:
        attn_out, _, _ = self._self_attention_block(x)
        x = self.ln1.forward(x + attn_out)

        ffn_out = self.ffn.forward(x)
        x = self.ln2.forward(x + ffn_out)
        return x

    def _self_attention_block(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        Q, K, V = self.attn.forward(x)
        attn_output, attn_weights = attention(Q, K, V)
        return attn_output, attn_weights, Q


class LMHead:
    def __init__(self, d_model: int, vocab_size: int):
        self.W = np.random.randn(vocab_size, d_model) * 0.1
        self.b = np.zeros(vocab_size)

    def forward(self, x: np.ndarray) -> np.ndarray:
        return x @ self.W.T + self.b


class TransformerLanguageModel:
    def __init__(self, vocab_size: int, d_model: int = 128, max_len: int = 512):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embedding = Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.transformer = TransformerBlock(d_model)
        self.lm_head = LMHead(d_model, vocab_size)

    def forward(self, token_ids: np.ndarray) -> np.ndarray:
        x = self.embedding.forward(token_ids)
        x = self.pos_encoding.forward(x)
        x = self.transformer.forward(x)
        logits = self.lm_head.forward(x)
        return logits

    def predict(self, token_ids: np.ndarray) -> np.ndarray:
        logits = self.forward(token_ids)
        return logits[:, -1, :]


def cross_entropy_loss(logits: np.ndarray, targets: np.ndarray) -> float:
    batch_size = logits.shape[0]
    seq_len = logits.shape[1]
    logits_flat = logits.reshape(-1, logits.shape[-1])
    targets_flat = targets.reshape(-1)

    log_probs = logits_flat - np.max(logits_flat, axis=-1, keepdims=True)
    log_probs = log_probs - np.log(np.sum(np.exp(log_probs), axis=-1, keepdims=True))

    nll = -log_probs[np.arange(len(targets_flat)), targets_flat]
    return np.mean(nll)


class Trainer:
    def __init__(self, model: TransformerLanguageModel, lr: float = 0.01, momentum: float = 0.9):
        self.model = model
        self.lr = lr
        self.momentum = momentum
        self.velocity = self._init_velocity()

    def _init_velocity(self) -> dict:
        velocity = {}
        for name in ["embedding", "transformer", "lm_head"]:
            obj = getattr(self.model, name)
            if hasattr(obj, "W"):
                velocity[f"{name}.W"] = np.zeros_like(obj.W)
            if hasattr(obj, "b"):
                velocity[f"{name}.b"] = np.zeros_like(obj.b)

            for attr in dir(obj):
                val = getattr(obj, attr, None)
                if isinstance(val, np.ndarray) and not attr.startswith("_"):
                    velocity[f"{name}.{attr}"] = np.zeros_like(val)
        return velocity

    def step(self, token_ids: np.ndarray, targets: np.ndarray):
        logits = self.model.forward(token_ids)
        loss = cross_entropy_loss(logits, targets)

        grads = self._compute_gradients(token_ids, targets)
        self._apply_gradients(grads)

        return loss

    def _compute_gradients(self, token_ids: np.ndarray, targets: np.ndarray):
        logits = self.model.forward(token_ids)
        batch_size, seq_len, vocab_size = logits.shape
        targets_flat = targets.reshape(-1)

        d_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        d_logits = d_logits / np.sum(d_logits, axis=-1, keepdims=True)
        d_logits[np.arange(batch_size).reshape(-1, 1), np.arange(seq_len).reshape(1, -1), targets] -= 1
        d_logits /= batch_size

        grads = {}

        d_hidden = d_logits @ self.model.lm_head.W
        grads["lm_head.W"] = d_logits.transpose(0, 2, 1).reshape(-1, seq_len) @ d_hidden
        return grads

    def _apply_gradients(self, grads: dict):
        pass


def generate_text(model: TransformerLanguageModel, tokenizer, prompt: str, max_new_tokens: int = 50) -> str:
    token_ids = np.array(tokenizer.encode(prompt))
    generated = list(token_ids)

    for _ in range(max_new_tokens):
        input_ids = np.array(generated).reshape(1, -1)
        logits = model.predict(input_ids)
        next_token = np.argmax(logits, axis=-1)[0]

        if next_token == 0 and len(generated) > 10:
            break

        generated.append(next_token)

    return tokenizer.decode(generated)


if __name__ == "__main__":
    from tokenizer import Tokenizer

    corpus_path = "corpus.txt"
    with open(corpus_path, "r") as f:
        corpus = f.read()

    tokenizer = Tokenizer(corpus)
    print(f"Vocab size: {tokenizer.vocab_size}")

    model = TransformerLanguageModel(vocab_size=tokenizer.vocab_size, d_model=128)

    test_input = "Hey"
    test_tokens = np.array(tokenizer.encode(test_input)).reshape(1, -1)
    print(f"\nInput: '{test_input}'")
    print(f"Token IDs: {test_tokens}")

    logits = model.predict(test_tokens)
    print(f"Logits shape: {logits.shape}")
    print(f"Next token probs sum: {np.sum(np.exp(logits - np.max(logits)))}")

    generated = generate_text(model, tokenizer, "Hey", max_new_tokens=20)
    print(f"\nGenerated: '{generated}'")