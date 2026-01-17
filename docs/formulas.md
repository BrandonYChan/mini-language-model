## Token Embeddings

Each token embedding is a unique vector of $d_{model}$ random values.

$$TE_i \sim U(-1, 1) \quad \text{for i = 1, ..., }d_{model}$$

## Position Encodings

Each positional encoding is a vector of length $d_{model}$.


$$PE(pos, 2i) = \sin\left(\frac{pos}{10000^{\frac{2i}{d_{model}}}}\right)$$

$$PE(pos, 2i + 1) = \cos\left(\frac{pos}{10000^{\frac{2i}{d_{model}}}}\right)$$

## QKV Initialization

$$W_{ij} \sim U\left(-\sqrt{\frac{6}{d_{model}+d_k}}, \sqrt{\frac{6}{d_{model}+d_k}}\right)$$

$$d_k = \frac{d_{model}}{\text{num heads}}$$

## Softmax Activation Function

$$softmax(x_i) = \frac{e^{x_i}}{\sum_{j=1}^n{e^{x_j}}}$$

## Attention Function

$$Attention(Q, K, V) = softmax\left(\frac{QK^T}{\sqrt{d_{model}}}\right)V$$