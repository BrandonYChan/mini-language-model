## Token Embeddings

Each token embedding is a unique vector of $d_{model}$ random values.

$$TE_i \sim U(-1, 1) \quad \text{for i = 1, ..., }d_{model}$$

## Position Encodings

Each positional encoding is a vector of length $d_{model}$.


$$PE(pos, 2i) = \sin\left(\frac{pos}{10000^{\frac{2i}{d_{model}}}}\right)$$

$$PE(pos, 2i + 1) = \cos\left(\frac{pos}{10000^{\frac{2i}{d_{model}}}}\right)$$