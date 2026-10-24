# From Words to Embeddings, Attention, and Transformers

A visual, beginner-friendly lesson that starts from zero (how text becomes numbers) and builds up to a full Transformer block, with matrices, heatmaps and small runnable NumPy examples.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/<your-username>/embeddings-attention-transformers/blob/main/Embeddings_Attention_Transformers.ipynb)

## What it covers

| Part | Topics |
|---|---|
| Text to vectors | Tokens, token IDs, one-hot encoding, the embedding matrix, cosine similarity |
| How embeddings learn | One gradient update by hand, training a tiny embedding table, Word2Vec-style skip-gram, subword tokenizers, static vs contextual embeddings |
| Attention | Why attention is needed, softmax, cross-attention, Query / Key / Value, scaled dot-product attention |
| Self-attention | Attention heatmaps, why we divide by √d_k, positional encoding, causal masking |
| The Transformer | Multi-head attention, feed-forward network, residual connections and layer normalization, encoder, decoder and encoder-decoder models |
| Practice | Trace-one-token activity, common confusions, practice questions with answers, a final cheat sheet |

The core idea in one line:

```
Attention(Q, K, V) = softmax(Q Kᵀ / √d_k) V
```

## How to run

Only **NumPy** and **Matplotlib** are used, so it runs anywhere: no GPU, no model downloads and no API key.

- **Colab:** click **Open in Colab** above, then **Runtime → Run all**.
- **Locally:**

```bash
pip install numpy matplotlib jupyter
jupyter notebook Embeddings_Attention_Transformers.ipynb
```

## Reference

Vaswani et al., *Attention Is All You Need*, 2017. [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
