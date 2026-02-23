import mlx.core as mx


class Embedding:
    def __init__(self, vocab_size: int, embedding_dim: int, weight: mx.array):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.weight = weight

    def __call__(self, x: mx.array) -> mx.array:
        return self.weight[x]

    # weight: vocab_size x embedding_dim
    # Input: N.. x embedding_dim
    # Output: N.. x vocab_size
    def as_linear(self, x: mx.array) -> mx.array:
        return mx.matmul(x, self.weight.T)
