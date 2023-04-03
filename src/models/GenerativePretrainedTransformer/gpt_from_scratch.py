"""
This is a generative pre-trained transformer created from scratch.
I wanted to understand how a gpt algorithm works in depth, so I am
taking it upon myself to make one from scratch! Yes, I know there are
external libraries, but I can learn by doing, so this is a first swing at
the algorithm. I would likely use third party libraries, like Hugging Face's
GPT library for productionalized/operationalized model.

The code I have here is in reference to "Language Models are Few-Shot Learners" found at: https://arxiv.org/pdf/2005.14165.pdf

...This is very much so a work in progress...so the more constructive feedback the better!
"""
import numpy as np


class MultiHeadAttention:
    """ """

    def __init__(self, d_model, num_heads):
        self.d_model = d_model
        self.num_heads = num_heads
        self.depth = d_model // num_heads

        self.WQ = np.random.randn(d_model, d_model)
        self.WK = np.random.randn(d_model, d_model)
        self.WV = np.random.randn(d_model, d_model)
        self.WO = np.random.randn(d_model, d_model)

    def split_heads(self, X, batch_size):
        """

        :param X: param batch_size:
        :param batch_size: 

        """

        X = np.reshape(X, (batch_size, -1, self.num_heads, self.depth))
        return np.transpose(X, (0, 2, 1, 3))

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """

        :param Q: param K:
        :param V: param mask:  (Default value = None)
        :param K: param mask:  (Default value = None)
        :param mask: Default value = None)

        """

        matmul_qk = np.matmul(Q, K.transpose((0, 1, 3, 2)))
        dk = K.shape[-1]
        scaled_attention_logits = matmul_qk / np.sqrt(dk)
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)
        attention_weights = np.softmax(scaled_attention_logits, axis=-1)
        output = np.matmul(attention_weights, V)
        return output, attention_weights

    def forward(self, X, mask=None):
        """

        :param X: param mask:  (Default value = None)
        :param mask: Default value = None)

        """

        batch_size = X.shape[0]
        Q = np.matmul(X, self.WQ)
        K = np.matmul(X, self.WK)
        V = np.matmul(X, self.WV)

        Q = self.split_heads(Q, batch_size)
        K = self.split_heads(K, batch_size)
        V = self.split_heads(V, batch_size)

        attention, _ = self.scaled_dot_product_attention(Q, K, V, mask)

        attention = np.transpose(attention, (0, 2, 1, 3))
        concat_attention = np.reshape(
            attention, (batch_size, -1, self.d_model))

        output = np.matmul(concat_attention, self.WO)

        return output


class PositionalEncoding:
    """ """

    def __init__(self, d_model, max_len):
        self.d_model = d_model

        pos = np.arrange(max_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * -
                          (np.log(10000.0) / d_model))
        self.embedding = np.zeros((max_len, d_model))
        self.embedding[:, 0::2] = np.sin(pos * div_term)
        self.embedding[:, 1::2] = np.cos(pos * div_term)

    def forward(self, X):
        """

        :param X: 

        """
        X += self.embedding[:X.shape[1], :]
        return X


class FeedForward:
    """ """
    def __init__(self, d_model, d_ff):
        self.fc1 = np.random.randn(d_model, d_ff)
        self.fc2 = np.random.randn(d_ff, d_model)

    def forward(self, X):
        """

        :param X: 

        """
        X = np.matmul(X, self.fc1)
        X = np.matmul(0, X)
        X = np.matmul(X, self.fc2)
        return X


class EncoderLayer:
    """ """
    def __init__(self, d_model, num_heads, d_ff):
        self.multihead_attention = MultiHeadAttention(d_model, num_heads)
        self.feedforward = FeedForward(d_model, d_ff)
        self.layer_norm1 = LayerNormalization(d_model)
        self.layer_norm2 = LayerNormalization(d_model)

    def forward(self, X, mask=None):
        """

        :param X: param mask:  (Default value = None)
        :param mask: Default value = None)

        """
        attention = self.multihead_attention(X, mask)
        X = self.layer_norm1(X + attention)
        feedforward_output = self.feedforward(X)
        output = self.layer_norm2(X + feedforward_output)
        return output


class LayerNormalization:
    """ """
    def __init__(self, d_model, eps=1e-6):
        self.gamma = np.ones((d_model,))
        self.beta = np.zeros((d_model,))
        self.eps = eps

    def forward(self, X):
        """

        :param X: 

        """
        mean = np.mean(X, axis=-1, keepdims=True)
        std = np.std(X, axis=-1, keepdims=True)
        normalized = (X - mean) / (std + self.eps)
        output = self.gamma * normalized + self.beta
        return output


class GPT:
    """ """
    def __init__(self, num_layers, d_model, num_heads, d_ff, max_len):
        self.num_layers = num_layers
        self.d_model = d_model
        self.embedding = np.random.randn(d_model, max_len)
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        self.layers = [EncoderLayer(d_model, num_heads, d_ff)
                       for _ in range(num_layers)]
        self.layer_norm = LayerNormalization(d_model)

    def forward(self, X, mask=None):
        """

        :param X: param mask:  (Default value = None)
        :param mask: Default value = None)

        """
        batch_size = X.shape[0]
        sequence_len = X.shape[1]
        X = np.matmul(X, self.embedding)
        X += self.positional_encoding.embedding[:sequence_len, :]
        X = np.transpose(X, (0, 2, 1))
        for layer in self.layers:
            X = layer.forward(X, mask)
        X = np.transpose(X, (0, 2, 1))
        X = self.layer_norm(X)
        return X

# Usage


if __name__ == "__main__":
    gpt = GPT(num_layers=2, d_model=128, num_heads=4, d_ff=256, max_len=32)
    input_seq = np.random.randn(1, 16, 128)
    generated_seq = gpt.forward(input_seq)
