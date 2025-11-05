import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------
# Positional Encoding
# ---------------------------
def positional_encoding(seq_len, embedding_dim):
    """
    Compute standard sinusoidal positional encoding.

    Args:
        seq_len (int): Length of the input sequence.
        embedding_dim (int): Dimension of embedding vectors.

    Returns:
        Tensor of shape (seq_len, embedding_dim)
    """
    position = torch.arange(seq_len).unsqueeze(1)  # (seq_len, 1)
    div_term = torch.exp(torch.arange(0, embedding_dim, 2) * (-torch.log(torch.tensor(10000.0)) / embedding_dim))  # (embedding_dim/2,)

    pos_encoding = torch.zeros(seq_len, embedding_dim)
    pos_encoding[:, 0::2] = torch.sin(position * div_term)
    pos_encoding[:, 1::2] = torch.cos(position * div_term)
    return pos_encoding


# ---------------------------
# Single Transformer Encoder Layer
# ---------------------------
class TransformerEncoderLayer(nn.Module):
    """
    Implements one layer of a Transformer encoder, consisting of:
    1. Multi-head self-attention sublayer (with residual connection + LayerNorm)
    2. Feedforward sublayer (with residual connection + LayerNorm)
    """

    def __init__(self, embedding_dim, feedforward_dim, num_heads, dropout):
        super().__init__()
        assert embedding_dim % num_heads == 0, \
            "embedding_dim must be divisible by num_heads for even head splitting."

        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads

        # Linear projections for Query, Key, and Value
        self.query_projection = nn.Linear(embedding_dim, embedding_dim)
        self.key_projection = nn.Linear(embedding_dim, embedding_dim)
        self.value_projection = nn.Linear(embedding_dim, embedding_dim)

        # Output projection after concatenating all heads
        self.output_projection = nn.Linear(embedding_dim, embedding_dim)

        # Feedforward network (Position-wise MLP)
        self.feedforward_network = nn.Sequential(
            nn.Linear(embedding_dim, feedforward_dim),
            nn.ReLU(),
            nn.Linear(feedforward_dim, embedding_dim)
        )

        # Layer Normalization (applied after residual addition)
        self.layer_norm_1 = nn.LayerNorm(embedding_dim)
        self.layer_norm_2 = nn.LayerNorm(embedding_dim)

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_embeddings):
        """
        Args:
            input_embeddings (Tensor): shape (batch_size, seq_len, embedding_dim)
                Values should be 0 for positions to attend to, and -inf (or a large negative number)
                for masked positions.
        """
        batch_size, sequence_length, embedding_dim = input_embeddings.shape
        num_heads = self.num_heads
        head_dim = self.head_dim

        # ---------------------------
        # Multi-Head Self-Attention
        # ---------------------------
        Q = self.query_projection(input_embeddings).view(batch_size, sequence_length, num_heads, head_dim).transpose(1, 2)
        K = self.key_projection(input_embeddings).view(batch_size, sequence_length, num_heads, head_dim).transpose(1, 2)
        V = self.value_projection(input_embeddings).view(batch_size, sequence_length, num_heads, head_dim).transpose(1, 2)

        # Scaled dot-product attention
        attention_scores = (Q @ K.transpose(-2, -1)) / (head_dim ** 0.5)  # (B, H, S, S)

        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_output = attention_weights @ V  # (B, H, S, d_h)

        # Combine attention heads
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, sequence_length, embedding_dim)
        attention_output = self.output_projection(attention_output)

        # Add & Norm (with dropout)
        x = self.layer_norm_1(input_embeddings + self.dropout(attention_output))

        # ---------------------------
        # Feedforward Network
        # ---------------------------
        feedforward_output = self.feedforward_network(x)
        x = self.layer_norm_2(x + self.dropout(feedforward_output))

        return x  # (B, S, D)


# ---------------------------
# Transformer Encoder (stack of layers)
# ---------------------------
class TransformerEncoder(nn.Module):
    """
    A simple Transformer encoder consisting of:
    - Token embedding layer
    - Sinusoidal positional encoding
    - Stack of TransformerEncoderLayers
    """

    def __init__(self, embedding_dim, feedforward_dim, num_layers, num_heads, dropout=0.1):
        super().__init__()

        self.layers = nn.ModuleList([
            TransformerEncoderLayer(embedding_dim, feedforward_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, embeddings):
        """
        Args:
            embeddings (Tensor): shape (batch_size, seq_len) check what the shape of embeddings is!!
        """
        sequence_length = embeddings.size(1)

        # Add positional encoding
        positional_encodings = positional_encoding(sequence_length, embeddings.size(-1)).to(embeddings.device)
        embeddings = embeddings + positional_encodings.unsqueeze(0)  # Broadcast to batch

        x = self.dropout(embeddings)

        # Pass through each Transformer layer
        for layer in self.layers:
            x = layer(x)

        return x  # (B, S, D)
