"""
Transformer models and variants

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
"""


import math
import torch
import torch.nn.functional as F

from torch import nn, Tensor
from typing import List, Tuple, Optional, Union

class SelfAttention(nn.Module):
    """
    Attention layer that computes the scaled dot-product attention

    Parameters:
    -----------
    hidden_size: int, default: 512
        Size of the hidden state embeddings
    num_heads: int, default: 8
        Number of heads
    dropout_prob: float, default: 0.1
        Dropout probability for attention weights
    return_weights: bool, default: False
        If True, return the self attention weights
    """
    def __init__(self,
        hidden_size: int = 512,
        num_heads: int = 8,
        dropout_prob: float = 0.1,
        return_weights: bool = False
    ) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"The given hidden size {hidden_size} should be divisible by "
                f"the number of attention heads {num_heads}."
            )
        self.sub_hidden_size = int(hidden_size / num_heads)
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.return_weights = return_weights

        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)

        self.dropout = nn.Dropout(dropout_prob)

    def reshape(self, x: Tensor) -> Tensor:
        new_x_shape = x.size()[:-1] + (
            self.num_heads,
            self.sub_hidden_size
        )
        x = x.view(*new_x_shape)
        return x.permute(1, 0, 2)

    def forward(self,
        x: Tensor, attn_mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Parameters:
        -----------
        x: Tensor
            (N, K) K-dimensional embeddings for N instances
        attn_mask: Tensor, optional
            (1, N, N) Binary attention mask
        
        Returns:
        --------
        m_r: Tensor
            (N, K) K-dimensional weighted messages (values) for N instances
        attn_data: Tensor, Optional
            If required, return self attention weights. Otherwise return None.
        """
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        q_r = self.reshape(q)
        k_r = self.reshape(k)
        v_r = self.reshape(v)

        # Compute the scaled dot-product attention
        # For tensors with more than two dimensions, batched matrix multiplication
        # is applied, where the last two dimensions constitute the matrices
        weights = torch.matmul(q_r, k_r.transpose(-1, -2))
        weights = weights / math.sqrt(self.sub_hidden_size)
        # Zero out post-softmax probabilities according to the mask
        if attn_mask is not None:
            weights = weights.masked_fill(attn_mask == 0, -1e9)
        # Normalise the attention weights
        weights = F.softmax(weights, dim=-1)
        # Dropping out entire tokens to attend to
        weights = self.dropout(weights)

        # Compute messages (or contextualised embeddings)
        m = torch.matmul(weights, v_r)
        # Permute and reshape
        m_p = m.permute(1, 0, 2).contiguous()
        new_m_shape = m_p.size()[:-2] + (self.hidden_size,)
        m_r = m_p.view(*new_m_shape)

        if self.return_weights:
            attn_data = weights
        else:
            attn_data = None

        return m_r, attn_data

class SelfAttentionOutput(nn.Module):
    """
    Output layer for self attention that aggregates all heads and update
    the hidden state of each token

    Parameters:
    -----------
    hidden_size: int, default: 512
        Size of the hidden state embeddings
    dropout_prob: float, default: 0.1
        Dropout probability for attention weights
    """
    def __init__(self,
        hidden_size: int = 512,
        dropout_prob: float = 0.1
    ) -> None:
        super().__init__()
        self.aggregate = nn.Linear(hidden_size, hidden_size)
        self.norm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, m: Tensor, x: Tensor) -> Tensor:
        """
        Parameters:
        -----------
        m: Tensor
            (N, K) K-dimensional weighted messages (values) for N instances
        x: Tensor
            (N, K) K-dimensional embeddings for N instances
        
        Returns:
        --------
        x: Tensor
            (N, K) Updated embeddings
        """
        m = self.aggregate(m)
        m = self.dropout(m)
        x = self.norm(x + m)
        return x

class SelfAttentionLayer(nn.Module):
    """
    Multi-head self attention layer with update

    Parameters:
    -----------
    hidden_size: int, default: 512
        Size of the hidden state embeddings
    num_heads: int, default: 8
        Number of heads
    dropout_prob: float, default: 0.1
        Dropout probability for attention weights
    return_weights: bool, default: False
        If True, return the self attention weights
    """
    def __init__(self,
        hidden_size: int = 512,
        num_heads: int = 8,
        dropout_prob: float = 0.1,
        return_weights: bool = False
    ) -> None:
        super().__init__()
        self.attention = SelfAttention(
            hidden_size=hidden_size, num_heads=num_heads,
            dropout_prob=dropout_prob, return_weights=return_weights
        )
        self.output = SelfAttentionOutput(
            hidden_size=hidden_size, dropout_prob=dropout_prob
        )

    def forward(self,
        x: Tensor, attn_mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Parameters:
        -----------
        x: Tensor
            (N, K)  K-dimensional embeddings for N instances
        attn_mask: Tensor, optional
            (1, N, N) Binary attention mask
        
        Returns:
        --------
        x: Tensor
            (N, K) Updated embeddings
        attn_data: Tensor, optional
            If required, return self attention weights. Otherwise return None.
        """
        m, attn_data = self.attention(x, attn_mask)
        x = self.output(m, x)
        return x, attn_data

class FeedForwardNetwork(nn.Module):
    """
    Position-wise feed-forward networks succeeding the attention layer 

    Parameters:
    -----------
    hidden_size: int, default: 512
        Size of the hidden state embeddings
    intermediate-size: int, default: 2048
        Size of the intermediate embeddings
    dropout_prob: float, default: 0.1
        Dropout probability for attention weights
    """
    def __init__(self,
        hidden_size: int = 512,
        intermediate_size: int = 2048,
        dropout_prob: float = 0.1
    ) -> None:
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.GELU(),
            nn.Linear(intermediate_size, hidden_size)
        )
        self.dropout = nn.Dropout(dropout_prob)
        self.norm = nn.LayerNorm(hidden_size, eps=1e-12)
    def forward(self, x: Tensor) -> Tensor:
        y = self.ffn(x)
        y = self.dropout(y)
        x = self.norm(x + y)
        return x

class TransformerEncoderLayer(nn.Module):
    """
    Transformer encoder layer

    Parameters:
    -----------
    hidden_size: int, default: 512
        Size of the hidden state embeddings
    intermediate-size: int, default: 2048
        Size of the intermediate embeddings
    num_heads: int, default: 8
        Number of heads
    dropout_prob: float, default: 0.1
        Dropout probability for attention weights
    return_weights: bool, default: False
        If True, return the self attention weights
    """
    def __init__(self,
        hidden_size: int = 512,
        intermediate_size: int = 2048,
        num_heads: int = 8,
        dropout_prob: float = 0.1,
        return_weights: bool = False
    ) -> None:
        super().__init__()
        self.attention = SelfAttentionLayer(
            hidden_size=hidden_size,
            num_heads=num_heads,
            dropout_prob=dropout_prob,
            return_weights=return_weights
        )
        self.ffn = FeedForwardNetwork(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            dropout_prob=dropout_prob
        )

    def forward(self,
        x: Tensor, attn_mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, Optional[Tensor]]:
        x, attn_data = self.attention(x, attn_mask)
        x = self.ffn(x)
        return x, attn_data

class TransformerEncoder(nn.Module):
    """
    Transformer encoder

    Parameters:
    -----------
    hidden_size: int, default: 512
        Size of the hidden state embeddings
    intermediate-size: int, default: 2048
        Size of the intermediate embeddings
    num_heads: int, default: 8
        Number of heads
    num_layers: int, default: 6
        Number of encoder layers
    dropout_prob: float, default: 0.1
        Dropout probability for attention weights
    return_weights: bool, default: False
        If True, return the self attention weights
    """
    def __init__(self,
        hidden_size: int = 512,
        intermediate_size: int = 2048,
        num_heads: int = 8,
        num_layers: int = 6,
        dropout_prob: float = 0.1,
        return_weights: bool = False
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                num_heads=num_heads,
                dropout_prob=dropout_prob,
                return_weights=return_weights
            ) for _ in range(num_layers)
        ])
        self.num_layers = num_layers
        self.return_weights = return_weights

    def forward(self,
        x: Tensor, keep_history: bool = False,
        attn_mask: Optional[Union[Tensor, List[Tensor]]] = None
    ) -> Tuple[Tensor, List[Tensor], List[Tensor]]:
        """
        Parameters:
        -----------
        x: Tensor
            (N, K)  K-dimensional embeddings for N instances
        keep_history: bool
            If True, return outputs from all layers
        attn_mask: Tensor or List[Tensor], optional
            (N, N) Binary attention mask. When provided as a list of tensors,
            each tensor correspond to the mask for one layer. When provided as
            a single tensor, the mask will be used for all layers.

        Returns:
        --------
        x: Tensor
            (N, K) Updated embeddings
        history: List[Tensor]
            If required, return a list of outputs from each layer, otherwise
            return an empty list
        weights: List[Tensor]
            If required, return a list of self attention weights from each layer.
            Otherwise return an empty list
        """
        if attn_mask is None:
            masks = [None for _ in range(self.num_layers)]
        elif isinstance(attn_mask, Tensor):
            masks = [attn_mask.unsqueeze(0) for _ in range(self.num_layers)]
        else:
            masks = [m.unsqueeze(0) for m in attn_mask]

        history = []
        weights = []
        for i, layer in enumerate(self.layers):
            x, w = layer(x, attn_mask=masks[i])
            if keep_history:
                history.append(x)
            if self.return_weights:
                weights.append(w)

        return x, history, weights

class CrossAttention(SelfAttention):
    """Cross attention layer that computes one-directional messages"""
    def forward(self,
        x: Tensor, y: Tensor,
        attn_mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Parameters:
        -----------
        x: Tensor
            (N, K) K-dimensional embeddings for N receiving nodes
        y: Tensor
            (M, K) K-dimensional embeddings for M sending nodes
        attn_mask: Tensor, optional
            (1, M, N) Binary attention mask

        Returns:
        --------
        m_r: Tensor
            (N, K) K-dimensional weighted messages (values) for N instances
        attn_data: Tensor, Optional
            If required, return self attention weights. Otherwise return None.
        """
        q = self.query(x)
        k = self.key(y)
        v = self.value(y)

        q_r = self.reshape(q)
        k_r = self.reshape(k)
        v_r = self.reshape(v)

        # Compute cross attention weights
        weights = torch.matmul(q_r, k_r.transpose(-1, -2))
        weights = weights / math.sqrt(self.sub_hidden_size)
        # Zero out post-softmax probabilities according to the mask
        if attn_mask is not None:
            weights = weights.masked_fill(attn_mask == 0, -1e9)
        # Normalise the attention weights
        weights = F.softmax(weights, dim=-1)

        # Compute weighted messages
        m = torch.matmul(weights, v_r)
        # Permute and reshape
        m_p = m.permute(1, 0, 2).contiguous()
        new_m_shape = m_p.size()[:-2] + (self.hidden_size,)
        m_r = m_p.view(*new_m_shape)

        if self.return_weights:
            attn_data = weights
        else:
            attn_data = None

        return m_r, attn_data

class CrossAttentionLayer(nn.Module):
    """
    Multi-head cross attention layer with update

    Parameters:
    -----------
    hidden_size: int, default: 512
        Size of the hidden state embeddings
    num_heads: int, default: 8
        Number of heads
    dropout_prob: float, default: 0.1
        Dropout probability for attention weights
    return_weights: bool, default: False
        If True, return the self attention weights
    """
    def __init__(self,
        hidden_size: int = 512,
        num_heads: int = 8,
        dropout_prob: float = 0.1,
        return_weights: bool = False
    ) -> None:
        super().__init__()
        self.attention = CrossAttention(
            hidden_size=hidden_size, num_heads=num_heads,
            dropout_prob=dropout_prob, return_weights=return_weights
        )
        self.output = SelfAttentionOutput(
            hidden_size=hidden_size, dropout_prob=dropout_prob
        )

    def forward(self,
        x: Tensor, y: Tensor,
        attn_mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Parameters:
        -----------
        x: Tensor
            (N, K) K-dimensional embeddings for N receiving nodes
        y: Tensor
            (M, K) K-dimensional embeddings for M sending nodes
        attn_mask: Tensor, optional
            (1, M, N) Binary attention mask

        Returns:
        --------
        m_r: Tensor
            (N, K) K-dimensional weighted messages (values) for N instances
        attn_data: Tensor, Optional
            If required, return self attention weights. Otherwise return None.
        """
        m, attn_data = self.attention(x, y, attn_mask)
        x = self.output(m, x)
        return x, attn_data
