# This code is referenced from https://github.com/dhansmair/flamingo-mini/blob/main/flamingo_mini/gated_cross_attention.py

"""
Gated cross-attention layer adapted from flamingo-pytorch.
"""
from typing import Optional, Tuple

import torch
from einops import rearrange
from einops_exts import rearrange_many
from torch import einsum, nn, tanh

from utils.model import feed_forward_layer

class MaskedCrossAttention(nn.Module):
    """Cross attention layer with masking.

    Args:
        dim: d_token, d_visual dimensionality of language and visual tokens
        dim_head: dimensionality of the q, k, v vectors inside one attention head
        heads: number of attention heads
    """

    def __init__(self, dim: int, dim_features: int, dim_head: int = 64, heads: int = 8):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        inner_dim = dim_head * heads
        self.layer_norm = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim_features, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(
        self,
        input: torch.FloatTensor,
        features: Optional[torch.FloatTensor] = None,
        mask: Optional[torch.Tensor] = None,
        previous_kv: tuple = None,
        output_kv: bool = False,
    ):
        """
        Args:
            input: text features (batch_size, n_token, d_token)
            features: other features (batch_size, n_features, d_features)
            mask: mask for the features (batch_size, n_features)
            previous_kv: tuple of previous keys and values. Passed when caching is used during text generation
            output_kv: whether to return the keys and values

        Returns:
            Tensor (batch_size, n_token, d_token)
        """
        
        input = self.layer_norm(input)
        
        # Compute the queries from the text tokens
        q = self.to_q(input)
        q = q * self.scale

        # Compute the keys and values from the features
        if previous_kv is None:
            k, v = self.to_kv(features).chunk(2, dim=-1)
            q, k, v = rearrange_many((q, k, v), 'b n (h d) -> b h n d', h=self.heads)
        else:
            k, v = previous_kv
            q = rearrange(q, 'b n (h d) -> b h n d', h=self.heads)

        # To solve bug in beam search
        Bk = k.shape[0]
        Bq = q.shape[0]
        if Bk!=Bq:
            assert Bq % Bk == 0
            multiplier = int(Bq / Bk)
            k = torch.repeat_interleave(k, multiplier, dim=0)
            v = torch.repeat_interleave(v, multiplier, dim=0)
            mask = torch.repeat_interleave(mask, multiplier, dim=0)
            
        # Compute the attention scores from the queries and keys
        sim = einsum('... i d, ... j d -> ... i j', q, k)

        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)
            expanded_mask = mask.expand_as(sim)
            sim = torch.masked_fill(sim, expanded_mask==0, -torch.inf)
            
        alphas = sim.softmax(dim=-1)
        alphas = torch.nan_to_num(alphas, nan=0.0) # This because there are cases in which we don't have any data.
        
        out = einsum('... i j, ... j d -> ... i d', alphas, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        conditioned_tokens = self.to_out(out)
        
        if output_kv:
            return conditioned_tokens, (k, v)

        return conditioned_tokens, None


class GatedCrossAttentionBlock(nn.Module):
    """
    Args:
        dim: d_token, d_visual
        dim_head: dimensionality of q, k, v inside the attention head
        heads: number of attention heads
        ff_mult: factor for the number of inner neurons in the ffw layer
    """

    def __init__(
        self,
        dim: int,
        dim_features: int,
        dim_head: int = 64,
        heads: int = 8,
        ff_mult: int = 4,
        act: str = 'gelu',
    ):
        super().__init__()
        self.attn = MaskedCrossAttention(
            dim=dim, dim_features=dim_features, dim_head=dim_head, heads=heads
        )
        self.alpha_attn = nn.Parameter(torch.tensor([0.0]))  # type: ignore[reportPrivateUsage]

        self.ffw = feed_forward_layer(dim, mult=ff_mult, activation=act)
        self.alpha_ffw = nn.Parameter(torch.tensor([0.0]))  # type: ignore[reportPrivateUsage]

    def forward(self, input: torch.LongTensor, features: torch.FloatTensor, mask:torch.Tensor=None, previous_kv=None, output_kv=False):
        """
        Args:
            y: language features from previous LM layer (batch_size, n_tokens, d_token)
            media: visual features, encoded by perceiver resample (batch_size, n_media, n_queries, dim)
        """
        shape_before = input.shape

        # kv will be None if output_kv=False
        attn_out, kv = self.attn(input, features, mask=mask, previous_kv=previous_kv, output_kv=output_kv)
        input = input + tanh(self.alpha_attn) * attn_out
        assert input.shape == shape_before
        input = input + tanh(self.alpha_ffw) * self.ffw(input)
        assert input.shape == shape_before
        return input, kv


class ModifiedLMBlock(nn.Module):
    """
    A block that wraps a gated cross-attention layer, followed by a LM layer.
    We replace the original layers in the LM with these at a certain frequency
    to introduce the xattn layer. This layer mimics the functionality and behavior
    of the underlying LM block. This way, the LM can be used in the same way as before,
    and we can do the conditioning without altering the LM implementation.

    One drawback of this approach is that we cannot pass the visual features to forward()
    directly, but instead we need to pass them before the actual forward pass, via a
    side-channel, which is the condition() method. In addition, when use_cache is used,
    the cached keys and values for the xattn layers need to be retrieved separately from
    the kv_output property.

    (!) This implementation works with GPT-2 layers, but hasn't been tested with other LMs yet.
    """

    def __init__(self, lm_block, **kwargs):
        super().__init__()
        self.xattn_block = GatedCrossAttentionBlock(**kwargs)
        self.lm_block = lm_block
        self.features = None
        self.xattn_layer_past = None
        self.kv_output = None

    def condition(self, features: torch.FloatTensor, mask: torch.Tensor=None, xattn_layer_past=None):
        """
        Conditioning. Called from outside of the LM before passing the features to the LM.
        This way, the gated cross-attention layers get informed about the information contained in the features
        without the need to pipe the visual input through the LM forward() function.

        xattn_layer_past can contain the cached cross-attention keys and values (computed
        from the input features). Passing them is useful to speed up the autoregressive text
        generation where the keys and values will be the same for every word, since the input doesn't change
        
        If both features and xattn_layer_past are passed, features will be ignored in the xattn layers.
        """
        self.features = features
        self.mask = mask
        self.xattn_layer_past = xattn_layer_past

    def forward(self, hidden_states: Optional[Tuple[torch.FloatTensor]], use_cache: Optional[bool] = False, **kwargs):
        """
        This forward function mimics forward() of GPT2Block, so it has the same input and output.
        """

        # Pass through xattn
        hidden_states, kv = self.xattn_block(
            input=hidden_states,
            features=self.features,
            mask=self.mask,
            previous_kv=self.xattn_layer_past,
            output_kv=use_cache,
        )
        self.kv_output = kv

        # Pass through original LM layer
        return self.lm_block(hidden_states, use_cache=use_cache, **kwargs)