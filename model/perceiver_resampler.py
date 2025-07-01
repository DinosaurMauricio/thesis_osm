# Code from from https://github.com/dhansmair/flamingo-mini and https://github.com/shan18/Perceiver-Resampler-XAttn-Captioning

import torch
from einops import rearrange, repeat
from einops_exts import rearrange_many
from torch import einsum, nn
from utils.model import feed_forward_layer


class PerceiverAttentionLayer(nn.Module):
    def __init__(self, *, dim_head=64, heads=8, mult=2, activation="relu"):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        self.dim_head = dim_head
        self.mult = mult
        self.activation = activation
        inner_dim = dim_head * heads

        # trainable components of PerceiverAttentionLayer
        self.norm_features = nn.LayerNorm(inner_dim)
        self.norm_latents = nn.LayerNorm(inner_dim)

        self.to_q = nn.Linear(inner_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(inner_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(inner_dim, inner_dim, bias=False)
        self.ff = feed_forward_layer(
                    dim=inner_dim, mult=self.mult, activation=self.activation
                )

    def forward(self, features, latents, mask=None, return_attn_scores=False):
        """
        Latent vectors are cross-attending to the visual features x.
        :param features: Tensor (n_batch, n_features, dim)
        :param latents: Tensor (n_batch, n_latents, dim)
                        latent learnt vectors from which the queries are computed.
        :return:        Tensor (n_batch, n_latents, dim)
        :return_attn_scores: Debug, returns attention scores in the perceiver layers
        """
        assert features.ndim == 3
        assert latents.ndim == 3
        assert features.shape[0] == latents.shape[0]
        assert features.shape[2] == latents.shape[2]

        n_heads = self.heads
        n_batch, n_features, _ = features.shape
        n_queries = latents.shape[1]

        # layer normalization
        x = self.norm_features(features)
        latents = self.norm_latents(latents)
        
        # queries
        # compute the queries from the latents, for all attention heads simultaneously.
        q = self.to_q(latents)
        q = rearrange(q, "b q (h d) -> b h q d", h=n_heads)
        assert q.shape == torch.Size([n_batch, n_heads, n_queries, self.dim_head])
        
        # keys, values
        k = self.to_k(x)
        v = self.to_v(x)

        # split so we have an extra dimension for the heads
        # batch, features, (heads, dim)
        k, v = rearrange_many((k, v), "b f (h d) -> b h f d", h=n_heads)
        assert v.shape == torch.Size(
            [n_batch, n_heads, n_features, self.dim_head]
        )

        q = q * self.scale

        # attention scores
        sim = einsum("b h q d, b h f d -> b h q f", q, k)

        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)
            expanded_mask = mask.expand_as(sim)
            sim = torch.masked_fill(sim, expanded_mask==0, -torch.inf)
        
        alphas = sim.softmax(dim=-1)
        alphas = torch.nan_to_num(alphas, nan=0.0) # This because there are cases in which we don't have osm data.

        out = einsum("b h q f, b h f v -> b h q v", alphas, v)

        out = rearrange(out, "b h q v -> b q (h v)")
        
        if return_attn_scores:
            return self.ff(out), alphas
        else:
            return self.ff(out), None

class PerceiverResampler(nn.Module):
    """Perceiver Resampler with multi-head attention layer"""

    def __init__(
        self,
        config,
    ):
        super().__init__()

        self.dim = config.dim
        self.num_queries = config.num_latents
        self.dim_head = config.dim_head
        self.depth = config.depth
        self.heads = config.heads
        self.ff_mult = config.ff_mult
        self.activation = config.activation
        self.trainable = config.trainable
        self.inner_dim = self.dim_head*self.heads

        self.latents = nn.Parameter(torch.randn(self.num_queries, self.inner_dim)) # Generate the latents as learnable parameters

        self.layers = nn.ModuleList([])
        for _ in range(self.depth):
            self.layers.append(
                    PerceiverAttentionLayer(
                        dim_head=self.dim_head, heads=self.heads, mult=self.ff_mult, activation=self.activation
                    )
                )
        self.project_down = nn.Linear(self.dim, self.inner_dim, bias=False) # bias not needed since we use layernorm after
        #self.project_up = nn.Linear(self.inner_dim, self.dim, bias=False) # bias not needed since we use layernorm after
        # Layer normalization takes as input the query vector length
        self.norm_features_in = nn.LayerNorm(self.dim)
        self.norm_latents_out = nn.LayerNorm(self.inner_dim)
        
        self._update_trainable_state(self.trainable)

    def _update_trainable_state(self, trainable: bool = True):
        for param in self.parameters():
            param.requires_grad = trainable

    def forward(self, features: torch.Tensor, mask: torch.LongTensor = None, return_attn_scores:bool=False):
        """Run perceiver resampler on the input features (also called embeddings)

        Args:
            features: Input feature embeddings of shape (batch_size, n_tokens, d_visual)
            mask: Mask for the input feature embeddings of shape (batch_size, n_tokens)

        Returns:
            Resampler features of shape (batch_size, num_queries, d_visual)
        """
        attn_scores = None
        batch_size, _, dim = features.shape

        assert dim == self.dim
        # Copy the latents for every element in the batch
        x = repeat(self.latents, "q d -> b q d", b=batch_size)
        
        features = self.norm_features_in(features)
        features = self.project_down(features)

        # Apply attention and feed forward layer
        for layer in self.layers:
            modified_latents, attn_scores = layer(features=features, latents=x, mask=mask, return_attn_scores=return_attn_scores)
            x = x + modified_latents

        assert x.shape == torch.Size([batch_size, self.num_queries, self.inner_dim])
        
        #x = self.project_up(x)
        
        if return_attn_scores:
            return self.norm_latents_out(x), attn_scores
        else:
            return self.norm_latents_out(x), None
