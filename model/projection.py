import torch
from .config.osmcap import OSMCAP_Config

class MLPMap(torch.nn.Module):
    def __init__(self, n_layers: int, input_size: int, hidden_size: int, **kwargs):
        super().__init__()
        self.n_layers = n_layers
        # Append first layer
        layers = [torch.nn.Linear(input_size, hidden_size, **kwargs)]
        for _ in range(n_layers - 1):
            layers.append(torch.nn.GELU())
            layers.append(torch.nn.Linear(hidden_size, hidden_size, **kwargs))

        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)

class Proj(torch.nn.Module):
    '''
    Visual projection layer. This layer projects the visual tokens to the same embedding dimension of the language model to enable prompt concatenation.
    If n_layers == 1 this corresponds to a linear layer. 
    If n_layers > 1 this corresponds to an MLP with GELU activation function. 
    '''
    def __init__(self, config: OSMCAP_Config):
        super().__init__()
        
        kwargs = {}
        
        kwargs["n_layers"] = config.projection.mlp.n_layers
        kwargs["input_size"] = config.vision.hidden_size
        kwargs["hidden_size"] = config.llm.hidden_size
        kwargs["bias"] = config.projection.mlp.bias
        self.proj = MLPMap(**kwargs)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Hv = visual embedding dimension
        Hl = language embedding dimension
        Input:
        - Visual hidden states: shape (B, L, Hv)
        Output:
        - Projected hidden states: shape (B, L, Hl)
        """
        
        return self.proj(hidden_states)
