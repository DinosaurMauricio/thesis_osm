import torch
from transformers import ViTModel, ViTImageProcessor
from .config.osmcap import OSMCAP_Config


class ViT(torch.nn.Module):
    """
    Simple wrapper around a ViT pretrained model.
    It outputs the activations of the last layer (without the CLS token activation).
    """

    def __init__(self, config: OSMCAP_Config):
        super().__init__()
        rank = getattr(config, "rank", None)
        device = getattr(config, "device", None)

        kwargs = {"device_map": rank if rank is not None else device}

        quantization_config = getattr(config, "quantization_config", None)
        if quantization_config is not None and config.quantize_vision:
            kwargs["quantization_config"] = quantization_config
            kwargs["low_cpu_mem_usage"] = True

        self.processor = ViTImageProcessor.from_pretrained(config.vision.path)
        self.visual = ViTModel.from_pretrained(config.vision.path, **kwargs)

    def _freeze_vision(self):
        self.visual.eval()
        for param in self.visual.parameters():
            param.requires_grad = False

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Receive a batch of (RGB) images and return the hidden states of the visual model
        Inputs:
        - images: (B, 3, H, W) tensor
        Outputs:
        - hidden_states: (B, L, H) tensor of last layer hidden states
        """
        # This is done on CPU, so I have to move the tensor on the GPU
        # beacuse DistributedDataParallel limitations with the processor
        # if the tensor is not on the same device there will be an error
        # because its not on the same device as the model
        inputs = self.processor(
            images=images, return_tensors="pt", do_rescale=False
        ).to(images.device)

        outputs = self.visual(**inputs)
        outputs = outputs.last_hidden_state[:, 1:, :]

        return outputs