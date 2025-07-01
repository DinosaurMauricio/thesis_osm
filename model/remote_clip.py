import torch
from .config.osmcap import OSMCAP_Config
from huggingface_hub import hf_hub_download
import clip
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode

class RemoteCLIP(torch.nn.Module):
    """
    Simple wrapper around a CLIPVision pretrained model.
    It outputs the activations of the last layer (without the CLS token activation).
    """

    def __init__(self, config: OSMCAP_Config):
        super().__init__()
        rank = getattr(config, "rank", None)
        device = getattr(config, "device", None)
        
        self.use_only_cls_token = config.vision.use_only_cls_token

        model_device = rank if rank is not None else device

        kwargs = {"device_map": model_device}

        quantization_config = getattr(config, "quantization_config", None)
        if quantization_config is not None and config.quantize_vision:
            kwargs["quantization_config"] = quantization_config
            kwargs["low_cpu_mem_usage"] = True

        if config.vision.path not in ["RN50", "ViT-B/32", "ViT-L-14"]:
            raise ValueError(f"Invalid model name: {config.vision.path}")

        if config.vision.path == "ViT-B/32":
            # Reminder: ViT-B/32 is the default model but to download we need it to be ViT-B-32
            # Most likely it happens with the other models
            model_name = "ViT-B-32"

        # check if the model is already downloaded if no download it
        checkpoint_path = hf_hub_download(
            "chendelong/RemoteCLIP",
            f"RemoteCLIP-{model_name}.pt",
            cache_dir="checkpoints",
        )

        self.model, _ = clip.load(config.vision.path, device=model_device)
        self.preprocess = self._build_transform(n_px=self.model.visual.input_resolution)

        ckpt = torch.load(checkpoint_path, map_location=model_device)

        self.model.load_state_dict(ckpt)
        
    def _build_transform(self, n_px):
        return T.Compose([
            T.Resize(n_px, interpolation=InterpolationMode.BICUBIC),
            T.CenterCrop(n_px),
            T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

    def _freeze_vision(self):
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Receive a batch of (RGB) images and return the hidden states of the model
        Inputs:
        - images: (B, 3, H, W) tensor
        Outputs:
        - hidden_states: (B, L, H) tensor of last layer hidden states
        """

        transformed_images = self.preprocess(images)
        embeddings = self.visual_clip(transformed_images)
        
        return embeddings
        
    def visual_clip(self, x: torch.Tensor):
        x = self.model.visual.conv1(
            x.type((self.model.dtype))
        )  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [
                self.model.visual.class_embedding.to(x.dtype)
                + torch.zeros(
                    x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
                ),
                x,
            ],
            dim=1,
        )  # shape = [*, grid ** 2 + 1, width]

        x = x + self.model.visual.positional_embedding.to(x.dtype)
        x = self.model.visual.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.model.visual.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.model.visual.ln_post(x)
        
        if self.use_only_cls_token:
            x = x[:, 0, :]
            x = x.unsqueeze(1)
            
        return x
