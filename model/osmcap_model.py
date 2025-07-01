import torch
from typing import Optional
from transformers import PreTrainedModel

from model.image_osm_sim import ImageOSMSim

from .clip_vision import CLIPVision
from .remote_clip import RemoteCLIP
from .vit import ViT
from .projection import Proj
from utils.model import calculate_right_model_weights

from .config.osmcap import OSMCAP_Config
from .phi2_model import PHI2Model
from .gpt2_model import GPT2Model
from .perceiver_resampler import PerceiverResampler

from peft import prepare_model_for_kbit_training, get_peft_model

from dataset.constants import MLP, CROSS_ATTN


class OSMCAP(PreTrainedModel):

    def __init__(
        self,
        config: OSMCAP_Config,
        image_sim_config: OSMCAP_Config = None,
        master_process=True,
    ):
        super().__init__(config)
        self.config = config
        self.vision = self._load_vision_model(config)
        if config.projection.type == MLP:
            self.proj = Proj(config)

        if config.trainer.use_resampler:
            self.resampler = PerceiverResampler(config.resampler)

        self.language = self._load_language_model(config.llm.name, config)

        self.object_embedding = torch.nn.Embedding(
            num_embeddings=100, embedding_dim=self.config.llm.hidden_size, padding_idx=0
        )

        if self.config.trainer.sentence_embedding.enabled:
            self.project_osm = torch.nn.Linear(
                in_features=config.trainer.sentence_embedding.dim,
                out_features=self.config.llm.hidden_size,
            )

        if config.tokenizer is None:
            raise ValueError("Tokenizer was not set")

        self.tokenizer = config.tokenizer
        self.master_process = master_process

        if self.config.image_osm_similarity.enabled:
            self.image_osm_similarity = ImageOSMSim(image_sim_config)
            state_dict = torch.load(
                f"{self.config.img_osm_path}/best_model.pth",
                map_location=self.device,
            )
            self.image_osm_similarity.load_state_dict(state_dict)
            print("Loaded Image OSM Similarity model")

        # Prepare model for training
        self._prepare_model_for_training()
        self._unfreeze_modules(list_module_names=[])
        self._print_model_weights()

    def _load_vision_model(self, vision_config):
        vision_models = dict(clip=CLIPVision, remote=RemoteCLIP, vit=ViT)

        assert (
            vision_config.vision.name in vision_models
        ), f"vision model can only be one of {vision_models.keys()}"

        return vision_models[vision_config.vision.name](vision_config)

    def _load_language_model(self, llm: str, osmcap_config: OSMCAP_Config = None):
        llms = dict(phi=PHI2Model, gpt=GPT2Model)
        assert llm in llms, f"activation can only be one of {llms.keys()}"

        if osmcap_config is None:
            ValueError("osmcap_config was not set")

        return llms[llm](osmcap_config)

    def _print_model_weights(self):
        if self.master_process:
            self.total_vision_params = calculate_right_model_weights(
                self.vision.parameters(),
                is_loaded_in_4bit=self.config.trainer.quantize_vision,
            )
            self.total_trainable_vision_params = sum(
                p.numel() for p in self.vision.parameters() if p.requires_grad
            )
            print(f"Total number of vision parameters: {self.total_vision_params}")
            print(
                f"Total number of vision trainable parameters: {self.total_trainable_vision_params}"
            )

            if self.config.projection.type == MLP:
                self.total_proj_params = sum(p.numel() for p in self.proj.parameters())
                self.total_proj_trainable_params = sum(
                    p.numel() for p in self.proj.parameters() if p.requires_grad
                )
                print(
                    f"Total number of projection parameters: {self.total_proj_params}"
                )
                print(
                    f"Total number of projection trainable parameters: {self.total_proj_trainable_params}"
                )

            if self.config.projection.type == CROSS_ATTN:
                layer_list = self.language.modified_layers
                self.total_cross_attn_params = sum(
                    p.numel()
                    for layer in layer_list
                    for n, p in layer.named_parameters()
                    if "xattn_block" in n
                )
                self.total_cross_attn_trainable_params = sum(
                    p.numel()
                    for layer in layer_list
                    for n, p in layer.named_parameters()
                    if "xattn_block" in n and p.requires_grad
                )
                print(
                    f"Total number of cross attention parameters: {self.total_cross_attn_params}"
                )
                print(
                    f"Total number of cross attention trainable parameters: {self.total_cross_attn_trainable_params}"
                )
            else:
                self.total_cross_attn_params = 0
                self.total_cross_attn_trainable_params = 0

            if self.config.trainer.use_resampler:
                self.total_resampler_params = sum(
                    p.numel() for p in self.resampler.parameters() if p.requires_grad
                )
                self.total_resampler_trainable_params = sum(
                    p.numel() for p in self.resampler.parameters() if p.requires_grad
                )
                print(
                    f"Total number of resampler parameters: {self.total_resampler_params}"
                )
                print(
                    f"Total number of resampler trainable parameters: {self.total_resampler_trainable_params}"
                )

            self.total_language_params = calculate_right_model_weights(
                self.language.parameters(),
                is_loaded_in_4bit=self.config.trainer.quantize_language,
            )
            self.total_trainable_language_params = sum(
                p.numel() for p in self.language.parameters() if p.requires_grad
            )

            print(
                f"Total number of language parameters: {self.total_language_params-self.total_cross_attn_params}"
            )
            print(
                f"Total number of language trainable parameters: {self.total_trainable_language_params-self.total_cross_attn_trainable_params}"
            )

            self.total_parameters = (
                self.total_language_params + self.total_vision_params
            )
            self.total_trainable_parameters = sum(
                p.numel() for p in self.parameters() if p.requires_grad
            )
            print(f"Total number of parameters: {self.total_parameters}")
            print(
                f"Total number of trainable parameters: {self.total_trainable_parameters}"
            )

    def _prepare_model_for_training(self):
        if self.config.quantize_vision:
            self.vision = prepare_model_for_kbit_training(
                self.vision, use_gradient_checkpointing=False
            )
        if self.config.quantize_language:
            self.language = prepare_model_for_kbit_training(
                self.language, use_gradient_checkpointing=False
            )

    def _unfreeze_modules(self, list_module_names: list[str] = []):
        if self.config.projection.type == MLP:
            self._unfreeze_proj()
        elif self.config.projection.type == CROSS_ATTN:
            self.language._unfreeze_cross_attn()

        for name, param in self.named_parameters():
            for name_module in list_module_names:
                if name_module in name:
                    # Cast the param to dtype float32 before unfreezing (training is better in float32)
                    param.data = param.data.to(torch.float32)
                    param.requires_grad = True

    def _unfreeze_proj(self):
        for param in self.proj.parameters():
            # Cast the param to dtype float32 before unfreezing (training is better in float32)
            param.data = param.data.to(torch.float32)
            param.requires_grad = True

    def insert_adapters(self, lora_config):
        self.language.llm = get_peft_model(self.language.llm, lora_config)

    def forward(
        self,
        images: torch.Tensor,
        images_mask: torch.Tensor,
        input_ids: torch.LongTensor = None,
        input_attention_mask: Optional[torch.LongTensor] = None,
        osm_content: Optional[torch.LongTensor] = None,
        osm_attention_mask: Optional[torch.LongTensor] = None,
        osm_object_identifiers: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:

        visual_features = self.vision(images)
        B, L, _ = visual_features.shape
        device = visual_features.device
        images_mask = images_mask.repeat(1, L)

        if not self.config.trainer.sentence_embedding.enabled:
            # Get the embeddings of the osm tokens
            osm_features = self.language._get_input_embeddings(osm_content)
            object_identifiers_embds = self.object_embedding(osm_object_identifiers)
            osm_features += object_identifiers_embds  # Add the object identifier (TODO see if it is useful only when not embedding..)
        else:
            osm_content = osm_content.to(
                dtype=visual_features.dtype
            )  # Cast the features to match the model's datatype
            if osm_content is not None:

                if self.config.image_osm_similarity.enabled:
                    with torch.no_grad():
                        osm_content, osm_attention_mask = (
                            self.image_osm_similarity.get_most_similar_osm_embeddings(
                                osm_content,
                                images,
                                self.config.image_osm_similarity.top_k,
                            )
                        )
            # Project the osm content
            osm_features = self.project_osm(osm_content)

        # Concatenate the features
        features = torch.cat([visual_features, osm_features], dim=1)
        features_attn_mask = torch.cat([images_mask, osm_attention_mask], dim=1)

        # Use the resampler if enabled
        if self.config.trainer.use_resampler:
            resampled_features, _ = self.resampler(
                features=features, mask=features_attn_mask, return_attn_scores=False
            )

            B, L, _ = resampled_features.shape
            # Modify the attention mask as the tokens are now resampled
            features_attn_mask = torch.ones(B, L, device=device)

        outputs = self.language(
            features=(
                resampled_features if self.config.trainer.use_resampler else features
            ),
            features_attn_mask=features_attn_mask,
            input_ids=input_ids,
            input_attention_mask=input_attention_mask,
            labels=labels,
        )

        return outputs

    def generate_caption(
        self,
        images: torch.Tensor,
        images_mask: torch.Tensor,
        start_word: str,
        osm_content: Optional[torch.LongTensor] = None,
        osm_attention_mask: Optional[torch.LongTensor] = None,
        osm_object_identifiers: Optional[torch.LongTensor] = None,
        max_new_tokens: Optional[int] = None,
    ) -> torch.LongTensor:

        with torch.no_grad():

            visual_features = self.vision(images)
            B, L, _ = visual_features.shape
            device = visual_features.device
            images_mask = images_mask.repeat(1, L)

            if not self.config.trainer.sentence_embedding.enabled:
                # Get the embeddings of the osm tokens
                osm_features = self.language._get_input_embeddings(osm_content)
                object_identifiers_embds = self.object_embedding(osm_object_identifiers)
                osm_features += object_identifiers_embds
            else:
                osm_content = osm_content.to(
                    dtype=visual_features.dtype
                )  # Cast the features to match the model's datatype
                if self.config.image_osm_similarity.enabled:
                    osm_content, osm_attention_mask = (
                        self.image_osm_similarity.get_most_similar_osm_embeddings(
                            osm_content,
                            images,
                            self.config.trainer.image_osm_similarity.top_k,
                        )
                    )
                # Project the osm content
                osm_features = self.project_osm(osm_content)

            # Concatenate the features
            features = torch.cat([visual_features, osm_features], dim=1)
            features_attn_mask = torch.cat([images_mask, osm_attention_mask], dim=1)

            # Use the resampler if enabled
            if self.config.trainer.use_resampler:
                resampled_features, _ = self.resampler(
                    features=features, mask=features_attn_mask, return_attn_scores=False
                )

                B, L, _ = resampled_features.shape
                # Modify the attention mask as the tokens are now resampled
                features_attn_mask = torch.ones(B, L, device=device)

            outputs = self.language.generate(
                features=(
                    resampled_features
                    if self.config.trainer.use_resampler
                    else features
                ),
                features_attn_mask=features_attn_mask,
                max_new_tokens=max_new_tokens,
                start_word_id=self.tokenizer.encode(start_word, return_tensors="pt").to(
                    device=device
                ),
            )

        return outputs
