import torch
from torch import nn
from transformers import GenerationConfig
from abc import ABC, abstractmethod
from .config.osmcap import OSMCAP_Config
from .cross_attention import ModifiedLMBlock
from typing import List

from dataset.constants import IGNORE_INDEX, MLP, CROSS_ATTN


class BaseLanguageModel(nn.Module, ABC):
    def __init__(self, config: OSMCAP_Config):
        super().__init__()
        self.config = config
        self.llm = self._initialize_model(pretrained=True)
        self.llm.resize_token_embeddings(config.vocab_size)

        self.dropout = nn.Dropout(p=config.transformations.dropout.p)

        # Parameters for the cross attention
        if config.projection.type == CROSS_ATTN:
            self.freq = config.projection.cross_attention.freq
            self.dim_head = config.projection.cross_attention.dim_head
            self.heads = config.projection.cross_attention.heads
            self.ff_mult = config.projection.cross_attention.ff_mult
            self.activation = config.projection.activation
            self.dim_features = config.projection.cross_attention.dim_features

            self._add_gated_cross_attention()
            
        if config.llm.freeze:
            self._freeze_llm()

    def _init_layers(self, lm_layers: torch.nn.ModuleList):
        """Adding cross attention layer between LM layers"""
        self.modified_layers: List[ModifiedLMBlock] = []

        for i, lm_layer in enumerate(lm_layers):
            if i % self.freq != 0:
                continue

            modified_layer = ModifiedLMBlock(
                lm_layer,
                dim=self.llm.config.hidden_size,
                dim_features=self.dim_features,
                dim_head=self.dim_head,
                heads=self.heads,
                ff_mult=self.ff_mult,
                act=self.activation,
            )
            self.modified_layers.append(modified_layer)
            lm_layers[i] = modified_layer

    @abstractmethod
    def _add_gated_cross_attention(self):
        pass

    @abstractmethod
    def _initialize_model(self):
        pass

    @abstractmethod
    def _get_input_embeddings(self, input_ids):
        pass

    def _freeze_llm(self):
        # Freeze everything
        for param in self.parameters():
            param.requires_grad = False
            
    def _unfreeze_cross_attn(self):
        # If cross attention is enabled, don't freeze cross_attn layers
        if self.config.projection.type == CROSS_ATTN:
            for name, param in self.llm.named_parameters():
                if "xattn_block" in name:
                    # Cast the param to dtype float32 before unfreezing (training is better in float32)
                    param.data = param.data.to(torch.float32)
                    param.requires_grad = True

    def forward(
        self,
        features: torch.Tensor,
        features_attn_mask: torch.LongTensor,
        input_ids: torch.LongTensor,
        input_attention_mask: torch.LongTensor,
        labels: torch.LongTensor,
    ) -> torch.Tensor:
        """
        Receive a batch of tokenized captions (input_ids) and associated attention_masks, visual_features of the images and optional osm_features.
        Run the forward pass in the llm and returns the output.

        input_ids: (B, L) tensor of tokenized captions
        input_attention_mask: (B, L) tensor of captions attention masks
        visual_features: (B, Lf, Df) tensor of visual features (Lf: number of visual features embeddings, Df: dimension of visual features embeddings)
        osm_features: (B, Lo, Do) tensor of osm resampled embeddings (Lo: number of latents embeddings, Do: dimension of the latents)
        """

        # Cut the osm data to a predefined threshold
        if features.shape[1]>self.config.trainer.max_features_length:
            features = features[:, : self.config.trainer.max_features_length]
            features_attn_mask = features_attn_mask[:, :self.config.trainer.max_features_length]
            
        if self.config.transformations.dropout.enabled:
            features = self.dropout(features)

        # Condition the model with the features
        for xattn in self.modified_layers:
            xattn.condition(features=features, mask=features_attn_mask, xattn_layer_past=None)
                
        # Forward to the llm
        outputs = self.llm(
            input_ids=input_ids,
            attention_mask=input_attention_mask,
            labels=labels,
        )

        return outputs

    def generate(
        self,
        features: torch.Tensor,
        features_attn_mask: torch.LongTensor,
        max_new_tokens: int,
        start_word_id: int,
    ) -> torch.LongTensor:
        """
        Receive a batch of image features and start generating sentences that describe the image
        input_ids: (B, L) tensor
        attention_mask: (B, L) tensor
        """
        # Create a generation config
        greedy_decoding_conf = GenerationConfig(
            do_sample=False,
            num_beams=1,
            bos_token_id=self.config.llm.bos_token_id,
            eos_token_id=self.config.llm.eos_token_id,
            pad_token_id=self.config.llm.pad_token_id,
            max_new_tokens=max_new_tokens
        )
        
        B, _, _ = features.shape
        
        start_word_ids = start_word_id.repeat(B, 1)
        attention_mask = torch.ones((B, 1), device=features.device)
        
        if features.shape[1]>self.config.trainer.max_features_length:
            features = features[:, : self.config.trainer.max_osm_length]
            features_attn_mask = features_attn_mask[:, :self.config.trainer.max_osm_length]

        # Condition the model with the features
        for xattn in self.modified_layers:
            xattn.condition(features=features, mask=features_attn_mask, xattn_layer_past=None)
                
        # Call the generate function
        outputs = self.llm.generate(
            input_ids=start_word_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            generation_config=greedy_decoding_conf,
        )

        return outputs