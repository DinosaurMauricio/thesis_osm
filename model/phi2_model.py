from transformers import PhiForCausalLM
from .base_language_model import BaseLanguageModel
from .config.osmcap import OSMCAP_Config


class PHI2Model(BaseLanguageModel):
    """
    PHI2Model variant using PhiForCausalLM.
    """

    def __init__(self, config: OSMCAP_Config):
        super().__init__(config)

    def _initialize_model(self):
        kwargs = self._get_model_kwargs()
        return PhiForCausalLM.from_pretrained(self.config.llm.path, **kwargs)
    
    def _add_gated_cross_attention(self):
        self._init_layers(self.llm.model.layers)
        for xattn in self.modified_layers:
            for param in xattn.xattn_block.parameters():
                param.requires_grad = True

    def _get_model_kwargs(self):
        kwargs = {}
        quantization_config = getattr(self.config, "quantization_config", None)
        if quantization_config is not None and self.config.quantize_language:
            kwargs["quantization_config"] = quantization_config
            kwargs["low_cpu_mem_usage"] = True

        kwargs["attn_implementation"] = "sdpa"
        return kwargs

    def _get_input_embeddings(self, input_ids):
        return self.llm.model.get_input_embeddings()(input_ids)

    def _freeze_llm(self):
        """
        Freeze everything but not the embedding layer.
        This because I added tokens in the embedding layer and I want to train also those.
        """
        self.llm.eval()
        for name, param in self.llm.named_parameters():
            param.requires_grad = "embed_tokens" in name

    def unfreeze_vocab_emb(self):
        for name, param in self.llm.named_parameters():
            param.requires_grad = "embed_tokens" in name or "lora" in name
