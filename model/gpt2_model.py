from transformers import GPT2LMHeadModel, AutoConfig
from .base_language_model import BaseLanguageModel
from .config.osmcap import OSMCAP_Config

class GPT2Model(BaseLanguageModel):
    """
    GPT2Model variant using GPT2LMHeadModel.
    """

    def __init__(self, config: OSMCAP_Config):
        super().__init__(config)

    def _initialize_model(self, pretrained=True):
        kwargs = self._get_model_kwargs()
        if pretrained:
            return GPT2LMHeadModel.from_pretrained(self.config.llm.path, **kwargs)
        else:
            config = AutoConfig.from_pretrained(
                "gpt2",
                n_layer=2,
                n_head=8,
                **kwargs
            )
            return GPT2LMHeadModel(config)

    def _add_gated_cross_attention(self):
        self._init_layers(self.llm.transformer.h)
        for xattn in self.modified_layers:
            for param in xattn.xattn_block.parameters():
                param.requires_grad = True

    def _get_model_kwargs(self):

        kwargs = {
            "device_map": (
                self.config.rank if self.config.rank is not None else self.config.device
            )
        }

        quantization_config = getattr(self.config, "quantization_config", None)
        if quantization_config is not None and self.config.quantize_language:
            kwargs["quantization_config"] = quantization_config
            kwargs["low_cpu_mem_usage"] = True

        kwargs["attn_implementation"] = "sdpa"
        return kwargs

    def _get_input_embeddings(self, input_ids):
        return self.llm.transformer.get_input_embeddings()(input_ids)
        