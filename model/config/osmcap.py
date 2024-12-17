from transformers import PretrainedConfig


class LLMConfig(PretrainedConfig):
    """
    Configuration for the LLM component.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class TrainerConfig(PretrainedConfig):
    """
    Configuration for the Perceiver Resampler component.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class TraningConfig(PretrainedConfig):
    """
    Configuration for the Traning component.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class ProjectionConfig(PretrainedConfig):
    """
    Configuration for the Projection module.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class VisonConfig(PretrainedConfig):
    """
    Configuration for the Vison component.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class ResamplerConfig(PretrainedConfig):
    """
    Configuration for the osm integration part.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class OSMCAP_Config(PretrainedConfig):
    model_type = "OSMCAP"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class TransformationsConfig(PretrainedConfig):
    """
    Configuration for the transformations.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class ImageOSMSimConfig(PretrainedConfig):
    """
    Configuration for the Image OSM Similarity model.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class SentenceEmbeddingConfig(PretrainedConfig):
    """
    Configuration for the Sentence Embedding model.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)