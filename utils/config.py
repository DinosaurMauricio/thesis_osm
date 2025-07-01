import optuna
import torch
import os
from omegaconf import DictConfig, OmegaConf
from transformers import AutoTokenizer, BitsAndBytesConfig
from torch.nn.parallel import DistributedDataParallel as DDP
from optuna.pruners import ThresholdPruner

from model.osmcap_model import OSMCAP
from model.config.osmcap import (
    ImageOSMSimConfig,
    LLMConfig,
    ProjectionConfig,
    SentenceEmbeddingConfig,
    VisonConfig,
    OSMCAP_Config,
    ResamplerConfig,
    TrainerConfig,
    TransformationsConfig,
)


def set_device_settings(cfg: DictConfig):
    cfg.device_count = torch.cuda.device_count()

def set_command_line_args(cfg: DictConfig, args):
    cfg.arg_llm = args.llm
    cfg.arg_vision = args.vision
    cfg.arg_dataset = args.dataset


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True


def set_cuda_settings():
    # Enable backends for TF32 magic
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def load_model(osmcap_config, config, master_process, distributedgpu=False, rank=None, img_osm_config=None):
    model = OSMCAP(config=osmcap_config, image_sim_config=img_osm_config, master_process=master_process)

    model.config.use_cache = False

    # Reminder: Any model config should be applied before wrapping it with DDP (calling our setup_gpu_for_model method)
    # else it might cause problems as it might wrap the whole modle in DDP and cause errors

    model = setup_gpu_for_model(
        model,
        config.device_count,
        osmcap_config.device,
        master_process,
        distributedgpu,
        rank,
    )

    return model

def setup_gpu_for_model(
    model,
    num_gpus,
    model_device,
    master_process,
    distributed_gpu=False,
    rank=None,
):

    if num_gpus is None:
        raise ValueError("device_count was not set in the config")

    if distributed_gpu:
        if master_process:
            print("Using Distributed GPU")
        model = model.to(rank)
        model = DDP(model, device_ids=[rank])
    else:
        if master_process:
            print("Using Single GPU")
        model = model.to(model_device)

    return model


def load_config(project_path, config_path):
    config_path = os.path.join(project_path, config_path)
    return OmegaConf.load(config_path)


def setup_tokenizer(llm_path, llm_name):
    tokenizer = AutoTokenizer.from_pretrained(
        llm_path,
        use_fast=True,
        clean_up_tokenization_spaces=True,
    )
    if llm_name in ["phi", "gpt"]:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def initialize_model_config(
    config, llm_name, vision_name, distributegpu, tokenizer, model_device, llm, vision_model, rank=None
):

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    osmcap_config = OSMCAP_Config(
        llm=LLMConfig(name=llm_name, **llm, freeze=config.llm_models.freeze),
        quantization_config=quantization_config,
        quantize_language=config.trainer.quantize_language,
        quantize_vision=config.trainer.quantize_vision,
        projection=ProjectionConfig(**config.projection),
        vision=VisonConfig(name=vision_name, **vision_model, freeze=config.vision_model.freeze, use_only_cls_token=config.vision_model.use_only_cls_token),
        resampler=ResamplerConfig(**config.resampler),
        trainer=TrainerConfig(**config.trainer),
        vocab_size=len(tokenizer),
        device=model_device,
        rank=rank if distributegpu else None,
        tokenizer=tokenizer,
        use_lora=config.lora.use_lora,
        transformations=TransformationsConfig(**config.transformations),
        image_osm_similarity=ImageOSMSimConfig(**config.trainer.image_osm_similarity),
        sentence_embedding=SentenceEmbeddingConfig(**config.trainer.sentence_embedding),
    )

    return osmcap_config


def configure_optuna_trial(trial, config, distributedgpu=False):
    if distributedgpu:
        trial = optuna.integration.TorchDistributedTrial(trial)
    # TODO: Probably add to config file these parameters
    lr = trial.suggest_float("lr", 1e-6, 1e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 0.0, 0.0001)
    dropout_rate = trial.suggest_float("dropout", 0.2, 0.5)
    config.trainer.lr = lr
    config.trainer.weight_decay = weight_decay
    config.transformations.dropout.p = dropout_rate

    return trial


def create_and_run_study(function, rank=0, number_trials=0, threshold=0):
    study = optuna.create_study(
        study_name="OpenStreetSatellite",
        direction="minimize",
        pruner=ThresholdPruner(upper=threshold),
    )
    study.optimize(
        lambda trial: function(rank=rank, trial=trial),
        n_trials=number_trials,
    )
    return study
