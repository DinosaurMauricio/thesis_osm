import wandb
from omegaconf import OmegaConf


def initialize_logging(config, project_name, group_by: str = None):
    wandb.require("core")
    wandb._service_wait = 60
    wandb.init(
        project=project_name,
        config=OmegaConf.to_container(config, resolve=True),
        group=group_by,
    )
