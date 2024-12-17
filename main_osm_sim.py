from omegaconf import OmegaConf
import torch
import argparse
import torch.optim as optim
import os

from transformers.utils import logging
from transformers import AutoTokenizer
from torch.optim.lr_scheduler import ConstantLR
import wandb

from dataset.OSMSimilarity import OSMSentenceTransformerSimilarityDataset


from dataset.collate import custom_collate_sim_fn
from model.image_osm_sim import ImageOSMSim
from utils.config import load_config, initialize_model_config, set_seed
from utils.dataset import create_dataloaders
from utils.general_utils import create_directory_if_not_exists, save_config
from utils.logging import initialize_logging
from utils.model import save_model, train_sim, evaluate_sim

logging.set_verbosity(
    40
)  # To suppress warnings from long OSM data sequences in the tokenizer

parser = argparse.ArgumentParser(description="Image OSM Project")

parser.add_argument(
    "--save", type=str, default="best_model", help="Name of the model to save"
)
parser.add_argument(
    "--seed", type=int, default=None, help="Seed for reproducibility of results"
)
parser.add_argument("--log", action="store_true", help="Log to wandb")

args = parser.parse_args()

PATH_PROJECT = os.path.dirname(os.path.abspath(__file__))
LLM = "gpt"
VISION = "clip"
BATCH_SIZE = 8
CONFIG_FILE = "config_osm_sim.yaml"

config = load_config(PATH_PROJECT, CONFIG_FILE)
PATH = config.dataset.osm.path


if args.log:
    print(f"Config:\n\n{OmegaConf.to_yaml(config)}")
    initialize_logging(config, "Image OSM Project")


def get_complete_path(directory):
    output_path = os.path.join(
        PATH_PROJECT,
        directory,
    )
    create_directory_if_not_exists(output_path)

    return output_path


if args.seed:
    set_seed(args.seed)
    print(f"Seed set to: {args.seed}")

tokenizer = AutoTokenizer.from_pretrained(
    "gpt2",
    use_fast=True,
    clean_up_tokenization_spaces=True,
)
tokenizer.pad_token = tokenizer.eos_token

# - Flag to allow saving the config only once
allow_config_save = True

llm = config.llm_models[LLM]
print(f"Using {LLM} model: {llm.path}")
vision_model = config.vision_model[VISION]
print(f"Using {VISION} model: {vision_model.path}")

output_model_path = get_complete_path(
    os.path.join(config.similarity_output_dir, args.save)
)

config_inputs = {
    **config.trainer,
    **config.transformations,
}

osmcap_config = initialize_model_config(
    config, LLM, VISION, False, tokenizer, config.device, llm, vision_model
)

dataset_train = OSMSentenceTransformerSimilarityDataset(
    tokenizer, PATH, "train", **config_inputs
)
dataset_val = OSMSentenceTransformerSimilarityDataset(
    tokenizer, PATH, "val", **config_inputs
)


dataloaders_dict = create_dataloaders(
    {
        "train": (dataset_train, config.trainer.batch_size),
        "val": (dataset_val, config.trainer.batch_size),
    },
    False,
    args.seed,
    collate_fn=custom_collate_sim_fn,
)

train_dataloader = dataloaders_dict["train"]
val_dataloader = dataloaders_dict["val"]

model = ImageOSMSim(osmcap_config)

optimizer = optim.AdamW(
    model.parameters(), lr=config.trainer.lr, weight_decay=config.trainer.weight_decay
)

scheduler = ConstantLR(
    optimizer=optimizer,
    factor=config.trainer.schedulers.constant_lr.factor,
    total_iters=config.trainer.schedulers.constant_lr.total_iters,
)

best_val_loss = 1000

for epoch in range(config.trainer.epochs):
    avg_loss, metric_score = train_sim(
        epoch=epoch,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        dataloader_train=train_dataloader,
        top_k=config.trainer.image_osm_similarity.top_k,
    )

    if args.log:
        print(f"Avg. Training Loss {avg_loss}")
        wandb.log({"train_loss": avg_loss, "metric_score": metric_score}, commit=False)

    avg_val_loss, metric_val_score = evaluate_sim(
        epoch=epoch,
        model=model,
        split="val",
        dataloader=val_dataloader,
        top_k=config.trainer.image_osm_similarity.top_k,
    )

    if args.log:
        wandb.log({"eval_loss": avg_val_loss, "metric_val_score": metric_val_score})
        print(f"Avg. Validation Loss {avg_val_loss}")

    if avg_val_loss < best_val_loss:
        print(
            f"Saving best model, avg loss went from {best_val_loss} to {avg_val_loss}"
        )

        save_path = os.path.join(output_model_path, "best_model.pth")
        save_model(model, save_path, True, False)

        if allow_config_save:
            # Freeze the model and save the config so that the model can be loaded later without the need to change the config file
            config_save_path = os.path.join(output_model_path, "config.yaml")
            config.trainer["image_osm_similarity"].freeze = True
            config.vision_model.freeze = True
            save_config(config, config_save_path)
            allow_config_save = False

            # Unfreeze the model for training (if needed)
            config.trainer["image_osm_similarity"].freeze = False
            config.vision_model.freeze = False

        best_val_loss = avg_val_loss
