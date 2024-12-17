import torch
import torch.nn as nn
from tqdm import tqdm
from .general_utils import tokens_to_text, b_score
from .metrics import calculate_compability_metric, evaluate_top_k


def train_sim(
    epoch: int,
    model,
    optimizer,
    scheduler,
    dataloader_train,
    top_k: int,
):
    model.train()
    model_device = model.device

    total_loss = 0.0
    progress_bar = tqdm(
        dataloader_train, desc=f"Training Epoch {epoch}", postfix={"Loss": 0.0}
    )

    for _, batch in enumerate(progress_bar):

        optimizer.zero_grad(set_to_none=True)

        inputs = {
            "images": batch["images"].to(model_device),
            "annotations": batch["annotations"],
            "contents": batch["contents"],
        }

        ground_truth, prediction = model(**inputs)

        # Calculate cosine similarity loss
        cosine_sim = torch.nn.functional.cosine_similarity(
            ground_truth, prediction, dim=1
        )
        loss = (1 - cosine_sim).mean()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix({"Loss": loss.item()})

        scheduler.step()

    avg_loss = total_loss / len(dataloader_train)

    progress_bar.close()

    metric_score = evaluate_top_k(
        model=model,
        dataloader=dataloader_train,
        split="train",
        top_k=top_k,
        percentage=0.1,
    )

    return avg_loss, metric_score


def evaluate_sim(epoch: int, dataloader, model, top_k: int, split="val"):
    progress_bar = tqdm(
        dataloader,
        desc=f"{split} Epoch {epoch}",
        postfix={"Loss": 0.0},
    )
    model_device = model.device
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
            for batch in progress_bar:
                inputs = {
                    "images": batch["images"].to(model_device),
                    "annotations": batch["annotations"],
                    "contents": batch["contents"],
                }
                ground_truth, prediction = model(**inputs)

                # Calculate cosine similarity loss
                cosine_sim = torch.nn.functional.cosine_similarity(
                    ground_truth, prediction, dim=1
                )
                loss = (1 - cosine_sim).mean()

                total_loss += loss.item()
                progress_bar.set_postfix({"Loss": loss.item()})

    avg_loss = total_loss / len(dataloader)

    progress_bar.close()

    metric_score = evaluate_top_k(
        model=model, dataloader=dataloader, split=split, top_k=top_k
    )

    return avg_loss, metric_score
