"""
These are all model utilities that are used to load a model, process inputs and outputs, and other model related tasks.
"""

import torch
import torch.nn as nn
from tqdm import tqdm
from .general_utils import tokens_to_text, b_score
from .metrics import calculate_compability_metric, evaluate_top_k

def calculate_right_model_weights(total_parameters, is_loaded_in_4bit=False):
    # Needs to be imported here to avoid error with CUDA
    import bitsandbytes as bnb

    # Check if the model is loaded in 4 bits
    total_numel = []
    for param in total_parameters:
        if is_loaded_in_4bit and isinstance(param, bnb.nn.Params4bit):
            total_numel.append(param.numel() * 2)
        else:
            total_numel.append(param.numel())

    return sum(total_numel)

class SquaredReLU(nn.Module):
    """Squared ReLU activation function"""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.pow(torch.relu(x), 2)


def feed_forward_layer(dim: int, mult: int = 4, activation: str = "gelu"):
    """Feed forward layer with given activation function"""

    activations = dict(gelu=nn.GELU, sqrelu=SquaredReLU, relu=nn.ReLU)
    assert (
        activation in activations
    ), f"activation can only be one of {activations.keys()}"

    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias=False),
        activations[activation](),
        nn.Linear(inner_dim, dim, bias=False),
    )

def save_model(model, output_dir, master_process=False, using_distributedgpu=False):
    # Save model for Distributed Data Parallel (DDP) training
    if using_distributedgpu and master_process:
        torch.save(
            # Reminder:
            # - Use model.state_dict() when saving checkpoints for resuming DDP training.
            # - Use model.module.state_dict() for saving models intended for testing.
            model.module.state_dict(),
            output_dir,
        )
    else:
        # Save model for single GPU training
        torch.save(model.state_dict(), output_dir),

def evaluate_metrics(
    model,
    dataloader,
    split,
    epoch,
    start_word:str,
    distributedgpu:bool=False,
    master_process:bool=False,
    max_new_tokens:int=100,
    return_results:bool=False
):
    progress_bar = tqdm(
        dataloader,
        desc=f"Calculating {split} Bleu of epoch {epoch}",
        disable=not master_process,
    )

    model.eval()
    model_device = model.device
    
    predictions = {}
    ground_truths = {}

    if return_results:
        results = []

    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
            for batch in progress_bar:
                images = batch["images"].to(model_device)
                
                image_keys = batch["image_keys"]
                
                inputs = {
                    "images": images,
                    "images_mask": batch["images_mask"].to(model_device),
                    "osm_content": batch["osm_content"].to(model_device),
                    "osm_attention_mask": batch["osm_attention_mask"].to(model_device),
                    "osm_object_identifiers": batch["osm_object_identifiers"].to(model_device),
                    "max_new_tokens": max_new_tokens,
                    "start_word": start_word,
                }

                # Predict the captions
                preds = (
                    model.module.generate_caption(**inputs)
                    if distributedgpu
                    else model.generate_caption(**inputs)
                )
                
                gts = batch["gts"]

                tokenizer = model.module.tokenizer if distributedgpu else model.tokenizer
                predicted_captions = tokens_to_text(preds, tokenizer=tokenizer)
                for i in range(len(predicted_captions)):
                    predictions[image_keys[i]] = predicted_captions[i]

                for i in range(len(gts)):
                    ground_truths[image_keys[i]] = gts[i]
                
                if return_results:
                    objects_texts = batch["objects_texts"]
                    image_keys = batch["image_keys"]
                    for i in range(len(images)):
                        data = {
                            "image_key": image_keys[i],
                            "image": images[i],
                            "prediction": predicted_captions[i],
                            "osm_content": objects_texts[i],
                            "gts": gts[i],
                        }

                        results.append(data)

    gts = {key: ground_truths[key] for key in ground_truths.keys()}
    res = {key: [predictions[key]] for key in predictions.keys()}

    bleu_score = b_score(gts, res)
    
    compatibility_metric = calculate_compability_metric(dataloader, predictions)
    
    progress_bar.close()

    if return_results:
        return (bleu_score,compatibility_metric), results
    else:
        return (bleu_score,compatibility_metric)


def train(
    epoch:int,
    model,
    optimizer,
    scheduler,
    dataloader_train,
    dataloader_inference,
    master_process:bool,
    distributedgpu:bool=False,
    use_mixed_precision:bool=False,
    start_word:str="gen",
):
    model.train()
    model_device = model.device
    scaler = torch.cuda.amp.GradScaler(enabled=use_mixed_precision)

    if distributedgpu:
        # At the beginning of each epoch is necessary to make shuffling work properly across multiple epochs.
        # Otherwise, the same ordering will be used in each epoch.
        dataloader_train.sampler.set_epoch(epoch)

    total_perplexity = 0.0
    progress_bar = tqdm(
        dataloader_train,
        desc=f"Training Epoch {epoch}",
        postfix={"Perplexity": 0.0},
        disable=not master_process,
    )

    for _, batch in enumerate(progress_bar):

        optimizer.zero_grad(set_to_none=True)

        inputs = {
            "images": batch["images"].to(model_device),
            "images_mask": batch["images_mask"].to(model_device),
            "input_ids": batch["input_ids"].to(model_device),
            "input_attention_mask": batch["input_attention_mask"].to(model_device),
            "osm_content": batch["osm_content"].to(model_device),
            "osm_attention_mask": batch["osm_attention_mask"].to(model_device),
            "osm_object_identifiers": batch["osm_object_identifiers"].to(model_device),
            "labels": batch["labels"].to(model_device),
        }

        if scaler.is_enabled():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                outputs = model(**inputs)
                loss = outputs.loss
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                scaler.step(optimizer)
                scaler.update()

        else:
            outputs = model(**inputs)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()

        perp = torch.exp(loss).item() # Calculate perplexity

        total_perplexity += perp
        progress_bar.set_postfix({"Perplexity": perp})

        scheduler.step()

    avg_perplexity = total_perplexity / len(dataloader_train)

    progress_bar.close()

    # Evaluate bleu
    metrics = None
    if dataloader_inference is not None:
        metrics = evaluate_metrics(
            model=model,
            dataloader=dataloader_inference,
            split="train",
            epoch=epoch,
            distributedgpu=distributedgpu,
            master_process=master_process,
            max_new_tokens=150,
            start_word=start_word,
            return_results=False
        )

    return avg_perplexity, metrics

def evaluate(
    epoch:int, 
    model,
    dataloader_eval,
    dataloader_eval_inference,
    master_process:bool,
    distributedgpu:bool,
    split:str,
    start_word:str="gen",
    ):
    
    progress_bar = tqdm(
        dataloader_eval,
        desc=f"Validating on {split}, model epoch {epoch}",
        postfix={"Perplexity": 0.0},
        disable=not master_process,
    )
    model_device = model.device
    model.eval()
    total_perplexity = 0.0
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
            for batch in progress_bar:
                inputs = {
                    "images": batch["images"].to(model_device),
                    "images_mask": batch["images_mask"].to(model_device),
                    "input_ids": batch["input_ids"].to(model_device),
                    "input_attention_mask": batch["input_attention_mask"].to(model_device),
                    "osm_content": batch["osm_content"].to(model_device),
                    "osm_attention_mask": batch["osm_attention_mask"].to(model_device),
                    "osm_object_identifiers": batch["osm_object_identifiers"].to(model_device),
                    "labels": batch["labels"].to(model_device),
                }
                outputs = model(**inputs)
                loss = outputs.loss
                perp = torch.exp(loss).item() # Calculate perplexity

                total_perplexity += perp
                progress_bar.set_postfix({"Perplexity": perp})


    avg_perplexity = total_perplexity / len(dataloader_eval)

    progress_bar.close()

    # Evaluate bleu
    metrics = None
    if dataloader_eval_inference is not None:
        metrics = evaluate_metrics(
            model=model,
            dataloader=dataloader_eval_inference,
            split="val",
            epoch=epoch,
            distributedgpu=distributedgpu,
            master_process=master_process,
            start_word=start_word,
            return_results=False
        )

    return avg_perplexity, metrics
  
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

def filter_osm_content(content, blacklist, filter_name_keys=True):
    if not content:
        return None

    filtered_content = []

    for item in content:
        # Create a new dictionary excluding blacklisted keys with value 0
        filtered_item = {
            key: value
            for key, value in item.items()
            if key not in blacklist
            or blacklist[key] == 1
            or (blacklist[key] == 2 and not filter_name_keys)
        }

        if not (
            len(filtered_item) == 1 and "position" in filtered_item.keys()
        ):  # Removing objects where only the position is left!
            filtered_content.append(filtered_item)

    return filtered_content
