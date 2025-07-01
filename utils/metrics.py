import itertools
import math
import torch
import numpy as np

from tqdm import tqdm
from sentence_transformers import SentenceTransformer


def precision_at_k(similarity_scores, ground_truth_scores, top_k):

    k = (
        top_k
        # check the top_k is less than the number of osm words if not take all
        if top_k is not None and top_k < ground_truth_scores.shape[-1]
        else ground_truth_scores.shape[-1]
    )

    _, top_k_pred_indices = torch.topk(similarity_scores, k, dim=1)
    _, top_k_gt_indices = torch.topk(ground_truth_scores, k, dim=1)

    precisions = []
    for i in range(similarity_scores.size(0)):
        relevant_at_k = torch.isin(
            top_k_pred_indices[i], top_k_gt_indices[i]
        ).float()  # Check if top-k items are relevant
        precisions.append(
            relevant_at_k.mean().item()
        )  # Calculate precision for the current example

    return sum(precisions) / len(precisions)


def evaluate_top_k(model, dataloader, split, top_k, percentage=1.0):
    model.eval()
    results = []

    total_batch = len(dataloader)
    limited_batches = math.ceil(total_batch * percentage)

    # Use itertools to limit the DataLoader
    limited_dataloader = itertools.islice(dataloader, limited_batches)

    progress_bar = tqdm(
        limited_dataloader,
        desc=f"Evaluating {split} top-{top_k}",
        total=limited_batches,
    )

    with torch.no_grad():
        for batch in progress_bar:
            inputs = {
                "images": batch["images"].to(model.config.device),
                "annotations": batch["annotations"],
                "contents": batch["contents"],
            }

            ground_truth, prediction = model(**inputs)

            precision_at_k_score = precision_at_k(prediction, ground_truth, top_k)

            results.append(precision_at_k_score)

        progress_bar.close()

        # average precision at k
        return sum(results) / len(results)


def calculate_compability_metric(dataloader, predictions):
    emb_model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
    tot_sim = 0

    not_discarded_total = 0
    for items in tqdm(dataloader):
        for image_key, texts in zip(items["image_keys"], items["compatibility_texts"]):

            if "Discarded" in texts or not texts:
                continue
            not_discarded_total += 1
            
            embedded_objects = emb_model.encode(texts, normalize_embeddings=True)

            pieces = [
                piece.strip()
                for piece in predictions[image_key].split(".")
                if piece.strip()
            ]

            # Encode all pieces
            encoded_pieces = emb_model.encode(pieces, normalize_embeddings=True)

            similarities = embedded_objects @ encoded_pieces.T

            max_sim = np.max(similarities, axis=1)
            tot_sim += np.mean(max_sim).item()

    compatibility_metric = tot_sim / not_discarded_total

    return compatibility_metric
