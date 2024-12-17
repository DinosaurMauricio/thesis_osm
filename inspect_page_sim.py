import torch
import os
import streamlit as st

from tqdm import tqdm

from dataset.OSMSimilarity import OSMSentenceTransformerSimilarityDataset
from dataset.collate import custom_collate_sim_fn
from model.image_osm_sim import ImageOSMSim
from utils.config import initialize_model_config, load_config
from transformers import AutoTokenizer

from torchvision.transforms import functional as F

from utils.dataset import create_dataloaders


PATH_PROJECT = os.path.dirname(os.path.abspath(__file__))
# Right now this is hardcoded as only using the gpt and clip model, haven't implemented the other models
LLM = "gpt"
DATASET = "osm"
VISION = "clip"
ANNOTATIONS_JSON_PATH = "annotations.json"
CHECKPOINT = "best_model"
TOP_K = 5


def next():
    st.session_state.counter += 1


def prev():
    st.session_state.counter -= 1


def initialize_config():
    config = load_config(PATH_PROJECT, f"similarity_trainings/{CHECKPOINT}/config.yaml")

    return config


def load_model_and_tokenizer(config):

    llm = config.llm_models[LLM]
    vision = config.vision_model[VISION]

    tokenizer = AutoTokenizer.from_pretrained(
        "gpt2",
        use_fast=True,
        clean_up_tokenization_spaces=True,
    )
    tokenizer.pad_token = tokenizer.eos_token

    osmcap_config = initialize_model_config(
        config, LLM, VISION, False, tokenizer, config.device, llm, vision
    )

    model = ImageOSMSim(osmcap_config)

    model.load_state_dict(
        torch.load(
            os.path.join(
                PATH_PROJECT,
                config.similarity_output_dir,
                CHECKPOINT,
                "best_model.pth",
            )
        )
    )
    model.eval()
    return model, tokenizer


def load_generated_captions():
    model.eval()
    results = []

    progress_bar = tqdm(
        dataloader,
        desc=f"Processing batches",
    )

    with torch.no_grad():
        results = []
        for batch in progress_bar:
            contents = batch["contents"]
            annotations = batch["annotations"]
            images = batch["images"].to(model.config.device)
            object_texts = batch["objects_text"]

            batch_results = model.infer_most_similar_osm_words(contents, images)
            ground_truth = model.get_ground_truth(annotations, contents)

            values = ground_truth.cpu().tolist()
            batch_values = batch_results.cpu().tolist()

            for img, gt_idxs, pred_idxs, texts in zip(
                batch["images"], values, batch_values, object_texts
            ):
                number_of_objects = len(texts)
                preds = [texts[idx] for idx in pred_idxs[:number_of_objects]]
                gts = [texts[idx] for idx in gt_idxs[:number_of_objects]]

                results.append(
                    {
                        "image": img,
                        "indices_gts": gt_idxs[:number_of_objects],
                        "indices_preds": pred_idxs[:number_of_objects],
                        "objects_gt": gts,
                        "objects_pred": preds,
                        "objects_text": texts,
                    }
                )

        progress_bar.close()

    return results


def load_dataset():
    dataset_config = config.dataset[DATASET]

    dataset = OSMSentenceTransformerSimilarityDataset(
        tokenizer, dataset_config.path, "val", **config.trainer
    )

    return dataset


def load_dataloader():
    dataset = load_dataset()
    
    val_dataloader = create_dataloaders(
        {
            "val": (dataset, config.trainer.batch_size),
        },
        False,
        seed=None,
        collate_fn=custom_collate_sim_fn,
    )["val"]

    return val_dataloader


st.title("Image OSM Similarity")


with st.spinner("Generating captions..."):
    if "config" not in st.session_state:
        st.session_state["config"] = initialize_config()

    config = st.session_state["config"]

    if "model" not in st.session_state and "tokenizer" not in st.session_state:
        st.session_state["model"], st.session_state["tokenizer"] = (
            load_model_and_tokenizer(config)
        )

    model = st.session_state["model"]
    tokenizer = st.session_state["tokenizer"]

    if "dataloader" not in st.session_state:
        st.session_state["dataloader"] = load_dataloader()

    dataloader = st.session_state["dataloader"]

    if "results" not in st.session_state:
        print("Loading results")
        st.session_state["results"] = load_generated_captions()
    results = st.session_state["results"]

if "counter" not in st.session_state:
    empty_index = 0
    st.session_state.counter = empty_index

cols = st.columns(3)
with cols[2]:
    st.button("Next ➡️", on_click=next, use_container_width=True)
with cols[0]:
    st.button("⬅️ Previous", on_click=prev, use_container_width=True)

current_result = results[st.session_state.counter]

st.write(f"Image {st.session_state.counter + 1} / {len(results)}")
st.image(
    F.to_pil_image(current_result["image"]),
    caption="Image",
    use_column_width=True,
)
st.write(f"**Ground Truth Indices**: {current_result['indices_gts']}")
st.write(f"**Ground Truth OSM Content**: {current_result['objects_gt']}")
st.write(f"**Predicted Indices**: {current_result['indices_preds']}")
st.write(f"**Predicted OSM Content**: {current_result['objects_pred']}")
st.write(f"**Number of OSM Text**: {len(current_result['objects_text'])}")
st.write(f"**OSM Text**: {current_result['objects_text']}")
