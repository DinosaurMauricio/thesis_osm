import torch
import os
import streamlit as st
import json
from functools import partial
from scipy.stats import skew
from collections import Counter

from utils.config import (
    initialize_model_config,
    load_config,
    load_model,
    set_device_settings,
    set_cuda_settings,
    setup_tokenizer,
)
from utils.dataset import create_dataloaders, load_datasets
from utils.general_utils import tokens_to_text
from torchvision.transforms import functional as F
from dataset.collate import CustomCollateFn

from utils.model import evaluate_metrics

DEVICE = "cuda:1"

PATH_PROJECT = os.path.dirname(os.path.abspath(__file__))
ANNOTATIONS_JSON_PATH = "annotations.json"

CHECKPOINT_FOLDER = "best_model"
CHECKPOINT_MODEL = "best_model.pth"

TEST_DATASET = "train" # Can be "test", "train" or "val"

START_WORD = "gen"


def calculate_text_metrics(split=""):
    with open(ANNOTATIONS_JSON_PATH) as f:
        json_data = json.load(f)

    all_texts = []

    for _, data in json_data.items():
        # if its not empty string
        if split:
            if data["split"] == split:
                for text in data["general_annot"]:
                    all_texts.append(text)
        else:
            for text in data["general_annot"]:
                all_texts.append(text.replace(".", "").replace(",", "").replace(";", "").replace(":", ""))

    all_words = [word.lower() for text in all_texts for word in text.split()]

    # Calculate metrics
    word_lens = [len(word) for word in all_words]
    sent_lens = [len(text.split()) for text in all_texts]
    chars_in_sents = [len(text) for text in all_texts]

    # Calculate averages and maximums
    word_per_sent = round(sum(sent_lens) / len(all_texts)) if all_texts else 0
    char_per_word = round(sum(word_lens) / len(all_words)) if all_words else 0
    char_per_sent = round(sum(chars_in_sents) / len(all_texts)) if all_texts else 0

    longest_sentence = max(sent_lens) if sent_lens else 0
    shortest_sentence = min(sent_lens) if sent_lens else 0
    longest_word = max(word_lens) if word_lens else 0

    # Calculate skewness of sentence lengths
    sentence_skewness = skew(sent_lens) if sent_lens else 0

    word_frequencies = Counter(all_words)

    return {
        "word_per_sent": word_per_sent,
        "char_per_word": char_per_word,
        "char_per_sent": char_per_sent,
        "longest_sentence": longest_sentence,
        "shortest_sentence": shortest_sentence,
        "longest_word": longest_word,
        "sentence_skewness": sentence_skewness,
        "most_common": word_frequencies.most_common(100),
        "less_common": word_frequencies.most_common()[-100:],
    }


def next():
    st.session_state.counter += 1


def prev():
    st.session_state.counter -= 1


def initialize_config():
    set_cuda_settings()
    config_file_to_load =  os.path.join("trainings", CHECKPOINT_FOLDER, "config.yaml")
    config = load_config(PATH_PROJECT, config_file_to_load)
    set_device_settings(config)

    return config


def load_model_and_tokenizer(config):
    llm = config.llm_models[config.arg_llm]
    vision = config.vision_model[config.arg_vision]
    config.device = DEVICE

    tokenizer = setup_tokenizer(llm.path, config.arg_llm)

    osmcap_config = initialize_model_config(
        config, config.arg_llm, config.arg_vision, False, tokenizer, config.device, llm, vision
    )

    image_osm_sim = initialize_model_config(config, "gpt", "clip", False, tokenizer, config.device, config.llm_models["gpt"], config.vision_model["clip"])
    
    model = load_model(osmcap_config, config, img_osm_config=image_osm_sim, master_process=True)


    model.load_state_dict(
        torch.load(
            os.path.join(
                PATH_PROJECT,
                "trainings",
                CHECKPOINT_FOLDER,
                CHECKPOINT_MODEL
            ),
            map_location=DEVICE
        )
    )
    model.eval()
    return model, tokenizer

def _load_dataset():
    dataset_config = config.dataset[config.arg_dataset]

    osm_extra_params = {
        **config.trainer,
        **config.transformations,
    }

    dataset = load_datasets(
        dataset_name=config.arg_dataset,
        dataset_config=dataset_config,
        inference=True,
        tokenizer=tokenizer,
        splits=(TEST_DATASET,),
        **osm_extra_params
    )[TEST_DATASET]

    custom_collate_fn = CustomCollateFn(sentence_embedding=config.trainer.sentence_embedding, inference=True)    
        
    dataloader = create_dataloaders(
        {
            TEST_DATASET: (dataset, 32),
        },
        False,
        None,
        collate_fn=custom_collate_fn)[TEST_DATASET]
    
    return dataloader


def load_generated_captions():
    results = []
    
    metrics, results = evaluate_metrics(
        model=model,
        dataloader=dataloader,
        split=TEST_DATASET,
        epoch="best_model",
        distributedgpu=False,
        master_process=True,
        start_word=START_WORD,
        return_results=True,
    )
    
    for i, val in enumerate(metrics[0]):
        print(f"Bleu {i+1}: {val}")

    return results


st.title("RS Captioning")

with st.spinner("Calculating text metrics..."):
    text_metrics = calculate_text_metrics()
    st.write("Text Metrics")
    st.write(f"Average words per sentence: {text_metrics['word_per_sent']}")
    st.write(f"Average characters per word: {text_metrics['char_per_word']}")
    st.write(f"Average characters per sentence: {text_metrics['char_per_sent']}")
    st.write(f"Longest sentence: {text_metrics['longest_sentence']}")
    st.write(f"Shortest sentence: {text_metrics['shortest_sentence']}")
    st.write(f"Longest word: {text_metrics['longest_word']}")
    st.write(f"Sentence skewness: {text_metrics['sentence_skewness']}")
    st.write(f"Most Common: {text_metrics['most_common']}")
    st.write(f"Less Common: {text_metrics['less_common']}")

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

    decode_tokens = partial(tokens_to_text, tokenizer=tokenizer)
    
    if "dataloader" not in st.session_state:
        st.session_state["dataloader"] = _load_dataset()

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
    
prediction_result_sentence = results[st.session_state.counter]["prediction"]
image_key = results[st.session_state.counter]["image_key"]

st.write(f"Image {st.session_state.counter + 1}//{len(results)}")
st.write(f"Image key: {image_key}")

st.image(
    F.to_pil_image(results[st.session_state.counter]["image"]),
    caption="Image",
    use_column_width=True,
)

st.write(
    f"**Prediction**: <span style='color:red'>{prediction_result_sentence}</span>",
    unsafe_allow_html=True,
)

st.write("**Ground Truth sentence**\n")
for gt in results[st.session_state.counter]['gts']:
    st.write(gt)
    
st.write(f"**OSM Content**: {results[st.session_state.counter]['osm_content']}")
# st.write("Raw Data:")
# st.write(results[st.session_state.counter])
