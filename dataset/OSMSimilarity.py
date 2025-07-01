import json
import os

from typing import List, Dict
from PIL import Image
from sentence_transformers import SentenceTransformer
from torch.utils.data import Dataset
from torchvision import transforms as T
from functools import partial

from utils.general_utils import (
    clean_annotation,
    clean_content,
    lower_case_words,
    modify_position_value,
    tokenize_text,
    unique_objects,
)
from utils.model import filter_osm_content


class OSMSentenceTransformerSimilarityDataset(Dataset):
    """
    This dataset loads the OSM data with the corresponding image path, general caption and osm caption.
    """

    def __init__(
        self,
        tokenizer,
        path: str,
        split: str,
        **kwargs,
    ):
        self.tokenize = partial(tokenize_text, tokenizer=tokenizer)
        self.path = path

        self.filter_osm_blacklist = kwargs.get("osm_blacklist", None)
        self.random_erasing = kwargs.get("random_erasing", None)
        self.sentence_embed_config = kwargs.get("sentence_embedding", None)
        self.change_position_value = kwargs.get("change_position_value", False)

        self.osm_embedder = SentenceTransformer(
            "sentence-transformers/" + self.sentence_embed_config.model,
            device="cpu",
        ).eval()

        self.osm_blacklist = self._load_data(self.filter_osm_blacklist.path)
        self.annotations = self._load_data("annotations.json")

        self.keys = list(self.annotations.keys())
        self.split = split
        self.samples = []
        for key, value in self.annotations.items():
            if self.split in value["split"]:
                content = self._load_content(self.path, key)
                if content:
                    if self.change_position_value:
                        content = modify_position_value(content)
                    if self.filter_osm_blacklist.enabled:
                        content = filter_osm_content(
                            content,
                            self.osm_blacklist,
                            self.filter_osm_blacklist.filter_name_keys,
                        )

                    for annot in value["general_annot"]:
                        entry = {
                            "general_annot": annot,
                            "content": content,
                        }

                        if "osm" in value.keys():
                            entry["osm"] = value["osm"]

                        self.samples.append((key, entry))

    @staticmethod
    def _load_data(path: str) -> List[Dict]:
        with open(path, "r") as file:
            data = json.load(file)
        return data

    @staticmethod
    def _load_content(path: str, key: str):
        complete_path = os.path.join(
            path,
            "osm_data",
            key.split("/")[1].split(".png")[0] + ".json",
        )

        content = None
        if os.path.exists(complete_path):
            with open(
                complete_path,
                "r",
            ) as file:
                content = json.load(file)

        return content

    def __len__(self) -> int:
        return len(self.samples)

    def _build_transforms(self):

        if self.random_erasing is None:
            return T.Compose([T.ToTensor()])

        transform_list = []

        transform_list.append(T.ToTensor())

        if self.random_erasing.enabled:
            transform_list.append(T.RandomErasing(p=self.random_erasing.probability))

        return T.Compose(transform_list)

    def __getitem__(self, idx: int) -> Dict:
        image_key, sample = self.samples[idx]
        img_path = os.path.join(self.path + "/images", image_key)

        content = sample["content"]
        annotation = sample["general_annot"]

        image = Image.open(img_path).convert("RGB")

        transform = self._build_transforms()

        image = transform(image)

        clean_osm_content = clean_content(content)
        clean_osm_content = unique_objects(clean_osm_content)

        splitted_annotations = [
            a.replace("-", " ").strip()
            for a in annotation.strip().split(".")
            if a.strip()
        ]
        clean_annots = clean_annotation(splitted_annotations)

        clean_osm_content = lower_case_words(clean_osm_content)
        clean_annots = lower_case_words(clean_annots)

        content_strings = [" ".join(c) for c in clean_osm_content]

        # Extract the embeddings
        osm_embeddings = self.osm_embedder.encode(
            content_strings, normalize_embeddings=True
        )
        L = osm_embeddings.shape[0]
        osm_data_encoded = {
            "input_ids": osm_embeddings,
            "attention_mask": [1] * L,
            "objects_identifiers": [i for i in range(1, L + 1)],
            "objects_text": content_strings,
        }

        annot_embeddings = self.osm_embedder.encode(
            clean_annots, normalize_embeddings=True
        )

        annotation_data_encoded = {
            "input_ids": annot_embeddings,
            "clean_annotations": clean_annots,
        }

        annotation_sample = {
            "image": image,
            "annotation": annotation_data_encoded,
            "content": osm_data_encoded,
        }

        return annotation_sample
