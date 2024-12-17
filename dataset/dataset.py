"""
This utilities create the dataset in Hugginface format. 
"""

import json
import os
import random
import torch
import numpy as np

from sentence_transformers import SentenceTransformer


from .constants import POSITION_MAPPING
from typing import List, Dict
from PIL import Image

from torchvision import transforms as T
from utils.general_utils import (
    clean_content,
    modify_position_value,
    random_choice,
    remove_special_characters,
    unique_objects,
)
from utils.model import filter_osm_content

from sentence_transformers import SentenceTransformer


class OSMDataset(torch.utils.data.Dataset):
    """
    This dataset loads the OSM data with the corresponding image path, general caption and osm caption.
    """

    def __init__(
        self,
        tokenizer,
        path: str,
        split: str,
        inference: bool = False,
        **kwargs,
    ):
        self.tokenizer = tokenizer
        self.path = path
        self.annotations = self._load_data("annotations.json")  # Changed
        self.keys = list(self.annotations.keys())
        # This settings are used to control the behavior of the dataset for finding the best configuration on the OSM data,
        # in the future we can remove them.
        self.use_short_sentences = kwargs.get("short_sentences", False)
        self.is_image_osm_similarity = kwargs.get("image_osm_similarity", False)
        self.change_position_value = kwargs.get("change_position_value", False)
        self.sentence_embedding = kwargs.get("sentence_embedding", None)

        self.split = split
        self.inference = inference
        self.samples = []

        for key, value in self.annotations.items():
            if self.split in value["split"]:
                if inference:
                    # Do not unroll the samples
                    general_annot = [cap.split(".")[0] if self.use_short_sentences.enabled else cap for cap in value["general_annot"]]
                    
                    if "osm" in value.keys():
                        osm_annot = [cap.split(".")[0] if self.use_short_sentences.enabled else cap for cap in value["osm"]]
                    else:
                        osm_annot = None
                
                    self.samples.append(
                        (
                            key,
                            {
                                "general_annot": general_annot,
                                "osm": osm_annot,
                            },
                        )
                    )
                else:
                    # Unroll the samples
                    for i in range(len(value["general_annot"])): # Here supposing that the number of general captions and the number of osm captions are same.
                        general_annot = value["general_annot"][i].split(".")[0] if self.use_short_sentences.enabled else value["general_annot"][i]
                        
                        if "osm" in value.keys():
                            osm_annot = value["osm"][i].split(".")[0] if self.use_short_sentences.enabled else value["osm"][i]
                        else:
                            osm_annot = None
                            
                        self.samples.append(
                            (
                                key,
                                {
                                    "general_annot": general_annot,
                                    "osm": osm_annot,
                                },
                            )
                        )

        # Get other parameters set in config.
        self.filter_osm_blacklist = kwargs.get("osm_blacklist", None)

        if self.filter_osm_blacklist is not None:
            self.osm_blacklist = self._load_data(self.filter_osm_blacklist.path)
            self.osm_blacklist["alt_name"] = 0
            self.osm_blacklist["name"] = 0
            self.osm_blacklist["official_name"] = 0
            self.osm_blacklist["old_name"] = 0
            self.osm_blacklist["county"] = 0
            self.osm_blacklist["operator"] = 0
            self.osm_blacklist["brand"] = 0
            self.osm_blacklist["cuisine"] = 0

        if self.sentence_embedding.enabled:
            self.embed_single_objects = kwargs.get("embed_single_objects", None)
            self.use_precomputed_embs = kwargs.get("use_precomputed_embs", None)
            if self.embed_single_objects and self.use_precomputed_embs:
                print(
                    "Using precoumputed embeddings to speed up computations. Be careful that this can lead to errors. In that case, remove the flag from config"
                )
                self.precomputed_embs = np.load("precomputed_embds.npy")
                with open("precomputed_emb_correspondance.json", "r") as file:
                    self.text_correspondance = json.load(file)

            else:
                # Load the model as we will compute on the fly.
                self.osm_embedder = SentenceTransformer(
                    "sentence-transformers/" + self.sentence_embedding.model,
                    device="cpu",
                ).eval()

        self.prob_osm_input = kwargs.get("prob_osm_input", 1)
        self.prob_img_input = kwargs.get("prob_img_input", 1)
        self.prob_osm_target = kwargs.get(
            "prob_osm_target", 1
        )  # Probability with which to take the osm annot as target
        self.force_osm_target = kwargs.get("force_osm_target", 1)

        if self.split == "train":
            self.transformations = 1  # Dummy way to make it not None
            self.gaussian_noise = kwargs.get("gaussian_noise", None)
            self.random_erasing = kwargs.get("random_erasing", None)
        else:
            self.transformations = None

    def _clean_content(self, content):
        clean_osm = clean_content(content)
        clean_osm = unique_objects(clean_osm)

        return clean_osm

    def _gauss_noise_tensor(self, img, std, mean):
        out = img + torch.randn(img.shape) * std + mean
        # Minmax scale it again
        out = (out - out.min()) / (out.max() - out.min())

        return out

    def _should_apply(self, probability, enabled):
        return random_choice(probability) if enabled else False

    def _build_transforms(self):

        if self.transformations is None:
            return T.Compose([T.ToTensor()])

        transform_list = []

        apply_gauss_noise = self._should_apply(
            self.gaussian_noise.probability, self.gaussian_noise.enabled
        )

        transform_list.append(T.ToTensor())

        if self.random_erasing.enabled:
            transform_list.append(T.RandomErasing(p=self.random_erasing.probability))

        if apply_gauss_noise:
            apply_gauss_noise = lambda x: self._gauss_noise_tensor(
                x, self.gaussian_noise.std, self.gaussian_noise.mean
            )
            transform_list.append(apply_gauss_noise)

        return T.Compose(transform_list)

    @staticmethod
    def _load_data(path: str) -> List[Dict]:
        with open(path, "r") as file:
            data = json.load(file)
        return data

    @staticmethod
    def _get_annotation_type(
        osm_data_input: bool, prob_osm_target: float, force_augmented=False
    ) -> str:
        if osm_data_input and random_choice(prob=prob_osm_target) or force_augmented:
            return "osm"
        else:
            return "general_annot"

    @staticmethod
    def _get_annotation(sample: dict, annot_type: bool = False) -> str:
        if annot_type == "osm":
            special_keyword = "aug "
        else:
            special_keyword = "gen "

        annotation = special_keyword + sample[annot_type] + "<|endoftext|>"

        return annotation

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

    @staticmethod
    def _get_ground_truth(self, image_key: str, annot_type: str):

        gts = (
            [sent.split(".")[0] for sent in self.annotations[image_key][annot_type]]
            if self.use_short_sentences.enabled
            else self.annotations[image_key][annot_type]
        )

        return gts
    
    def _content_to_strings(self, content):
        strings = []
        for element in content:
            str = " ".join(
                f"{k} {v}" if k != "position" else f"{v}"
                for k, v in element.items()
            )
            str = remove_special_characters(str)
            strings.append(str)

        strings = list(set(strings))
        return strings

    @staticmethod
    def _tokenize_content(self, content):
        if not content:
            return None
        
        strings = self._content_to_strings(content)

        strings = list(set(strings))
        # Random shuffle of the list to remove the dependency of the order of the osm objects
        random.shuffle(strings)
        # Return also a chunk identifier to implement the object embedding
        osm_ids = []
        osm_attention_mask = []
        object_identifiers = []

        if not self.sentence_embedding.enabled:
            for i, string in enumerate(strings, start=1):
                # Tokenize it
                string_tokenized = self.tokenizer(
                    string, truncation=False, add_special_tokens=False
                )
                osm_ids.extend(string_tokenized["input_ids"])
                osm_attention_mask.extend(string_tokenized["attention_mask"])
                object_identifiers.extend([i] * len(string_tokenized["input_ids"]))

            osm_data_encoded = {
                "input": osm_ids,
                "attention_mask": osm_attention_mask,
                "objects_identifiers": object_identifiers,
                "objects_text": strings,
            }
        else:
            if not self.embed_single_objects:
                strings = [
                    " ".join(str for str in strings)
                ]  # Concat everything in a single string

            if self.embed_single_objects and self.use_precomputed_embs:
                indexes = []
                for str in strings:
                    indexes.append(self.text_correspondance[str])
                    
                strings_embeddings = self.precomputed_embs[indexes, :]
            else:
                # Compute the embeddings on the fly
                strings_embeddings = self.osm_embedder.encode(
                    strings, normalize_embeddings=False
                )  # The embeddings are normalized later

            L = strings_embeddings.shape[0]

            osm_data_encoded = {
                "input": list(strings_embeddings),
                "attention_mask": [1] * L,
                "objects_identifiers": [i for i in range(1, L + 1)],
                "objects_text": strings,
            }

        return osm_data_encoded


    def _get_content_for_similairty(self, content):
        clean_content = self._clean_content(content)
        content_strings = [" ".join(item for item in c if item != "position") for c in clean_content]

        osm_embeddings = self.osm_embedder.encode(
            content_strings, normalize_embeddings=True
        )
        L = osm_embeddings.shape[0]
        osm_data_encoded = {
            "input": list(osm_embeddings),
            "attention_mask": [1] * L,
            "objects_identifiers": [i for i in range(1, L + 1)],
            "objects_text": content_strings,
        }

        return osm_data_encoded

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        image_key, sample = self.samples[idx]

        # IMAGE
        img_path = os.path.join(self.path + "/images", image_key)
        image = Image.open(img_path).convert("RGB")
        transform = self._build_transforms()
        image = transform(image)

        take_img = random_choice(prob=self.prob_img_input)

        # OSM data
        osm_data_input = self._load_content(self.path, image_key)

        if self.change_position_value:
            osm_data_input = modify_position_value(osm_data_input)

        if self.filter_osm_blacklist.enabled:
            osm_data_input = filter_osm_content(
                osm_data_input,
                self.osm_blacklist,
                self.filter_osm_blacklist.filter_name_keys,
            )

        # Random exclusion of the osm data
        osm_data_input_temp = (
            osm_data_input if random_choice(prob=self.prob_osm_input) else None
        )

        # check is enabled and that content is not empty
        if self.is_image_osm_similarity.enabled:
            osm_data_tokenized = self._get_content_for_similairty(osm_data_input_temp) if osm_data_input_temp else None
        else:
            osm_data_tokenized = self._tokenize_content(self, osm_data_input_temp)

        use_osm_data_in_input = osm_data_input_temp != None

        if osm_data_tokenized is None and not take_img:
            # Randomly select one of the two
            if random_choice(prob=0.5):
                take_img = True
            else:
                if self.is_image_osm_similarity.enabled:
                    osm_data_tokenized = self._get_content_for_similairty(osm_data_input) if osm_data_input else None
                else:
                    osm_data_tokenized = self._tokenize_content(self, osm_data_input)

                use_osm_data_in_input = True

        # ANNOTATION
        annot_type = self._get_annotation_type(
            osm_data_input=use_osm_data_in_input,
            prob_osm_target=self.prob_osm_target,
            force_augmented=self.force_osm_target,
        )

        if not self.inference:
            annotation = self._get_annotation(
                sample=sample,
                annot_type=annot_type,
            )
            annotation_tokenized = self.tokenizer(
                annotation, truncation=False, add_special_tokens=False
            )

            gts = self._get_ground_truth(
                self, image_key=image_key, annot_type=annot_type
            )

            annotation_sample = {
                "image": image,
                "annotation": annotation_tokenized,
                "osm_data": osm_data_tokenized,
                "gts": gts,
                "take_img": take_img,
            }

        else:
            gts = sample[annot_type]

            annotation_sample = {
                "image_key": image_key,  # Return also the image key for debugging purposes
                "image": image,
                "osm_data": osm_data_tokenized,
                "gts": gts,
                "take_img": take_img,
                # this is only used for the compability metric, easiest way to handle it without changing the code 
                "compatibility_text": self._content_to_strings(osm_data_input)
            }

        return annotation_sample


class UCM_dataset(torch.utils.data.Dataset):
    """
    Dataset that loads sample of the UCM dataset.
    This is used to test the learning capabilities of the network
    """

    def __init__(self, tokenizer, path: str, split: str, **kwargs):
        self.path = path
        self.tokenizer = tokenizer
        self.split = split
        self._load_data(path)

    def _load_data(self, path: str):
        samples = []
        gts = {}
        with open(os.path.join(path, "dataset.json"), "r") as file:
            data = json.load(file)
            for image in data["images"]:
                if image["split"] == self.split:
                    gts[image["filename"]] = []
                    for sentence in image["sentences"]:
                        samples.append((image["filename"], sentence["raw"]))
                        gts[image["filename"]].append(sentence["raw"])

        self.samples = samples
        self.gts = gts

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        image = Image.open(os.path.join(self.path, "images", sample[0])).convert("RGB")
        annotation = sample[1] + "<|endoftext|>" if self.split != "test" else sample[1]

        transform = T.ToTensor()
        image = transform(image)

        annotation_tokenized = self.tokenizer(
            annotation, truncation=False, add_special_tokens=False
        )

        transform = T.Compose([T.ToTensor()])

        image = transform(image)

        annotation_sample = {
            "image": image,
            "osm_data": None,
            "annotation": annotation_tokenized,
            "gts": self.gts[sample[0]],
        }

        return annotation_sample
