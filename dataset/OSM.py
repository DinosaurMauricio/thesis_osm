import os
import torch

from functools import partial
from sentence_transformers import SentenceTransformer

from typing import Dict
from PIL import Image

from utils.general_utils import (
    modify_position_value,
    random_choice,
)

from .image_transforms import ImageTransforms
from .osm_processor import OSMProcessor

from utils.general_utils import filter_osm_content
from utils.dataset import load_data, load_osm_content, shorten_stentence
from utils.annotations import get_annotation, get_annotation_type


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
        self.annotations = load_data("annotations.json")
        self.keys = list(self.annotations.keys())
        # This settings are used to control the behavior of the dataset
        # for finding the best configuration on the OSM data,
        # in the future we can remove them.
        self.use_short_sentences = kwargs.get("short_sentences", False)
        self.is_image_osm_similarity = kwargs.get("image_osm_similarity", False)
        self.change_position_value = kwargs.get("change_position_value", False)
        self.sentence_embedding = kwargs.get("sentence_embedding", None)
        ################################################################

        self.split = split
        self.inference = inference
        self.samples = []

        # partial sets already enabled so we dont have to send it all the time, its gonna be the same value afterall
        self.shorten = partial(
            shorten_stentence, enabled=self.use_short_sentences.enabled
        )

        for key, value in self.annotations.items():
            if self.split in value["split"]:
                if inference:
                    # Do not unroll the samples
                    general_annot = [
                        self.shorten(cap) for cap in value["general_annot"]
                    ]

                    osm_annot = (
                        [self.shorten(cap) for cap in value["osm"]]
                        if "osm" in value.keys()
                        else None
                    )

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
                    # Here we are assuming that the number of general captions
                    # and the number of osm captions are same so we just take
                    # the len of the general annotation
                    for i in range(len(value["general_annot"])):

                        general_annot = self.shorten(value["general_annot"][i])
                        osm_annot = (
                            self.shorten(value["osm"][i])
                            if "osm" in value.keys()
                            else None
                        )

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
            self.osm_blacklist = load_data(self.filter_osm_blacklist.path)

        if self.sentence_embedding.enabled:
            self.embed_single_objects = kwargs.get("embed_single_objects", None)
            # Load the model as we will compute on the fly.
            osm_embedder = SentenceTransformer(
                "sentence-transformers/" + self.sentence_embedding.model,
                device="cpu",
            ).eval()

        self.osm_processor = OSMProcessor(
            tokenizer, osm_embedder, self.sentence_embedding, self.embed_single_objects
        )

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
            # on other splits it must not apply transforms
            self.transformations = None

    def _get_ground_truth(self, image_key: str, annot_type: str):

        gts = (
            # for testing we shorten it if flag is enabled at first point.
            [sent.split(".")[0] for sent in self.annotations[image_key][annot_type]]
            if self.use_short_sentences.enabled
            else self.annotations[image_key][annot_type]
        )

        return gts

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        image_key, sample = self.samples[idx]

        # IMAGE
        img_path = os.path.join(self.path + "/images", image_key)
        image = Image.open(img_path).convert("RGB")
        transform = ImageTransforms.build_transforms(
            self.transformations, self.gaussian_noise, self.random_erasing
        )
        image = transform(image)

        # Random exclusion of the rs image data
        take_img = random_choice(prob=self.prob_img_input)

        # OSM data
        osm_data_input = load_osm_content(self.path, image_key)

        # Simplify the position naming
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

        # check is enabled and that content is not empty due to the random exclusion on previous line
        if self.is_image_osm_similarity.enabled:

            osm_data_tokenized = (
                self.osm_processor.get_content_for_similairty(osm_data_input)
                if osm_data_input_temp
                else None
            )
        else:
            osm_data_tokenized = self.osm_processor.tokenize_content(
                osm_data_input_temp
            )

        use_osm_data_in_input = osm_data_input_temp != None

        # the image or the osm data were excluded
        if osm_data_tokenized is None and not take_img:
            # Randomly select one of the two
            if random_choice(prob=0.5):
                take_img = True
            else:
                if self.is_image_osm_similarity.enabled:
                    osm_data_tokenized = (
                        self.osm_processor.get_content_for_similairty(osm_data_input)
                        if osm_data_input
                        else None
                    )
                else:
                    osm_data_tokenized = self.osm_processor.tokenize_content(
                        osm_data_input
                    )

                use_osm_data_in_input = True

        # ANNOTATION
        annot_type = get_annotation_type(
            osm_data_input=use_osm_data_in_input,
            prob_osm_target=self.prob_osm_target,
            force_augmented=self.force_osm_target,
        )

        if not self.inference:
            annotation = get_annotation(
                sample=sample,
                annot_type=annot_type,
            )
            annotation_tokenized = self.tokenizer(
                annotation, truncation=False, add_special_tokens=False
            )

            gts = self._get_ground_truth(image_key=image_key, annot_type=annot_type)

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
                "compatibility_text": self.osm_processor.content_to_strings(
                    osm_data_input
                ),
            }

        return annotation_sample
