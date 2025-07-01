import torch
import torch.nn.functional as F
from .constants import PAD_TOKEN_ID, IGNORE_INDEX, OBJ_IDENTIFIER_PAD_TOKEN
import numpy as np


class CustomCollateFn:
    def __init__(
        self,
        sentence_embedding,
        inference: bool = False,
    ):
        self.use_osm_embeddings = sentence_embedding.enabled  # This flag is used to know if we are embeddings osm info with sentence transformer or not.
        self.emb_dim = sentence_embedding.dim
        self.inference = inference

    def __call__(self, batch):
        # Extract input_ids from each element and find the maximum length among them
        if not self.inference:
            input_tokens = [e["annotation"]["input_ids"] for e in batch]
            input_tokens_maxlen = max([len(t) for t in input_tokens])
            input_ids_list = []
            attention_mask_list = []

        osm_data = [e["osm_data"]["input"] for e in batch if e["osm_data"] is not None]
        osm_tokens_maxlen = max([len(t) for t in osm_data]) if len(osm_data) != 0 else 1

        osm_content_list = []
        osm_attention_mask_list = []
        osm_object_identifiers = []
        labels_list = []
        images_list = []
        images_mask = []
        gts_list = []
        objects_texts = []
        compatibility_texts = []

        if self.inference:
            img_keys_list = []

        if self.use_osm_embeddings:
            PAD_TOKEN = np.zeros((self.emb_dim))
        else:
            PAD_TOKEN = PAD_TOKEN_ID

        for sample in batch:
            if not self.inference:
                # INPUT IDS
                input_ids = sample["annotation"]["input_ids"]
                attention_mask = sample["annotation"]["attention_mask"]
                # Calculate the padding length required to match the maximum token length
                input_pad_len = input_tokens_maxlen - len(input_ids)
                # Pad 'input_ids' with the pad token ID, 'labels' with IGNORE_INDEX, and 'attention_mask' with 0
                input_ids_list.append(input_ids + input_pad_len * [PAD_TOKEN_ID])
                labels_list.append(input_ids + input_pad_len * [IGNORE_INDEX])
                attention_mask_list.append(attention_mask + input_pad_len * [0])
            else:
                compatibility_texts.append(sample["compatibility_text"])

            # OSM DATA
            if sample["osm_data"] is not None:
                osm_input = sample["osm_data"]["input"]
                osm_attention_mask = sample["osm_data"]["attention_mask"]
                object_identifiers = sample["osm_data"]["objects_identifiers"]
                osm_pad_len = osm_tokens_maxlen - len(osm_input)
                # Append the text to the osmtext
                objects_texts.append(sample["osm_data"]["objects_text"])
            else:
                osm_input = [PAD_TOKEN] * osm_tokens_maxlen
                object_identifiers = [OBJ_IDENTIFIER_PAD_TOKEN] * osm_tokens_maxlen
                osm_attention_mask = [0] * osm_tokens_maxlen
                osm_pad_len = 0
                objects_texts.append(["Discarded"])

            # Do the same with "osm_input_ids", but here we don't have labels
            osm_content_list.append(osm_input + osm_pad_len * [PAD_TOKEN])
            osm_attention_mask_list.append(osm_attention_mask + osm_pad_len * [0])
            osm_object_identifiers.append(
                object_identifiers + osm_pad_len * [OBJ_IDENTIFIER_PAD_TOKEN]
            )

            images_list.append(sample["image"])
            gts_list.append(sample["gts"])

            if sample["take_img"]:
                images_mask.append(
                    torch.ones(
                        1,
                    )
                )
            else:
                images_mask.append(
                    torch.zeros(
                        1,
                    )
                )

            if self.inference:
                img_keys_list.append(sample["image_key"])

        if not self.inference:
            batch = {
                "input_ids": torch.tensor(input_ids_list, dtype=torch.long),
                "input_attention_mask": torch.tensor(
                    attention_mask_list, dtype=torch.long
                ),
                "osm_content": torch.tensor(
                    np.array(osm_content_list)
                ),  # Not specifying the dtype since it is inferred
                "osm_attention_mask": torch.tensor(
                    osm_attention_mask_list, dtype=torch.long
                ),
                "osm_object_identifiers": torch.tensor(
                    osm_object_identifiers, dtype=torch.long
                ),
                "images": torch.stack(images_list),
                "images_mask": torch.stack(images_mask),
                "labels": torch.tensor(labels_list, dtype=torch.long),
                "objects_texts": objects_texts,
                "gts": gts_list,
            }
        else:
            batch = {
                "osm_content": torch.tensor(
                    np.array(osm_content_list)
                ),  # Not specifying the dtype since it is inferred
                "osm_attention_mask": torch.tensor(
                    osm_attention_mask_list, dtype=torch.long
                ),
                "osm_object_identifiers": torch.tensor(
                    osm_object_identifiers, dtype=torch.long
                ),
                "images": torch.stack(images_list),
                "images_mask": torch.stack(images_mask),
                "labels": torch.tensor(labels_list, dtype=torch.long),
                "objects_texts": objects_texts,
                "gts": gts_list,
                "image_keys": img_keys_list,
                "compatibility_texts": compatibility_texts,
            }

        return batch


class CustomCollateFn_annotation_instead_of_osm:
    def __init__(self):
        pass

    def __call__(self, batch):
        # Extract input_ids from each element and find the maximum length among them
        input_tokens = [e["annotation"]["input_ids"] for e in batch]
        input_tokens_maxlen = max([len(t) for t in input_tokens])
        osm_tokens = [
            e["osm_data"]["input"] for e in batch if e["osm_data"] is not None
        ]  # Because we can randomly mask the use of OSM
        osm_tokens_maxlen = (
            max([len(t) for t in osm_tokens]) if len(osm_tokens) != 0 else 1
        )

        input_ids_list = []
        attention_mask_list = []
        osm_content_ids_list = []
        osm_attention_mask_list = []
        osm_object_identifiers = []
        labels_list = []
        images_list = []
        images_mask = []
        gts_list = []

        for sample in batch:
            input_ids = sample["annotation"]["input_ids"]
            attention_mask = sample["annotation"]["attention_mask"]

            if sample["osm_data"] is not None:
                osm_input_ids = sample["osm_data"]["input"]
                osm_attention_mask = sample["osm_data"]["attention_mask"]
                object_identifiers = sample["osm_data"]["objects_identifiers"]
                osm_pad_len = osm_tokens_maxlen - len(osm_input_ids)
            else:
                osm_input_ids = [PAD_TOKEN_ID] * osm_tokens_maxlen
                object_identifiers = [OBJ_IDENTIFIER_PAD_TOKEN] * osm_tokens_maxlen
                osm_attention_mask = [0] * osm_tokens_maxlen
                osm_pad_len = 0

            # Calculate the padding length required to match the maximum token length
            input_pad_len = input_tokens_maxlen - len(input_ids)

            # Pad 'input_ids' with the pad token ID, 'labels' with IGNORE_INDEX, and 'attention_mask' with 0
            input_ids_list.append(input_ids + input_pad_len * [PAD_TOKEN_ID])
            labels_list.append(input_ids + input_pad_len * [IGNORE_INDEX])
            attention_mask_list.append(attention_mask + input_pad_len * [0])
            # Do the same with "osm_input_ids", but here we don't have labels
            osm_content_ids_list.append(osm_input_ids + osm_pad_len * [PAD_TOKEN_ID])
            osm_attention_mask_list.append(osm_attention_mask + osm_pad_len * [0])
            osm_object_identifiers.append(
                object_identifiers + osm_pad_len * [OBJ_IDENTIFIER_PAD_TOKEN]
            )

            images_list.append(sample["image"])
            gts_list.append(sample["gts"])

            if sample["take_img"]:
                images_mask.append(
                    torch.ones(
                        1,
                    )
                )
            else:
                images_mask.append(
                    torch.zeros(
                        1,
                    )
                )

        batch = {
            "input_ids": torch.tensor(input_ids_list, dtype=torch.long),
            "input_attention_mask": torch.tensor(attention_mask_list, dtype=torch.long),
            "osm_content_ids": torch.tensor(input_ids_list, dtype=torch.long),
            "osm_attention_mask": torch.tensor(attention_mask_list, dtype=torch.long),
            "images": torch.stack(images_list),
            "images_mask": torch.stack(images_mask),
            "labels": torch.tensor(labels_list, dtype=torch.long),
            "gts": gts_list,
        }

        return batch


def custom_collate_sim_fn(batch):
    clean_annotation = []
    clean_content = []
    images_list = []
    objects_text = []

    max_annot_length = max(
        [sample["annotation"]["input_ids"].shape[0] for sample in batch]
    )
    max_content_length = max(
        [sample["content"]["input_ids"].shape[0] for sample in batch]
    )

    for sample in batch:
        input_ids = sample["annotation"]["input_ids"]
        osm_input_ids = sample["content"]["input_ids"]
        object_texts = sample["content"]["objects_text"]

        input_padded = F.pad(
            torch.tensor(input_ids), (0, 0, 0, max_annot_length - input_ids.shape[0])
        )

        osm_padded = F.pad(
            torch.tensor(osm_input_ids),
            (0, 0, 0, max_content_length - osm_input_ids.shape[0]),
        )

        clean_annotation.append(input_padded)
        clean_content.append(osm_padded)
        images_list.append(sample["image"])
        objects_text.append(object_texts)

    batch = {
        "annotations": torch.stack(clean_annotation),
        "contents": torch.stack(clean_content),
        "images": torch.stack(images_list),
        "objects_text": objects_text,
    }

    return batch
