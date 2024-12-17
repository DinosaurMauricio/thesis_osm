import random
from utils.general_utils import clean_content, unique_objects, remove_special_characters


class OSMProcessor:
    def __init__(
        self,
        tokenizer,
        osm_embedder=None,
        sentence_embedding=None,
        embed_single_objects=None,
    ):
        self.sentence_embedding = sentence_embedding
        self.tokenizer = tokenizer
        self.osm_embedder = osm_embedder
        self.embed_single_objects = embed_single_objects

    def _clean_content(self, content):
        """
        Clean content from punctuation or special characters
        while also just retaning unique objects and removing repeated ones
        """
        clean_osm = clean_content(content)
        clean_osm = unique_objects(clean_osm)

        return clean_osm

    def content_to_strings(self, content):
        """
        OSM Content format is in JSON, preprocess it into a string of key value while removing special characters
        """
        strings = []
        for element in content:
            str = " ".join(
                f"{k} {v}" if k != "position" else f"{v}" for k, v in element.items()
            )
            str = remove_special_characters(str)
            strings.append(str)

        strings = list(set(strings))
        return strings

    def tokenize_content(self, content):
        """
        Returns embeddings or sentence embeddings of the OSM data
        """
        if not content:
            return None

        strings = self.content_to_strings(content)

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
                strings = [" ".join(str for str in strings)]

            # Compute the embeddings
            strings_embeddings = self.osm_embedder.encode(
                strings, normalize_embeddings=False
            )

            L = strings_embeddings.shape[0]

            osm_data_encoded = {
                "input": list(strings_embeddings),
                "attention_mask": [1] * L,
                "objects_identifiers": [i for i in range(1, L + 1)],
                "objects_text": strings,
            }

        return osm_data_encoded

    def get_content_for_similairty(self, content):
        """
        If the OSM Similairty component is used we process the OSM embeddings
        """
        clean_content = self._clean_content(content)
        # due the complexity of the position it can bring, we ignore it for now
        content_strings = [
            " ".join(item for item in c if item != "position") for c in clean_content
        ]

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
