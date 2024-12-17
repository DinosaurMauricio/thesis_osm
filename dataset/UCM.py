import torch
import os
import json
from PIL import Image
from torchvision import transforms as T


class UCMDataset(torch.utils.data.Dataset):
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
