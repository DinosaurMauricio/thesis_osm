import torch
from .clip_vision import CLIPVision

from model.config.osmcap import OSMCAP_Config
from model.gpt2_model import GPT2Model
from utils.model import calculate_right_model_weights

from torch import nn


class ImageOSMSim(nn.Module):
    def __init__(self, config: OSMCAP_Config):
        super().__init__()
        self.config = config
        self.language = self._load_llm(config)
        self.vision = self._load_vision_model(config)
        self.device = config.device

        self.cos_sim = nn.CosineSimilarity(dim=-1)
        self.linear = nn.Linear(
            config.vision.hidden_size,
            config.trainer.sentence_embedding.dim,
            device=self.device,
        )

        # Freeze all parameters in the model
        if self.config.image_osm_similarity.freeze:
            for param in self.parameters():
                param.requires_grad = False
        else:
            self._print_model_weights()

    def _print_model_weights(self):
        self.total_vision_params = calculate_right_model_weights(
            self.vision.parameters(),
            is_loaded_in_4bit=self.config.trainer.quantize_vision,
        )
        print(f"Total number of vision parameters: {self.total_vision_params}")

        self.total_language_params = calculate_right_model_weights(
            self.language.parameters(),
            is_loaded_in_4bit=self.config.trainer.quantize_language,
        )
        print(f"Total number of language parameters: {self.total_language_params}")

        self.total_trainable_params = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )
        print(f"Total number of trainable parameters: {self.total_trainable_params}")

    def _load_vision_model(self, vision_config):
        vision_models = dict(clip=CLIPVision)

        assert (
            vision_config.vision.name in vision_models
        ), f"vision model can only be one of {vision_models.keys()}"

        return vision_models[vision_config.vision.name](vision_config)

    def _load_llm(self, osmcap_config: OSMCAP_Config = None):
        llms = dict(gpt=GPT2Model)

        assert osmcap_config.llm.name in llms, f"llm can only be one of {llms.keys()}"

        if osmcap_config is None:
            ValueError("osmcap_config was not set")

        return llms[osmcap_config.llm.name](osmcap_config)

    def get_ground_truth(self, annotations, contents):
        """
        This is helper function to get the ground truth for the model on the inspec_page.py file
        """

        with torch.no_grad():

            contents_exp = contents.unsqueeze(2)
            annotations_exp = annotations.unsqueeze(1)

            # compute cosine similarity is the ground truth
            similarity_scores = self.cos_sim(annotations_exp, contents_exp)
            gt_content_max_similarity, _ = torch.max(similarity_scores, dim=2)

            # take all the objects
            sorted_indices = torch.argsort(
                gt_content_max_similarity, dim=1, descending=True
            )

            return sorted_indices

    def infer_most_similar_osm_words(self, contents, images):

        with torch.no_grad():

            visual_features = self.vision(images)
            cls_tokens = visual_features[:, 0, :]

            projected_vision = self.linear(cls_tokens).unsqueeze(1)
            pred_visual_osm_similarity = self.cos_sim(
                contents.to(self.config.device), projected_vision
            )

            sorted_indices = torch.argsort(
                pred_visual_osm_similarity, dim=1, descending=True
            )

            return sorted_indices

    def get_most_similar_osm_embeddings(self, contents, images, top_k):
        with torch.no_grad():
            visual_features = self.vision(images)
            cls_tokens = visual_features[:, 0, :]

            projected_vision = self.linear(cls_tokens).unsqueeze(1)

            contents = contents.to(self.config.device)

            pred_visual_osm_similarity = self.cos_sim(contents, projected_vision)

            if top_k > pred_visual_osm_similarity.shape[-1]:
                top_k = pred_visual_osm_similarity.shape[-1]

            _, pred_indices = torch.topk(pred_visual_osm_similarity, dim=1, k=top_k)

            pred_indices_expanded = pred_indices.unsqueeze(-1).expand(-1, -1, 384)
            selected_embeddings = torch.gather(
                contents, dim=1, index=pred_indices_expanded
            )

            attention_mask = (selected_embeddings.abs().sum(dim=-1) > 0).int()

            return selected_embeddings, attention_mask

    def forward(self, annotations, contents, images) -> torch.Tensor:

        visual_features = self.vision(images)
        cls_tokens = visual_features[:, 0, :].to(self.device)

        contents_exp = contents.unsqueeze(2).to(self.config.device)
        annotations_exp = annotations.unsqueeze(1).to(self.config.device)

        # compute cosine similarity is the ground truth
        similarity_scores = self.cos_sim(annotations_exp, contents_exp)
        gt_content_max_similarity, _ = torch.max(similarity_scores, dim=2)

        # cosine similarity between cls token and the vision is the prediction
        projected_vision = self.linear(cls_tokens).unsqueeze(1)
        pred_visual_osm_similarity = self.cos_sim(
            contents.to(self.config.device), projected_vision
        )

        return gt_content_max_similarity, pred_visual_osm_similarity
