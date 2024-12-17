import torch
from torchvision import transforms as T
from utils.general_utils import random_choice


class ImageTransforms:
    def __init__(self):
        pass

    @staticmethod
    def gauss_noise_tensor(img, std, mean):
        out = img + torch.randn(img.shape) * std + mean
        # Minmax scale it again
        out = (out - out.min()) / (out.max() - out.min())

        return out

    @staticmethod
    def build_transforms(transformations, gaussian_noise, random_erasing):
        """
        Transformations to augment the data, depends on the flags activated and their probability set
        on the config file. Gaussian noise or random earsing can be applied.
        """

        if transformations is None:
            return T.Compose([T.ToTensor()])

        transform_list = []

        if gaussian_noise.enabled:
            apply_gauss_noise = random_choice(
                gaussian_noise.probability,
            )

        transform_list.append(T.ToTensor())

        if random_erasing.enabled:
            transform_list.append(T.RandomErasing(p=random_erasing.probability))

        if apply_gauss_noise:
            apply_gauss_noise = lambda x: ImageTransforms.gauss_noise_tensor(
                x, gaussian_noise.std, gaussian_noise.mean
            )
            transform_list.append(apply_gauss_noise)

        return T.Compose(transform_list)
