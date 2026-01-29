import os
import json
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import clip
from typing import Tuple, Dict
import scipy.io as sio
from torch.utils.data import Dataset
from torchvision import transforms as T

# CLIP normalization constants
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD  = (0.26862954, 0.26130258, 0.27577711)


# Loss (symmetric InfoNCE)
def clip_loss(logits_per_image, logits_per_text):
    batch_size = logits_per_image.size(0)
    labels = torch.arange(batch_size, device=logits_per_image.device)
    loss_i = F.cross_entropy(logits_per_image, labels)
    loss_t = F.cross_entropy(logits_per_text, labels)
    return (loss_i + loss_t) / 2


def load_flowers_splits(setid_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mat = sio.loadmat(setid_path)
    trn = mat["trnid"].squeeze()
    val = mat["valid"].squeeze()
    tst = mat["tstid"].squeeze()
    return trn, val, tst


def load_flowers_labels(imagelabels_path: str) -> np.ndarray:
    mat = sio.loadmat(imagelabels_path)
    return mat["labels"].squeeze()


def build_reversed_split_ids(setid_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
      train = original valid + original test
      test  = original train
    """
    trn, val, tst = load_flowers_splits(setid_path)
    train_ids = np.concatenate([val, tst], axis=0)
    test_ids = trn
    return train_ids, test_ids


def load_cat_to_name(cat_to_name_json: str) -> Dict[str, str]:
    with open(cat_to_name_json, "r", encoding="utf-8") as f:
        return json.load(f)


def build_train_transform(image_size: int = 224) -> T.Compose:
    """
    augmentation: random square crop from resized images (RandomResizedCrop)
    and CLIP normalization.
    """
    return T.Compose([
        T.Resize(image_size, interpolation=Image.BICUBIC),
        T.CenterCrop(image_size),
        T.Lambda(lambda img: img.convert("RGB")),
        T.ToTensor(),
        T.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
    ])


class Flowers102CLIPDataset(Dataset):
    """
    Flowers102 dataset for CLIP fine-tuning.
    Returns:
      image tensor: [3, H, W]
      token_ids:    [77]
    """
    def __init__(
        self,
        image_ids: np.ndarray,
        labels_1based: np.ndarray,
        images_dir: str,
        cat_to_name: Dict[str, str],
        image_transform,
        prompt_template: str = "a photo of a {}",
    ):
        self.image_ids = image_ids.astype(int).tolist()
        self.labels = labels_1based.astype(int)  # indexed by (img_id - 1)
        self.images_dir = images_dir
        self.cat_to_name = cat_to_name
        self.image_transform = image_transform
        self.prompt_template = prompt_template

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx: int):
        img_id = self.image_ids[idx]  # 1-based image index
        class_id = int(self.labels[img_id - 1])  # 1..102

        image_path = os.path.join(self.images_dir, f"image_{img_id:05d}.jpg")
        image = Image.open(image_path).convert("RGB")
        image = self.image_transform(image)

        class_name = self.cat_to_name[str(class_id)]
        text = self.prompt_template.format(class_name)

        token_ids = clip.tokenize([text], truncate=True).squeeze(0)  # [77]
        return image, token_ids
