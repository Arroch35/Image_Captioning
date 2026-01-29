import os
from PIL import Image
from torch.utils.data import Dataset
from flowers_names import FLOWER_CLASSES


DATA_DIR = "../data"
IMAGE_DIR = os.path.join(DATA_DIR, "jpg")

class FlowersCLIPDataset(Dataset):
    def __init__(self, image_ids, preprocess, tokenizer, labels):
        self.image_ids = image_ids
        self.preprocess = preprocess
        self.tokenizer = tokenizer
        self.labels = labels

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        label = self.labels[img_id - 1]

        image_path = os.path.join(IMAGE_DIR, f"image_{img_id:05d}.jpg")
        image = self.preprocess(Image.open(image_path).convert("RGB"))

        text = f"a photo of a {FLOWER_CLASSES[label]}"
        tokens = self.tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=77 #32
        )

        return image, tokens["input_ids"].squeeze(0), tokens["attention_mask"].squeeze(0)