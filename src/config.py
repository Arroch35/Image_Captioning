from dataclasses import dataclass
import torch
import os

SETID_MAT = "/export/fhome/vlia04/Challenge3/Image_Captioning/data/setid.mat"
IMAGELABELS_MAT = "/export/fhome/vlia04/Challenge3/Image_Captioning/data/imagelabels.mat"
CAT_TO_NAME_JSON = "/export/fhome/vlia04/Challenge3/Image_Captioning/src/cat_to_name.json"
DATA_DIR = "/export/fhome/vlia04/Challenge3/Image_Captioning/data/"
IMAGE_DIR = os.path.join(DATA_DIR, "jpg")
OUTPUT_DIR = "/export/fhome/vlia04/Challenge3/Image_Captioning/models/"
os.makedirs(OUTPUT_DIR, exist_ok=True)



@dataclass
class TrainConfig:
    model_name: str = "ViT-B/32"
    image_size: int = 224

    # training
    batch_size: int = 64
    epochs: int = 20
    lr: float = 1e-4
    weight_decay: float = 0.2
    warmup_ratio: float = 0.05
    max_grad_norm: float = 1.0
    amp: bool = True

    # output
    out_dir: str = "./checkpoints_flowers102"
    save_every: int = 1

    # runtime
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
