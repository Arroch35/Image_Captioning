from torchvision import transforms
from PIL import Image
import torch
import torch.nn.functional as F
from P3_Models import *
from torchvision import models
from transformers import BertModel, BertTokenizer
import clip

def build_clip_preprocess(image_size=224):
    return transforms.Compose([
        transforms.Resize(image_size, interpolation=Image.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711),
        ),
    ])

# ------------------------------------------------------
# CLIP LOSS
# ------------------------------------------------------
def clip_loss(logits_per_image, logits_per_text):
    batch_size = logits_per_image.size(0)
    labels = torch.arange(batch_size, device=logits_per_image.device)
    loss_i = F.cross_entropy(logits_per_image, labels)
    loss_t = F.cross_entropy(logits_per_text, labels)
    return (loss_i + loss_t) / 2


def build_custom_resnet50_bert_clip(device, embed_dim=512, init_temperature=0.07):
    # ------------------------------------------------------
    # Image encoder: ResNet50
    # ------------------------------------------------------
    resnet = models.resnet50(pretrained=True)
    resnet.fc = nn.Identity()  # Remove final classification layer

    image_encoder = EncoderWithProjection(
        encoder=ResNetWrapper(resnet),
        out_dim=embed_dim
    )

    # ------------------------------------------------------
    # Text encoder: BERT
    # ------------------------------------------------------
    bert = BertModel.from_pretrained("bert-base-uncased")
    text_encoder = EncoderWithProjection(
        encoder=BertWrapper(bert),
        out_dim=embed_dim
    )

    # ------------------------------------------------------
    # Build Modular CLIP model
    # ------------------------------------------------------
    model = ModularCLIP(
        image_encoder=image_encoder,
        text_encoder=text_encoder,
        embed_dim=embed_dim,
        init_temperature=init_temperature
    )

    return model.to(device)

