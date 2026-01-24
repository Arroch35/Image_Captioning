import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchvision import models, transforms
from transformers import (
    BertModel,
    GPT2Model
)

class EncoderWithProjection(nn.Module):
    def __init__(self, encoder, out_dim, encoder_type="bert"):
        super().__init__()
        self.encoder = encoder
        self.encoder_type = encoder_type
        self.in_dim = encoder.output_dim
        self.proj = nn.Linear(self.in_dim, out_dim)

    def forward(self, input_ids=None, attention_mask=None, images=None):
        if self.encoder_type == "bert":
            outputs = self.encoder(input_ids, attention_mask)
        elif self.encoder_type == "gpt":
            outputs = self.encoder(input_ids, attention_mask)
        else:  # images
            outputs = self.encoder(images)
        return self.proj(outputs)

class ResNetWrapper(nn.Module):
    def __init__(self, resnet):
        super().__init__()
        self.backbone = models.resnet50(pretrained=True)
        self.backbone.fc = nn.Identity()
        self.output_dim = 2048

    def forward(self, x):
        return self.backbone(x)
    
class SwinTinyWrapper(nn.Module):
    def __init__(self, swin):
        super().__init__()
        # Load pretrained Swin Tiny
        self.backbone = models.swin_t(weights="IMAGENET1K_V1")
        self.backbone.head = nn.Identity()  # remove classifier
        self.output_dim = 768  # Swin Tiny final embedding dimension

    def forward(self, x):
        return self.backbone(x)
    
class BERTWrapper(nn.Module):
    def __init__(self, name="bert-base-uncased"):
        super().__init__()
        self.model = BertModel.from_pretrained(name)
        self.output_dim = self.model.config.hidden_size

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]

    
class GPTWrapper(nn.Module):
    def __init__(self, name="gpt2"):
        super().__init__()
        self.model = GPT2Model.from_pretrained(name)
        self.output_dim = self.model.config.hidden_size

    def forward(self, input_ids, attention_mask=None):
        last_hidden = self.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        return last_hidden.mean(dim=1)  # mean pooling

class ModularCLIP(nn.Module):
    def __init__(
        self,
        image_encoder,
        text_encoder,
        embed_dim=512,
        init_temperature=0.07,
        learned_temperature=None
    ):
        super().__init__()
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder

        if learned_temperature is None:
            self.logit_scale = nn.Parameter(
                torch.ones([]) * math.log(1 / init_temperature)
            )
        else:
            self.logit_scale = nn.Parameter(
                torch.ones([]) * math.log(1 / learned_temperature)
            )

    def encode_image(self, images):
        feats = self.image_encoder(images=images)
        return feats / feats.norm(dim=-1, keepdim=True)

    def encode_text(self, input_ids, attention_mask):
        feats = self.text_encoder(input_ids, attention_mask)
        return feats / feats.norm(dim=-1, keepdim=True)

    def forward(self, images, input_ids, attention_mask):
        image_features = self.encode_image(images)
        text_features = self.encode_text(input_ids, attention_mask)

        # cosine similarity as logits   
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text
    