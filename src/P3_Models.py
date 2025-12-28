import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class EncoderWithProjection(nn.Module):
    def __init__(self, encoder, out_dim):
        super().__init__()
        self.encoder = encoder
        self.in_dim = encoder.output_dim
        self.proj = nn.Linear(self.in_dim, out_dim)

    def forward(self, *args, **kwargs):
        features = self.encoder(*args, **kwargs)
        return self.proj(features)

class ResNetWrapper(nn.Module):
    def __init__(self, resnet):
        super().__init__()
        self.backbone = resnet
        self.output_dim = 2048

    def forward(self, x):
        return self.backbone(x)
    
class BertWrapper(nn.Module):
    def __init__(self, bert):
        super().__init__()
        self.bert = bert
        self.output_dim = bert.config.hidden_size  # 768

    def forward(self, input_ids, attention_mask=None):
        return self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        ).last_hidden_state[:, 0]  # CLS

class ModularCLIP(nn.Module):
    def __init__(
        self,
        image_encoder,
        text_encoder,
        embed_dim=512,
        init_temperature=0.07
    ):
        super().__init__()
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder

        self.logit_scale = nn.Parameter(
            torch.ones([]) * math.log(1 / init_temperature)
        )

    def encode_image(self, images):
        feats = self.image_encoder(images)
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
    