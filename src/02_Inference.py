
import os
import torch
import clip
from PIL import Image
import scipy.io as sio
from tqdm import tqdm

# ------------------------------------------------------
# CONFIG
# ------------------------------------------------------
DATA_DIR = "data"
IMAGE_DIR = os.path.join(DATA_DIR, "jpg")
BATCH_SIZE = 32

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# ------------------------------------------------------
# LOAD CLIP
# ------------------------------------------------------
model, preprocess = clip.load("ViT-B/32", device=device)
model.eval()

# ------------------------------------------------------
# FLOWER CLASS NAMES (102)
# ------------------------------------------------------
FLOWER_CLASSES = [
    "pink primrose", "hard-leaved pocket orchid", "canterbury bells",
    "sweet pea", "english marigold", "tiger lily", "moon orchid",
    "bird of paradise", "monkshood", "globe thistle",
    "snapdragon", "colt's foot", "king protea", "spear thistle",
    "yellow iris", "globe-flower", "purple coneflower", "peruvian lily",
    "balloon flower", "giant white arum lily", "fire lily", "pincushion flower",
    "fritillary", "red ginger", "grape hyacinth", "corn poppy",
    "prince of wales feathers", "stemless gentian", "artichoke",
    "sweet william", "carnation", "garden phlox", "love in the mist",
    "mexican aster", "alpine sea holly", "ruby-lipped cattleya",
    "cape flower", "great masterwort", "siam tulip", "lenten rose",
    "barbeton daisy", "daffodil", "sword lily", "poinsettia",
    "bolero deep blue", "wallflower", "marigold", "buttercup",
    "oxeye daisy", "common dandelion", "petunia", "wild pansy",
    "primula", "sunflower", "pelargonium", "bishop of llandaff",
    "gaura", "geranium", "orange dahlia", "pink-yellow dahlia",
    "cautleya spicata", "japanese anemone", "black-eyed susan",
    "silverbush", "californian poppy", "osteospermum",
    "spring crocus", "bearded iris", "windflower", "tree poppy",
    "gazania", "azalea", "water lily", "rose", "thorn apple",
    "morning glory", "passion flower", "lotus lotus",
    "toad lily", "anthurium", "frangipani", "clematis",
    "hibiscus", "columbine", "desert rose", "tree mallow",
    "magnolia", "cyclamen", "watercress", "canna lily",
    "hippeastrum", "bee balm", "ball moss", "foxglove",
    "bougainvillea", "camellia", "mallow", "mexican petunia",
    "bromelia", "blanket flower", "trumpet creeper",
    "blackberry lily"
]

# ------------------------------------------------------
# LOAD LABELS & SPLITS
# ------------------------------------------------------
labels = sio.loadmat(os.path.join(DATA_DIR, "imagelabels.mat"))["labels"].squeeze() - 1
setid = sio.loadmat(os.path.join(DATA_DIR, "setid.mat"))

splits = {
    "train": setid["trnid"].squeeze(),
    "val": setid["valid"].squeeze(),
    "test": setid["tstid"].squeeze(),
}

# ------------------------------------------------------
# PREPARE TEXT FEATURES (ONCE)
# ------------------------------------------------------
prompts = [f"a photo of a {name}" for name in FLOWER_CLASSES]
text_tokens = clip.tokenize(prompts).to(device)

with torch.no_grad():
    text_features = model.encode_text(text_tokens)
    text_features /= text_features.norm(dim=-1, keepdim=True)

logit_scale = model.logit_scale.exp()

# ------------------------------------------------------
# EVALUATION FUNCTION
# ------------------------------------------------------
def evaluate_split(image_ids):
    correct_top1 = 0
    correct_top5 = 0
    total = 0

    for i in tqdm(range(0, len(image_ids), BATCH_SIZE)):
        batch_ids = image_ids[i:i + BATCH_SIZE]

        images = []
        targets = []

        for img_id in batch_ids:
            img_path = os.path.join(IMAGE_DIR, f"image_{img_id:05d}.jpg")
            images.append(preprocess(Image.open(img_path)))
            targets.append(labels[img_id - 1])

        images = torch.stack(images).to(device)
        targets = torch.tensor(targets).to(device)

        with torch.no_grad():
            image_features = model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            logits = logit_scale * image_features @ text_features.T

            top1 = logits.argmax(dim=-1)
            top5 = logits.topk(5, dim=-1).indices

        correct_top1 += (top1 == targets).sum().item()
        correct_top5 += sum(t in top5[i] for i, t in enumerate(targets))

        total += targets.size(0)

    return correct_top1 / total, correct_top5 / total

# ------------------------------------------------------
# RUN EVALUATION
# ------------------------------------------------------
for split_name, image_ids in splits.items():
    top1, top5 = evaluate_split(image_ids)
    print(f"\n{split_name.upper()} SET")
    print(f"Top-1 Accuracy: {top1:.4f}")
    print(f"Top-5 Accuracy: {top5:.4f}")