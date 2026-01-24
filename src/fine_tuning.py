import os
import math
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm as tqdm
from utils2 import *
from config import *
import clip
from torch.amp import GradScaler


def cosine_with_warmup(step: int, total_steps: int, warmup_steps: int) -> float:
    if step < warmup_steps:
        return float(step) / max(1, warmup_steps)
    progress = float(step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def main(cfg: TrainConfig):
    os.makedirs(cfg.out_dir, exist_ok=True)
    device = torch.device(cfg.device)

    # Load pretrained CLIP base
    model, _ = clip.load(cfg.model_name, device=device, jit=False)
    model = model.float()
    model.train()

    # unfreeze all parameters
    for p in model.parameters():
        p.requires_grad = True

    train_ids, _test_ids_unused = build_reversed_split_ids(SETID_MAT)
    labels = load_flowers_labels(IMAGELABELS_MAT)
    cat_to_name = load_cat_to_name(CAT_TO_NAME_JSON)

    train_transform = build_train_transform(cfg.image_size)
    train_ds = Flowers102CLIPDataset(
        image_ids=train_ids,
        labels_1based=labels,
        images_dir=IMAGE_DIR,
        cat_to_name=cat_to_name,
        image_transform=train_transform,
        prompt_template="a photo of a {}",
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        betas=(0.9, 0.98),
        eps=1e-6,
        weight_decay=cfg.weight_decay,
    )

    total_steps = cfg.epochs * len(train_loader)
    warmup_steps = int(cfg.warmup_ratio * total_steps)

    global_step = 0
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{cfg.epochs}", leave=True)

        for images, token_ids in pbar:
            images = images.to(device, non_blocking=True)
            token_ids = token_ids.to(device, non_blocking=True)

            # cosine LR schedule (per-step)
            lr_scale = cosine_with_warmup(global_step, total_steps, warmup_steps)
            for pg in optimizer.param_groups:
                pg["lr"] = cfg.lr * lr_scale

            optimizer.zero_grad(set_to_none=True)

            logits_per_image, logits_per_text = model(images, token_ids)
            loss = clip_loss(logits_per_image, logits_per_text)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.step()
            running_loss += loss.item()
            global_step += 1

            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=optimizer.param_groups[0]["lr"])

        avg_loss = running_loss / max(1, len(train_loader))
        print(f"? Epoch {epoch:02d}/{cfg.epochs} done - train loss: {avg_loss:.4f}")

    # ---- SAVE ONCE (FINAL) ----
    final_path = os.path.join(cfg.out_dir, "finetuned_clip_final.pth")

    # Save the full model object (what you asked for)
    torch.save(model.state_dict(), final_path)

    print(f"Saved final model to: {final_path}")


if __name__ == "__main__":
    cfg = TrainConfig()  # IMPORTANT: instantiate it
    main(cfg)
