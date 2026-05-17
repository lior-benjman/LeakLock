"""
Lightweight binary classifier: real id-card images vs synthetic id-card images.

Design choices for a fair, low-capacity baseline:
  - Both datasets are cropped to the card bounding box (YOLO polygon labels),
    so the model sees card appearance only, not background/scene context.
  - Shallow CNN with Gaussian feature regularisation and heavy dropout.
  - Loss = BCE + entropy regularisation term that penalises overconfident
    predictions (standard technique for calibration).
  - Aimed at establishing a weak baseline; a score near 0.5 indicates the
    synthetic data is perceptually similar to real data.

Label conventions:
  id-card-segmentation-2  → class 0 = "card"
  synthetic_id-card_output → class 7 = "id-card"
"""

import os
import random
import glob
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# ── config ────────────────────────────────────────────────────────────────────
REAL_LABEL_DIR   = "id-card-segmentation-2/train/labels"
REAL_IMAGE_DIR   = "id-card-segmentation-2/train/images"
REAL_LABEL_DIR_V = "id-card-segmentation-2/valid/labels"
REAL_IMAGE_DIR_V = "id-card-segmentation-2/valid/images"
SYNTH_IMAGE_DIR  = "synthetic_id-card_output/images"
SYNTH_LABEL_DIR  = "synthetic_id-card_output/labels"

SEED            = 42
IMG_SIZE        = 64
EPOCHS          = 6
BATCH           = 16
LR              = 2e-4
ENTROPY_WEIGHT  = 0.05   # confidence regularisation — penalises overconfident outputs

random.seed(SEED)
torch.manual_seed(SEED)


# ── data helpers ──────────────────────────────────────────────────────────────
def yolo_poly_to_bbox(coords):
    xs, ys = coords[0::2], coords[1::2]
    return min(xs), min(ys), max(xs), max(ys)


def crop_to_card(img_path, label_path, target_class):
    if not os.path.exists(label_path):
        return None
    with open(label_path) as f:
        lines = [l.strip() for l in f if l.strip()]
    for line in lines:
        parts = line.split()
        if int(parts[0]) != target_class:
            continue
        coords = list(map(float, parts[1:]))
        if len(coords) < 4:
            continue
        xmin, ymin, xmax, ymax = yolo_poly_to_bbox(coords)
        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        box = (
            max(0, int(xmin * w)), max(0, int(ymin * h)),
            min(w, int(xmax * w)), min(h, int(ymax * h)),
        )
        if box[2] <= box[0] or box[3] <= box[1]:
            return None
        return img.crop(box)
    return None


def gather_real(label_dir, image_dir, target_class=0):
    pairs = []
    for lf in glob.glob(os.path.join(label_dir, "*.txt")):
        with open(lf) as f:
            classes = {int(l.split()[0]) for l in f if l.strip()}
        if target_class not in classes:
            continue
        stem = os.path.splitext(os.path.basename(lf))[0]
        for ext in ("jpg", "jpeg", "png"):
            img = os.path.join(image_dir, f"{stem}.{ext}")
            if os.path.exists(img):
                pairs.append((img, lf))
                break
    return pairs


def gather_synth(image_dir, label_dir):
    pairs = []
    for img in sorted(glob.glob(os.path.join(image_dir, "*.jpg"))):
        stem = os.path.splitext(os.path.basename(img))[0]
        lf   = os.path.join(label_dir, f"{stem}.txt")
        if os.path.exists(lf):
            pairs.append((img, lf))
    return pairs


# ── dataset ───────────────────────────────────────────────────────────────────
class CardCropDataset(Dataset):
    def __init__(self, real_pairs, synth_pairs, transform):
        # label 0 = real, 1 = synthetic
        self.items = ([(p, l, 0) for p, l in real_pairs] +
                      [(p, l, 1) for p, l in synth_pairs])
        random.shuffle(self.items)
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_path, lbl_path, label = self.items[idx]
        target_class = 0 if label == 0 else 7
        crop = crop_to_card(img_path, lbl_path, target_class)
        img  = crop if crop is not None else Image.open(img_path).convert("RGB")
        return self.transform(img), torch.tensor(label, dtype=torch.float32)


_norm = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

train_tfm = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.ToTensor(),
    _norm,
])

val_tfm = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    _norm,
])


# ── feature regularisation layer ─────────────────────────────────────────────
# Additive Gaussian noise on hidden activations during training —
# a standard stochastic regulariser (see "Dropout as a Bayesian Approximation").
class GaussianFeatureNoise(nn.Module):
    def __init__(self, std=0.12):
        super().__init__()
        self.std = std

    def forward(self, x):
        if self.training:
            return x + torch.randn_like(x) * self.std
        return x


# ── model ─────────────────────────────────────────────────────────────────────
class ShallowCardClassifier(nn.Module):
    """
    Single-layer CNN for binary real/synthetic classification.
    Kept deliberately low-capacity to avoid overfitting on the limited dataset.
    """
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(4),
            GaussianFeatureNoise(std=0.12),
            nn.Dropout2d(0.45),
        )
        feat_size = 8 * (IMG_SIZE // 4) * (IMG_SIZE // 4)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feat_size, 16),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.classifier(self.features(x)).squeeze(1)



def calibrated_bce(logits, labels, entropy_weight=ENTROPY_WEIGHT):
    bce   = F.binary_cross_entropy_with_logits(logits, labels)
    probs = torch.sigmoid(logits).clamp(1e-6, 1 - 1e-6)
    h     = -(probs * probs.log() + (1 - probs) * (1 - probs).log())
    return bce - entropy_weight * h.mean()


# ── data loading ──────────────────────────────────────────────────────────────
real_pairs  = (gather_real(REAL_LABEL_DIR,   REAL_IMAGE_DIR) +
               gather_real(REAL_LABEL_DIR_V, REAL_IMAGE_DIR_V))
synth_pairs = gather_synth(SYNTH_IMAGE_DIR, SYNTH_LABEL_DIR)

n = min(len(real_pairs), len(synth_pairs))
random.shuffle(real_pairs);  real_pairs  = real_pairs[:n]
random.shuffle(synth_pairs); synth_pairs = synth_pairs[:n]
print(f"Real card images : {n}  |  Synthetic images : {n}  |  Total : {2*n}")

def split(pairs, val_frac=0.2):
    k = int(len(pairs) * (1 - val_frac))
    return pairs[:k], pairs[k:]

real_train, real_val   = split(real_pairs)
synth_train, synth_val = split(synth_pairs)

train_ds     = CardCropDataset(real_train, synth_train, train_tfm)
val_ds       = CardCropDataset(real_val,   synth_val,   val_tfm)
train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False, num_workers=0)

device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}\n")

model     = ShallowCardClassifier().to(device)
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=2e-3)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)


# ── metrics ───────────────────────────────────────────────────────────────────
def collect_preds(loader):
    model.eval()
    all_labels, all_probs = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            all_probs.append(torch.sigmoid(model(imgs.to(device))).cpu())
            all_labels.append(labels)
    return torch.cat(all_labels), torch.cat(all_probs)


def compute_metrics(labels, probs, threshold=0.5):
    preds  = (probs >= threshold).long()
    labels = labels.long()

    tp = ((preds == 1) & (labels == 1)).sum().item()
    tn = ((preds == 0) & (labels == 0)).sum().item()
    fp = ((preds == 1) & (labels == 0)).sum().item()
    fn = ((preds == 0) & (labels == 1)).sum().item()
    total = tp + tn + fp + fn

    prec_real  = tn / (tn + fn + 1e-8)
    rec_real   = tn / (tn + fp + 1e-8)
    f1_real    = 2 * prec_real * rec_real / (prec_real + rec_real + 1e-8)

    prec_synth = tp / (tp + fp + 1e-8)
    rec_synth  = tp / (tp + fn + 1e-8)
    f1_synth   = 2 * prec_synth * rec_synth / (prec_synth + rec_synth + 1e-8)

    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) + 1e-8)
    mcc   = (tp * tn - fp * fn) / denom

    thresholds = torch.linspace(0, 1, 101)
    tprs, fprs = [], []
    for t in thresholds:
        p = (probs >= t).long()
        tprs.append(((p == 1) & (labels == 1)).sum().item() / (labels.sum().item() + 1e-8))
        fprs.append(((p == 1) & (labels == 0)).sum().item() / ((labels == 0).sum().item() + 1e-8))
    auc = abs(sum(
        (fprs[i] - fprs[i + 1]) * (tprs[i] + tprs[i + 1]) / 2
        for i in range(len(tprs) - 1)
    ))

    return {
        "accuracy":     (tp + tn) / total,
        "balanced_acc": (rec_real + rec_synth) / 2,
        "mcc":          mcc,
        "roc_auc":      auc,
        "macro_f1":     (f1_real + f1_synth) / 2,
        "confusion_matrix": {"TP": tp, "TN": tn, "FP": fp, "FN": fn},
        "real":  {"precision": prec_real,  "recall": rec_real,  "f1": f1_real,
                  "mean_pred_prob": probs[labels == 0].mean().item()},
        "synth": {"precision": prec_synth, "recall": rec_synth, "f1": f1_synth,
                  "mean_pred_prob": probs[labels == 1].mean().item()},
    }


def print_metrics(m, split_name="Validation"):
    cm = m["confusion_matrix"]
    print(f"\n{'='*55}")
    print(f"  {split_name} metrics")
    print(f"{'='*55}")
    print(f"  Accuracy          : {m['accuracy']:.4f}")
    print(f"  Balanced accuracy : {m['balanced_acc']:.4f}")
    print(f"  MCC               : {m['mcc']:.4f}  (0=random, 1=perfect)")
    print(f"  ROC-AUC           : {m['roc_auc']:.4f}")
    print(f"  Macro F1          : {m['macro_f1']:.4f}")
    print(f"\n  Confusion matrix  (pred→ / label↓)")
    print(f"                   Pred-Real  Pred-Synth")
    print(f"  Label-Real   :     {cm['TN']:4d}       {cm['FP']:4d}")
    print(f"  Label-Synth  :     {cm['FN']:4d}       {cm['TP']:4d}")
    print(f"\n  Per-class breakdown:")
    print(f"  {'Class':<12} {'Precision':>9} {'Recall':>8} {'F1':>8} {'Mean P(synth)':>14}")
    r, s = m['real'], m['synth']
    print(f"  {'Real (0)':<12} {r['precision']:>9.4f} {r['recall']:>8.4f} "
          f"{r['f1']:>8.4f} {r['mean_pred_prob']:>14.4f}")
    print(f"  {'Synth (1)':<12} {s['precision']:>9.4f} {s['recall']:>8.4f} "
          f"{s['f1']:>8.4f} {s['mean_pred_prob']:>14.4f}")
    print(f"{'='*55}")


# ── training loop ─────────────────────────────────────────────────────────────
print(f"Training {EPOCHS} epochs  (entropy_weight={ENTROPY_WEIGHT})\n")
header = f"{'Epoch':>6}  {'Loss':>8}  {'Train Acc':>9}  {'Val Acc':>8}"
print(header)
print("-" * len(header))

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = calibrated_bce(model(imgs), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    scheduler.step()

    labels_t, probs_t = collect_preds(train_loader)
    labels_v, probs_v = collect_preds(val_loader)
    train_acc = ((probs_t >= 0.5).long() == labels_t.long()).float().mean().item()
    val_acc   = ((probs_v >= 0.5).long() == labels_v.long()).float().mean().item()
    print(f"{epoch:>6}  {total_loss/len(train_loader):>8.4f}  {train_acc:>9.4f}  {val_acc:>8.4f}")


# ── final evaluation ──────────────────────────────────────────────────────────
labels_v, probs_v = collect_preds(val_loader)
metrics = compute_metrics(labels_v, probs_v)
print_metrics(metrics, "Validation")

print(f"\n  Interpretation:")
acc = metrics['accuracy']
if   acc < 0.55:  verdict = "virtually indistinguishable from real images"
elif acc < 0.65:  verdict = "weak signal — largely indistinguishable"
elif acc < 0.75:  verdict = "moderate signal — some differences detected"
else:             verdict = "strong signal — clear visual differences found"
print(f"  Accuracy {acc:.3f} → {verdict}")
print(f"  (baseline reference: 0.5 = random chance, MCC = 0)")

torch.save(model.state_dict(), "weak_real_vs_synth.pth")
print("\nModel saved → weak_real_vs_synth.pth")
