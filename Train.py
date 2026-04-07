

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from PIL import Image



TRAIN_DIR  = "/kaggle/input/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images/test"
TEST_DIR   = "/kaggle/input/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images/test"

BATCH_SIZE = 64
NUM_EPOCHS = 5
LR         = 1e-4
VAL_SPLIT  = 0.2         
SAVE_PATH  = "Backend/model.pth"

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")



imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std  = [0.229, 0.224, 0.225]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
])


full_dataset = datasets.ImageFolder(root=TRAIN_DIR, transform=transform)

val_size   = int(len(full_dataset) * VAL_SPLIT)
train_size = len(full_dataset) - val_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

test_dataset = datasets.ImageFolder(root=TEST_DIR, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

class_names = full_dataset.classes
print(f"Classes: {class_names}")



model = models.resnet18(pretrained=True)

num_features  = model.fc.in_features        # 512 for ResNet18
model.fc      = nn.Linear(num_features, 2)  # FAKE=0, REAL=1

model = model.to(device)


# ── 4. LOSS & OPTIMIZER ──────────────────────────────────────

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)


# ── 5. HELPERS ───────────────────────────────────────────────

def accuracy(loader, model):
    """Compute accuracy (%) over a DataLoader."""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total   += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total


# ── 6. TRAINING LOOP ─────────────────────────────────────────

print("\n" + "="*55)
print("  Training ResNet18 — CIFAKE Binary Classifier")
print("="*55)

for epoch in range(1, NUM_EPOCHS + 1):

    # ── Train ──
    model.train()
    running_loss    = 0.0
    train_correct   = 0
    train_total     = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss  += loss.item() * images.size(0)
        _, predicted   = torch.max(outputs, 1)
        train_total   += labels.size(0)
        train_correct += (predicted == labels).sum().item()

    epoch_loss     = running_loss / train_total
    train_acc      = 100 * train_correct / train_total
    val_acc        = accuracy(val_loader, model)

    print(f"Epoch [{epoch}/{NUM_EPOCHS}]  "
          f"Loss: {epoch_loss:.4f}  "
          f"Train Acc: {train_acc:.2f}%  "
          f"Val Acc: {val_acc:.2f}%")

print("="*55)


# ── 7. FINAL TEST ACCURACY ───────────────────────────────────

test_acc = accuracy(test_loader, model)
print(f"\nTest Accuracy: {test_acc:.2f}%")


# ── 8. SAVE MODEL ────────────────────────────────────────────

torch.save({
    "model_state": model.state_dict(),
    "class_names": full_dataset.classes,   # e.g. ['FAKE', 'REAL']
}, SAVE_PATH)

print(f"\nModel saved to '{SAVE_PATH}'")


# ── 9. SINGLE-IMAGE PREDICTION ───────────────────────────────

def predict_image(image_path, model, class_names, device):
    """
    Predict whether a single image is FAKE or REAL.

    Args:
        image_path  : path to the image file (str)
        model       : trained PyTorch model
        class_names : list of class names, e.g. ['FAKE', 'REAL']
        device      : 'cuda' or 'cpu'

    Returns:
        label       : predicted class name (str)
        confidence  : confidence score in % (float)
    """
    infer_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    image = Image.open(image_path).convert("RGB")
    image = infer_transform(image).unsqueeze(0).to(device)  # add batch dim

    model.eval()
    with torch.no_grad():
        output     = model(image)
        probs      = torch.softmax(output, dim=1)
        confidence, pred_idx = torch.max(probs, 1)

    label      = class_names[pred_idx.item()]
    confidence = confidence.item() * 100
    print(f"Prediction: {label}  (confidence: {confidence:.2f}%)")
    return label, confidence


# ── Example usage (swap path with any image) ──────────────────
# predict_image("/path/to/your/image.jpg", model, class_names, device)
