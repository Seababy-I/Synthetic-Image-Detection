import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_model():
    model = models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)

    checkpoint = torch.load("model.pth", map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state"])

    model.eval()
    return model

# Must match exactly what was used during training
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

def predict(model, image: Image.Image):
    """
    Returns:
        label      : "fake" or "real"
        confidence : float between 0.0 and 1.0
        scores     : dict of class probabilities
    """
    img = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(img)
        probs  = F.softmax(output, dim=1)        # convert logits → probabilities
        confidence, pred_idx = torch.max(probs, dim=1)

    label = "fake" if pred_idx.item() == 0 else "real"
    scores = {
        "fake": round(probs[0][0].item(), 4),
        "real": round(probs[0][1].item(), 4)
    }
    return label, round(confidence.item(), 4), scores