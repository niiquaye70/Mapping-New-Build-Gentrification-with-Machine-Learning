import torch
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm import tqdm

def evaluate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    y_true, y_pred, y_scores = [], [], []

    with torch.no_grad():
        for img1, img2, labels in tqdm(val_loader, desc="Evaluating", leave=False):
            img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)

            outputs = model(img1, img2).squeeze()
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            y_true.extend(labels.cpu().numpy())
            y_pred.extend((outputs.detach().cpu().numpy() > 0.5).astype(int))
            y_scores.extend(outputs.detach().cpu().numpy())

    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_scores)
    avg_loss = val_loss / len(val_loader)

    return avg_loss, acc, auc
