import os
import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, roc_auc_score

def train(model, train_loader, val_loader, criterion, optimizer, scheduler, device, logger, epochs=15):
    best_val_accuracy = 0.0

    # Ensure checkpoint directory exists
    ckpt_dir = "outputs/checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        y_true, y_pred, y_scores = [], [], []
        
        for img1, img2, labels in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{epochs}"):
            img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(img1, img2).view(-1)     
            loss = criterion(outputs, labels.float()) 
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            y_true.extend(labels.cpu().numpy())
            y_pred.extend((outputs.detach().cpu().numpy() > 0.5).astype(int))
            y_scores.extend(outputs.detach().cpu().numpy())

        train_acc = accuracy_score(y_true, y_pred)
        train_auc = roc_auc_score(y_true, y_scores)
        logger.info(f"Epoch {epoch+1}: Train Loss={train_loss/len(train_loader):.4f}, "
                    f"Acc={train_acc:.4f}, AUC={train_auc:.4f}")

        # Validation
        from src.training.evaluate import evaluate
        val_loss, val_acc, val_auc = evaluate(model, val_loader, criterion, device)
        logger.info(f"Epoch {epoch+1}: Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}, Val AUC={val_auc:.4f}")
        
        scheduler.step(val_loss)

        # Save best checkpoint
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            ckpt_path = os.path.join(ckpt_dir, "best_siamese_model.pth")
            torch.save(model.state_dict(), ckpt_path)
            logger.info(f"Saved new best model → {ckpt_path} (Val Acc={best_val_accuracy:.4f})")

    return best_val_accuracy
