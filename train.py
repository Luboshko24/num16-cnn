
import torch
from torch.utils.tensorboard import SummaryWriter

def train(model, train_loader, val_loader, criterion, optimizer, device, epochs=10, log_dir="runs/exp1"):
    writer = SummaryWriter(log_dir)
    model.to(device)
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        train_loss = running_loss / len(train_loader)
        train_acc = correct / total
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
        writer.add_scalars("Loss", {"Train": train_loss, "Validation": val_loss}, epoch+1)
        writer.add_scalars("Accuracy", {"Train": train_acc, "Validation": val_acc}, epoch+1)
    writer.close()

def evaluate(model, data_loader, criterion, device):
    model.eval()
    total = 0
    correct = 0
    loss_total = 0.0
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss_total += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return loss_total / len(data_loader), correct / total
