
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from dataset import Num16Dataset
from models import CNN_A
from train import train, evaluate

def prepare_data():
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
    ])
    dataset = Num16Dataset(images_dir='images/', labels_file='labels.csv', transform=transform)
    total = len(dataset)
    train_len = int(0.7 * total)
    val_len = int(0.15 * total)
    test_len = total - train_len - val_len
    return random_split(dataset, [train_len, val_len, test_len])

if __name__ == "__main__":
    train_set, val_set, test_set = prepare_data()
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN_A()  # or CNN_B()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train(model, train_loader, val_loader, criterion, optimizer, device, epochs=10)
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"Final Test Accuracy: {test_acc * 100:.2f}%")
