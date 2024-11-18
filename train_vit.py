import argparse
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from transforms import train_transforms, val_transforms
from SportsVIT import SportsClassificationViT

def train_vit_model(model, num_epochs, lr, train_loader, val_loader, device, save_loc):
    """Training the model"""
    criterion = nn.CrossEntropyLoss()   # for classification task
    optimizer = optim.Adam(model.parameters(), lr=lr)   # Adam optimizer
    train_losses, val_losses = [], []   # storing the losses, so they can be logged
    best_val_loss = float("inf")
    for epoch_num in range(num_epochs): # training for num_epochs
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, epoch_num, device)
        val_loss = validate(model, val_loader, criterion, epoch_num, device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(f"At epoch {epoch_num}, train loss: {train_loss}, val loss: {val_loss}")  # LOGGING
        if val_loss < best_val_loss:
            print(f"Saving the new best model!")
            torch.save(model.state_dict(), save_loc)
            best_val_loss = val_loss
    return train_losses, val_losses

def train_one_epoch(model, train_loader, criterion, optimizer, epoch_num, device):
    """Training the model for a given epoch"""
    model.train()       # allowing gradients to be computed and weights to be adjusted
    train_loss = 0.0    # loss accummulator
    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch_num} in training set", leave=False):
        images, labels = images.to(device), labels.to(device)
        pred = model(images)    # predicting with model
        loss = criterion(pred, labels)  # computing the loss
        train_loss += loss.detach().cpu().item() / len(train_loader)    
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return train_loss
    
def validate(model, val_loader, criterion, epoch_num, device):
    """Validates the model for a given epoch"""
    model.eval()    # freezing the model
    val_loss = 0.0  # loss accumulator
    for images, labels in tqdm(val_loader, desc=f"Epoch {epoch_num} in validation set", leave=False):
        images, labels = images.to(device), labels.to(device)
        pred = model(images)    # predicting with model
        loss = criterion(pred, labels)  # computing the loss
        val_loss += loss.detach().cpu().item() / len(val_loader)
    return val_loss

if __name__ == "__main__":
    # Setting up arguments for training script
    parser = argparse.ArgumentParser(description="Training SportsViT Model")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--src_dir", type=str, default="sports_dataset")
    parser.add_argument("--log_dir", type=str, default="logs_trial")
    parser.add_argument("--use_augmentation", action="store_true")
    parser.add_argument("--save_dir", type=str, default="models_trial")
    args = parser.parse_args()

    # ensure the log and save_dir directories exist
    Path(args.log_dir).mkdir(exist_ok=True, parents=True)
    Path(args.save_dir).mkdir(exist_ok=True, parents=True)

    exp_id = f"sports_classification_lr_{args.lr}_epochs_{args.num_epochs}_batch_{args.batch_size}_augmentation_{args.use_augmentation}"
    print(f"Experiment ID: {exp_id}")

    # initializing the model
    sports_model = SportsClassificationViT()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Whether to use augmentation or not
    if args.use_augmentation:
        train_transforms = train_transforms
    else:
        train_transforms = val_transforms   # regular transforms for training

    # Setting up the data loaders
    train_loader = DataLoader(ImageFolder(Path(args.src_dir) / "train", transform = train_transforms), batch_size = args.batch_size, shuffle = True)
    val_loader = DataLoader(ImageFolder(Path(args.src_dir) / "valid", transform = val_transforms), batch_size = args.batch_size, shuffle = False)
    test_loader = DataLoader(ImageFolder(Path(args.src_dir) / "test", transform = val_transforms), batch_size = 1, shuffle = False)

    # Start training the model!
    sports_model.to(device)
    save_loc = Path(args.save_dir) / f"{exp_id}.pth"
    train_losses, val_losses = train_vit_model(sports_model, args.num_epochs, args.lr, train_loader, val_loader, device, save_loc)

    # Logging the losses
    pd_loss = pd.DataFrame({"train_loss": train_losses, "val_loss": val_losses})
    pd_loss.to_csv(f"{args.log_dir}/{exp_id}_loss.csv", index=False)    # saving the log file
