import argparse
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import f1_score
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from SportsVIT import SportsClassificationViT
from transforms import val_transforms

def test(model, test_loader, device):
    """Tests the model on the test set"""
    model.eval()    # freezing the model
    all_preds, all_labels, outputs = [], [], []
    for images, labels in tqdm(test_loader, desc="Testing SportsViT", leave=False):
        images, labels = images.to(device), labels.to(device)
        output = model(images)  # predicting the output
        _, preds = torch.max(output, 1) # getting class with max prob
        all_preds.extend(preds.to("cpu"))
        all_labels.extend(labels.to("cpu"))
        outputs.extend(output.detach().to("cpu").numpy())
    return all_preds, all_labels, outputs

if __name__ == "__main__":
    # Setting up arguments for testing
    parser = argparse.ArgumentParser(description="Testing SportsViT Model")
    parser.add_argument("--src_dir", type=str, default="sports_dataset")
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--save_dir", type=str, default="models")
    parser.add_argument("--exp_id", type=str, required=True)
    args = parser.parse_args()

    # initializing the model
    sports_model = SportsClassificationViT()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sports_model.to(device)

    # Setting up the test data loader
    test_loader = DataLoader(ImageFolder(Path(args.src_dir) / "test", transform = val_transforms), batch_size = 1, shuffle = False)

    # Loading in the best model
    save_path = Path(args.save_dir) / f"{args.exp_id}.pth"
    load_result = sports_model.load_state_dict(torch.load(save_path), strict=True)
    all_preds, all_labels, outputs = test(sports_model, test_loader, device)

    # save outputs to a CSV file
    pd_data = pd.DataFrame(outputs, columns=[f"class_{i}" for i in range(outputs[0].shape[0])])
    pd_data.to_csv(f"{args.log_dir}/{args.exp_id}_outputs.csv", index=False)
    print(f1_score(all_labels, all_preds, average='macro'))
