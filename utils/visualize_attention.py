import argparse
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from SportsVIT import SportsClassificationViT, ALL_ATTN_WEIGHTS
from transforms import val_transforms

def test(model, test_loader, device):
    """Tests the model on the test set"""
    model.eval()
    all_preds, all_labels, outputs = [], [], []
    for images, labels in tqdm(test_loader, desc="Testing SportsViT", leave=False):
        images, labels = images.to(device), labels.to(device)
        output = model(images)
        _, preds = torch.max(output, 1)
        all_preds.extend(preds.to("cpu"))
        all_labels.extend(labels.to("cpu"))
        outputs.extend(output.detach().to("cpu").numpy())
    return all_preds, all_labels, outputs

if __name__ == "__main__":
    # Setting up arguments for visualization
    parser = argparse.ArgumentParser(description="Visualizing Attention Weights!")
    parser.add_argument("--src_dir", type=str, default="sports_dataset")
    parser.add_argument("--save_dir", type=str, default="models")
    parser.add_argument("--num_heads", type=int, default=12)
    parser.add_argument("--num_layers", type=int, default=12)
    parser.add_argument("--exp_id", type=str, required=True)
    parser.add_argument("--visualize_mode", type=str, required=True, choices=["heads", "layers"])
    args = parser.parse_args()

    # initializing the model
    sports_model = SportsClassificationViT()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sports_model.to(device)

    # Setting up the test data loader
    test_dataset = ImageFolder(Path(args.src_dir) / "test", transform = val_transforms)
    test_loader = DataLoader(test_dataset, batch_size = 1, shuffle = False)

    # Loading in the model specified by the experiment ID
    save_path = Path(args.save_dir) / f"{args.exp_id}.pth"
    sports_model.load_state_dict(torch.load(save_path), strict=True)
    all_preds, all_labels, outputs = test(sports_model, test_loader, device)
    print(f1_score(all_labels, all_preds, average='macro'))
    
    # Save the attention weights
    os.makedirs(f"attention_weights_{args.visualize_mode}", exist_ok=True)
    for i, attn_weights in enumerate(ALL_ATTN_WEIGHTS):
        if (i+1) % args.num_layers == 0:
            starting_layer = i - args.num_layers + 1
            if args.visualize_mode == "layers":
                attn_weights = torch.stack(ALL_ATTN_WEIGHTS[starting_layer:i+1])
                attn_weights = attn_weights.mean(dim=1) # average over the heads
            else:
                attn_weights = ALL_ATTN_WEIGHTS[i]  # take last layer weights
            print(f"[LOGGING] shape of weights: {ALL_ATTN_WEIGHTS[i].shape}")
            # Save the aggregated attention weights
            np.save(f"attention_weights_{args.visualize_mode}/attention_weights_{int((i+1)/12) - 1}.npy", attn_weights.detach().numpy())
    
    # Save the attention map overlays
    os.makedirs(f"attention_maps_{args.visualize_mode}", exist_ok=True)
    for i, test_image in enumerate(test_dataset):
        # loading image and attention weights
        attn_weights = np.load(f"attention_weights_{args.visualize_mode}/attention_weights_{i}.npy")
        test_image = test_image[0].numpy().transpose((1, 2, 0))
        # Plotting the attention map overlays
        fig, ax = plt.subplots(ncols=4, nrows=3)
        for head in range(args.num_heads):
            # using head and layer does not matter, since same number of heads and layers
            cls_attention = attn_weights[head][0] # cls token
            cls_attention = cls_attention[1:].reshape((14, 14))
            cls_attention = cls_attention / np.max(cls_attention)
            attn_resized = np.array(Image.fromarray(cls_attention).resize(test_image.shape[:2], Image.BILINEAR))
            # Overlay the attention map on the image
            ax[int(head/4), head%4].imshow(test_image, alpha=1.0)
            ax[int(head/4), head%4].imshow(attn_resized, cmap='viridis', alpha=0.8)
            ax[int(head/4), head%4].axis('off')
            ax[int(head/4), head%4].set_title(f"Layer {head}")
        plt.savefig(f"attention_maps_{args.visualize_mode}/overlayed_attention_weights_{i}.png")
        plt.close()
