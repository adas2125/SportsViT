import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import torch
from SportsVIT import SportsClassificationViT

if __name__ == "__main__":
    # Setting up the arguments
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

    # Extracting the positional embeddings before fine-tuning
    pos_embedding_before_finetune = sports_model.model.state_dict()['encoder.pos_embedding'].cpu().detach().numpy()
    pos_embedding_patches_before_finetune = pos_embedding_before_finetune[:, 1:].squeeze(0)
    pos_embedding_cls_before_finetune = pos_embedding_before_finetune[:, 0].squeeze(0)   # Class Token at beginning

    # Loading in the model with specified experiment ID
    save_path = Path(args.save_dir) / f"{args.exp_id}.pth"
    load_result = sports_model.load_state_dict(torch.load(save_path), strict=True)

    # Extracting the positional embeddings after fine-tuning
    pos_embedding_after_finetune = sports_model.model.state_dict()['encoder.pos_embedding'].cpu().detach().numpy()
    pos_embedding_patches_after_finetune = pos_embedding_after_finetune[:, 1:].squeeze(0)
    pos_embedding_cls_after_finetune = pos_embedding_after_finetune[:, 0].squeeze(0)    # Class Token at beginning

    # Reshaping positional embeddings for plotting
    pos_embedding_cls_2d_before_finetune = pos_embedding_cls_before_finetune.reshape(16, 48) 
    pos_embedding_cls_2d_after_finetune = pos_embedding_cls_after_finetune.reshape(16, 48) 

    # Plotting the positional embeddings
    fig, ax = plt.subplots(2, 2, figsize=(10, 6))
    ax[0, 0].imshow(pos_embedding_patches_before_finetune, cmap='viridis')
    ax[0, 0].set_title("Positional Embeddings (Patches) Before Training")
    ax[0, 0].set_ylabel("Patch Index")
    ax[0, 0].set_xlabel("Embedding Element Index")
    ax[0, 1].imshow(pos_embedding_patches_after_finetune, cmap='viridis')
    ax[0, 1].set_title("Positional Embeddings (Patches) After Training")
    ax[0, 1].set_ylabel("Patch Index")
    ax[0, 1].set_xlabel("Embedding Element Index")
    ax[1, 0].imshow(pos_embedding_cls_2d_before_finetune, cmap='viridis')
    ax[1, 0].set_title("Positional Embeddings (Class Token) Before Training")
    ax[1, 1].imshow(pos_embedding_cls_2d_after_finetune, cmap='viridis')
    ax[1, 1].set_title("Positional Embeddings (Class Token) After Training")
    fig.suptitle("Visualizing Positional Embeddings Before and After Fine-Tuning")
    plt.tight_layout()
    plt.savefig("pos_embeds.png")
    plt.close()
