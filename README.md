# Intro to Comp. Neuroscience Programming Assignment
## Author: Amit Das

In this repository, I use a ViT to categorize images into their appropriate sports category. The dataset I use is publicly available and can be accessed [here](https://www.kaggle.com/datasets/gpiosenka/sports-classification).

### Directories
The `logs` directory consists of training logs and model outputs on the test set. The `models` directory consists of checkpoints for different models. The best performing model is `sports_classification_lr_1e-05_epochs_5_batch_16_augmentation_False.pth`. The `attention_maps_heads` consists of subplots of attention maps from different heads of the last attention layer. The `attention_maps_layers` directory consists of subplots of attention maps across layers averaged over the heads. 

### Scripts
The `train_vit.py` trains the ViT model. The `test_vit.py` script evaluates the trained ViT on the test set. The `SportsVIT.py` script implements the architecture of the ViT. The `utils` directory consists of additional scripts to plot loss curves from the log files, visualize positional embeddings from the model, and generate the attention maps stored in the `attention_maps_heads` and `attention_maps_layers` directories.

To reproduce these results, one needs to download the dataset from Kaggle and then run the scripts.