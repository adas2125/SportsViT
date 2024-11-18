from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# Path to the log files
log_dir = Path('logs')
log_files = list(log_dir.glob('*loss.csv'))

# Mapping log files to model configurations
log_file_mapping = {
    'sports_classification_lr_0.001_epochs_5_batch_16_augmentation_True_loss.csv': 'Model 2',
    'sports_classification_lr_0.1_epochs_5_batch_16_augmentation_True_loss.csv': 'Model 3',
    'sports_classification_lr_1e-05_epochs_5_batch_16_augmentation_False_loss.csv': 'Model 1',
    'sports_classification_lr_1e-05_epochs_5_batch_16_augmentation_True_loss.csv': 'Model 4',
    'sports_classification_lr_1e-05_epochs_5_batch_64_augmentation_True_loss.csv': 'Model 5'
}

# plotting the losses for the models
fig, ax = plt.subplots(nrows=1, ncols=5, figsize=(40, 5))
for i, log_file in enumerate(log_files):
    # Read the log file
    log = pd.read_csv(log_file)
    train_loss = log['train_loss']
    val_loss = log['val_loss']  
    # Plot the losses
    ax[i].plot(train_loss, label='train_loss')
    ax[i].plot(val_loss, label='val_loss')
    ax[i].set_title(log_file_mapping[log_file.name])
    ax[i].set_xlabel('Epochs')
    ax[i].set_ylabel('Loss')
    ax[i].legend()
fig.savefig('losses.png')
plt.close()
