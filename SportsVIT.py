import torch.nn as nn
from torchvision.models import vit_b_16
from torchvision.models.vision_transformer import EncoderBlock

ALL_ATTN_WEIGHTS = []   # global variable, accessed during visualization

def log_attn_forward(layer: EncoderBlock, x):
    """New forward method for EncoderBlock to get attn weights"""
    # Applying normalization as per the original implementation
    new_x = layer.ln_1(x)
    # collect both the output and weights, add weights to the global variable
    attn_output, attn_weights = layer.self_attention(new_x, new_x, new_x, need_weights=True, average_attn_weights=False)
    ALL_ATTN_WEIGHTS.append(attn_weights.squeeze().cpu().detach())
    # Following the original architecture (not changing anything)
    x = x + layer.dropout(attn_output)
    new_x = layer.ln_2(x)
    x = x + layer.mlp(new_x)
    return x

class SportsClassificationViT(nn.Module):
    """Implementation of Custom ViT for Sports Classification."""
    def __init__(self, num_classes: int = 100):
        """Constructor"""
        super(SportsClassificationViT, self).__init__()

        # Model Attributes
        self.model = vit_b_16(pretrained=True)  # loading the pre-trained model

        # replacing the last layer (head) with classification layer
        in_features = self.model.heads.head.in_features
        self.model.heads.head = nn.Linear(in_features, num_classes) # head now has num_classes outputs

        # Replacing the forward method of the EncoderBlock to get the attention weights
        for layer in self.model.encoder.layers:
            # layer refers to 'encoder_layer_{i}'
            layer.forward = log_attn_forward.__get__(layer) # binding this new forward method to the layer instance

    def forward(self, x):
        """uses the original model's forward method"""
        return self.model(x)
    