import torch
import torch.nn as nn
from PIL import Image
import os

class PositionalEncoding(nn.Module):
    def __init__(self, d_model):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model

    def forward(self, position):
        angle_rates = 1 / torch.pow(10000, (2 * torch.arange(self.d_model).float() / float(self.d_model)))
        positions = position * angle_rates
        positions = positions.view(-1, self.d_model // 4)

        pos_encoding = torch.cat([torch.sin(positions), torch.cos(positions)], dim=-1)

        return pos_encoding.permute(1, 0).unsqueeze(0)

# Create an instance of the PositionalEncoding class
d_model = 32 * 4  # You can choose the desired dimensionality
positional_encoding = PositionalEncoding(d_model)

# Define the range of positions you want to iterate over
start_position = 0
end_position = 10
step_size = .5

# Create a directory to save PNG images
save_dir = "./pos"
os.makedirs(save_dir, exist_ok=True)

# Iterate over positions and save features as PNG
for position_value in torch.arange(start_position, end_position, step_size):
    position = torch.tensor([position_value])
    output = positional_encoding(position)
    features = output.reshape(16, 16).numpy()

    # Normalize features to the range [0, 255]
    normalized_features = ((features - features.min()) / (features.max() - features.min()) * 255).astype('uint8')

    # Create an image from the features
    image = Image.fromarray(normalized_features)

    # Save the image
    image.save(os.path.join(save_dir, f"position_{position_value:.2f}.png"))

print("Images saved in", save_dir)
