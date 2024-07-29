import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttentionLayer(nn.Module):
    def __init__(self):
        super(SelfAttentionLayer, self).__init__()

    def scaled_dot_product_attention(self, Q, K, V):
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k).float())
        attention_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, V)
        return output, attention_weights

    def forward(self, volume):
        Q = volume
        K = volume
        V = volume

        attention_output, attention_weights = self.scaled_dot_product_attention(Q, K, V)

        return attention_output, attention_weights

# Example usage
attention_layer = SelfAttentionLayer()

# Assuming your input tensor is 'input_volume'
input_volume = torch.randn(1, 64, 64)

# Apply the self-attention layer
output_volume, attention_weights = attention_layer(input_volume)



