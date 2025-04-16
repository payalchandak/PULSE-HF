import torch 
from models.mlp import MLP
from models.resnet import ResNet18, ResNet34
from lightly.loss import NTXentLoss

class PCLR(torch.nn.Module):

    def __init__(self, config, objective): 
        super().__init__()
        self.config = config 
        self.ecg_encoder = ResNet18(embed_only=True, num_channels=12, dropout_prob=self.config.dropout_prob)      
        self.projection_head = MLP(layers=[256, 1024], dropout_prob=self.config.dropout_prob)
        self.objective = objective 

    def forward(self, batch):
        h_i, h_j = self.ecg_encoder(batch['x_i']), self.ecg_encoder(batch['x_j'])
        z_i, z_j = self.projection_head(h_i), self.projection_head(h_j)
        loss = self.objective(z_i, z_j)
        out = {
            'loss':loss,
            'h_i':h_i,
            'h_j':h_j,
            'z_i':z_i,
            'z_j':z_j,
        }
        return out 