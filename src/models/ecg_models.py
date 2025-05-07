import torch 
from models.mlp import MLP
from models.resnet import ResNet18, ResNet34
from lightning_modules import SupervisedTask

        
class ECGandPriorLVEFtoLabel(torch.nn.Module):

    def __init__(self, config, objective): 
        super().__init__()
        self.config = config 
        self.ecg_encoder = ResNet18(
            embed_only=True, 
            num_channels=len(self.config.ecg.channels), 
            dropout_prob=self.config.dropout_prob
        )         
        self.ecg_decoder = MLP(layers=[256, 1024, 512], dropout_prob=self.config.dropout_prob)
        self.prior_lvef_decoder = MLP(layers=[len(self.config.lvef.prior), 512], dropout_prob=self.config.dropout_prob)
        self.mlp = MLP(layers=[512, 128, 1], dropout_prob=self.config.dropout_prob)
        self.objective = objective 

    def forward(self, batch):
        ecg_repr = self.ecg_encoder(batch['ecg'])
        ecg_embed = self.ecg_decoder(ecg_repr)
        lvef_embed = self.prior_lvef_decoder(torch.stack([batch[c]for c in self.config.lvef.prior], dim=1))
        pred = self.mlp(ecg_embed + lvef_embed)
        pred = pred.squeeze()
        loss = self.objective(pred, batch['label'])
        out = {
            'loss':loss,
            'pred':pred,
            'label':batch['label'],
        }
        return out 