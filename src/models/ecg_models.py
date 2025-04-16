import torch 
from models.mlp import MLP
from models.resnet import ResNet18, ResNet34
from lightning_modules import SupervisedTask

class ECGtoLabel(torch.nn.Module):

    def __init__(self, config, objective): 
        super().__init__()
        self.config = config 
        self.ecg_encoder = ResNet18(embed_only=True, num_channels=12, dropout_prob=self.config.dropout_prob)         
        self.ecg_decoder = MLP(layers=[256, 1024, 512], dropout_prob=self.config.dropout_prob)
        self.mlp = MLP(layers=[512, 128, 1], dropout_prob=self.config.dropout_prob)
        self.objective = objective 

    def forward(self, batch):
        ecg_repr = self.ecg_encoder(batch['ecg'])
        ecg_embed = self.ecg_decoder(ecg_repr)
        pred = self.mlp(ecg_embed)
        pred = pred.squeeze()
        loss = self.objective(pred, batch['label'])
        out = {
            'loss':loss,
            'pred':pred,
            'label':batch['label'],
        }
        return out 

class ECGtoLabelviaDistrbution(torch.nn.Module):

    def __init__(self, config, objective): 
        super().__init__()
        self.config = config 
        self.future_mean_model = SupervisedTask.load_from_checkpoint(self.config.future_mean_model_pth).model
        self.future_std_model = SupervisedTask.load_from_checkpoint(self.config.future_std_model_pth).model
        self.objective = objective

    def forward(self, batch): 
        mean = self.future_mean_model(batch)['pred'].squeeze(dim=-1).to(torch.float64)
        std = self.future_std_model(batch)['pred'].squeeze(dim=-1).to(torch.float64)
        m = torch.distributions.normal.Normal(loc = mean, scale = std)
        if self.config.label == "future_lvef_any_hfref": 
            prob = m.cdf(torch.tensor([40.0]).type_as(mean))
        elif self.config.label == "future_lvef_any_hfmref": 
            prob = m.cdf(torch.tensor([50.0]).type_as(mean)) - m.cdf(torch.tensor([40.0]).type_as(mean))
        elif self.config.label == "future_lvef_any_hfpef": 
            prob = 1 - m.cdf(torch.tensor([50.0]).type_as(mean))
        else:
            raise ValueError(f"invalid label for prediction via distribution {self.config.label}")
        loss = self.objective(prob, batch[self.config.label])
        out = {
            'loss':loss, 
            'pred': prob, 
            'label':batch[self.config.label], 
        }
        return out 
        
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

class ECGandMeanPriorECGtoLabel(torch.nn.Module):

    def __init__(self, config, objective): 
        super().__init__()
        self.config = config 
        self.ecg_encoder = ResNet18(embed_only=True, num_channels=12, dropout_prob=self.config.dropout_prob)         
        self.ecg_decoder = MLP(layers=[256, 1024, 512], dropout_prob=self.config.dropout_prob)
        self.mlp = MLP(layers=[512, 128, 1], dropout_prob=self.config.dropout_prob)
        self.objective = objective 

    def encode_sequence(self, ecgs, batch_size, channels, samples, repr_dim): 
        data = ecgs.view(-1, channels, samples) # batch size * max prior ecg len, channels, samples 
        repr = self.ecg_encoder(data).view(batch_size, -1, repr_dim) # batch size, max prior ecg len, repr dim
        return repr
    
    def safe_masked_mean(self, x, mask): 
        x_masked = torch.where(mask, x, torch.tensor(float('nan')))
        x_mean = torch.nanmean(x_masked, dim=1)
        if x_masked.nelement() == 0: # all data points have 0 prior ecgs 
            x_mean = torch.zeros_like(x_mean)
        x_mean[x_mean.isnan()] = 0 # some datapoints have 0 prior ecgs 
        return x_mean


    def forward(self, batch):
        ecg_repr = self.ecg_encoder(batch['ecg'])

        prior_ecg_repr = self.encode_sequence(
            ecgs=batch['prior_ecg'],
            batch_size=ecg_repr.size(0),
            channels=batch['ecg'].size(1), 
            samples=batch['ecg'].size(2),  
            repr_dim=ecg_repr.size(1),
        ) 
        prior_ecg_mask = batch['prior_ecg_mask'].bool().unsqueeze(-1)
        prior_ecg_mean = self.safe_masked_mean(prior_ecg_repr, prior_ecg_mask)

        ecg_embed = self.ecg_decoder(ecg_repr + prior_ecg_mean)
        pred = self.mlp(ecg_embed)
        pred = pred.squeeze()
        loss = self.objective(pred, batch[self.config.label])
        out = {
            'loss':loss,
            'pred':pred,
            'label':batch[self.config.label],
        }
        return out 

class ECGandSequentialPriorECGtoLabel(torch.nn.Module):

    def __init__(self, config, objective): 
        super().__init__()
        self.config = config 
        self.ecg_encoder = ResNet18(embed_only=True, num_channels=12, dropout_prob=self.config.dropout_prob)         
        self.ecg_decoder = torch.nn.GRU(input_size=256, hidden_size=256, batch_first=True, dropout=self.config.dropout_prob)
        self.mlp = MLP(layers=[256, 128, 1], dropout_prob=self.config.dropout_prob)
        self.objective = objective 

    def forward(self, batch):
        ecg_repr = self.ecg_encoder(batch['ecg'])
        
        prior_ecg = batch['prior_ecg'].view(-1, batch['ecg'].size(1), batch['ecg'].size(2)) # batch size * max prior ecg len, channels, samples 
        x = self.ecg_encoder(prior_ecg).view(ecg_repr.size(0), -1, ecg_repr.size(1)) # batch size, max prior ecg len, repr dim
        
        mask = batch['prior_ecg_mask'].bool()
        lengths = mask.sum(dim=1).long()
        non_zero_indices = lengths > 0
        non_zero_lengths = lengths[non_zero_indices]
        sorted_lengths, sorted_indices = non_zero_lengths.sort(descending=True) 
        _, inverted_indices = sorted_indices.sort()
        x_sorted = x[non_zero_indices][sorted_indices] 
        x_packed = torch.nn.utils.rnn.pack_padded_sequence(x_sorted, sorted_lengths.cpu(), batch_first=True)
        packed_output, hidden = self.ecg_decoder(x_packed)
        x_seq = torch.zeros_like(ecg_repr)
        x_seq[non_zero_indices] = hidden[-1][inverted_indices]

        pred = self.mlp(ecg_repr + x_seq)
        pred = pred.squeeze()
        loss = self.objective(pred, batch[self.config.label])
        out = {
            'loss':loss,
            'pred':pred,
            'label':batch[self.config.label],
        }
        return out 