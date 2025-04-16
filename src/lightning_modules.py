import hydra
import torch
import numpy as np 
import lightning as L 
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryAUROC,
    BinaryAveragePrecision,
)
from torchmetrics.regression import (
    ExplainedVariance,
    MeanAbsoluteError,
    MeanSquaredError,
    R2Score,
)
from transformers import get_polynomial_decay_schedule_with_warmup

class SupervisedTask(L.LightningModule): 

    def __init__(self, cfg): 
        super().__init__()
        self.cfg = cfg
        self.model = hydra.utils.instantiate(cfg.model)
        self.build_metrics()
        self.save_hyperparameters(logger=False)
        self.cache = {}
        self.cohort_logger_prefix = '' # if specified, it should end with _ 

    def configure_optimizers(self):
        """Currently this module uses the AdamW optimizer, with configurable weight_decay, with a learning rate
        warming up from 0 on a per-step manner to the configurable `self.cfg.optimizer.init_lr`, then
        undergoes polynomial decay as specified via `self.cfg.optimizer`.
        """
        opt = torch.optim.AdamW(
            params = self.parameters(),
            lr = self.cfg.optimizer.init_lr,
            betas = tuple(self.cfg.optimizer.betas),
            weight_decay = self.cfg.optimizer.weight_decay
        )
        scheduler = get_polynomial_decay_schedule_with_warmup(
            optimizer=opt,
            num_warmup_steps=self.cfg.optimizer.num_warmup_steps,
            num_training_steps=self.cfg.optimizer.num_training_steps,
            power=self.cfg.optimizer.lr_decay_power,
            lr_end=self.cfg.optimizer.end_lr,
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }
    
    def build_metrics(self): 
        if self.cfg.model.config.classification: 
            self.metrics = torch.nn.ModuleDict({
                    "AUROC": BinaryAUROC(),
                    "accuracy": BinaryAccuracy(),
                    "AUPRC": BinaryAveragePrecision(),
                })
        else: # regression
            self.metrics = torch.nn.ModuleDict({
                    "explained_variance": ExplainedVariance(),
                    "MAE": MeanAbsoluteError(),
                    "MSE": MeanSquaredError(),
                    "r2_score": R2Score(),
                })

    def _get_cached_data(self, split): 
        loss = np.mean(self.cache[split+'_losses'])
        label = torch.cat(self.cache[split+'_labels'], dim=0)
        preds = torch.cat(self.cache[split+'_preds'], dim=0)
        preds = preds.squeeze()
        if self.cfg.model.config.classification: 
            label = label.long()
        return loss, preds, label

    def on_epoch_end(self, split): 
        loss, pred, label = self._get_cached_data(split)
        self.log(f"{split}/{self.cohort_logger_prefix}loss", loss, sync_dist=True, on_step=False, on_epoch=True)
        for metric_name, metric_fn in self.metrics.items(): 
            try: 
                self.log(f"{split}/{self.cohort_logger_prefix}{metric_name}", metric_fn(pred, label), sync_dist=True, on_step=False, on_epoch=True)
            except (ValueError, IndexError) as e:
                print(f"Failed to compute {metric_name} ")
                                                                                 
    def on_train_epoch_end(self):
        self.on_epoch_end('train')
    
    def on_validation_epoch_end(self):
        self.on_epoch_end('tune')

    def on_test_epoch_end(self):
        self.on_epoch_end('test')
        self.bootstrap('test')

    def bootstrap(self, split, confidence=0.95, num_samples=1000):
        loss, pred, label = self._get_cached_data(split)
        lower_idx, upper_idx = int(num_samples*(1-confidence)/2), int(num_samples*(confidence)/2)
        n = pred.shape[0]
        for metric_name, metric_fn in self.metrics.items():  
            samples = []
            for _ in range(num_samples): 
                idx = torch.multinomial(torch.ones(n), n, replacement=True)
                try:
                    samples.append( metric_fn(pred[idx], label[idx]).item() )
                except: 
                    print(f'Error in bootstrapping {metric_name}')
                    print(f'idx len: {len(idx)}; preds shape {pred[idx].shape}; label shape {label[idx].shape}')
                    print(f'metric function result {metric_fn(pred[idx], label[idx])}')
            assert len(samples) == num_samples, f'failed to get {num_samples} samples for bootstrapping'
            samples = sorted(samples)
            self.log(f"{split}/{self.cohort_logger_prefix}{metric_name}_mean", np.mean(samples))
            self.log(f"{split}/{self.cohort_logger_prefix}{metric_name}_lower", samples[lower_idx])
            self.log(f"{split}/{self.cohort_logger_prefix}{metric_name}_upper", samples[upper_idx])

    def on_epoch_start(self, split): 
        self.cache[split+'_losses'] = []
        self.cache[split+'_preds'] = []
        self.cache[split+'_labels'] = []
        
    def on_train_epoch_start(self):
        self.on_epoch_start('train')
    
    def on_validation_epoch_start(self):
        self.on_epoch_start('tune')

    def on_test_epoch_start(self):
        self.on_epoch_start('test')

    def step(self, batch, split): 
        out = self.model(batch)
        if split in ('train','tune','test'): 
            self.cache[split+'_losses'].append(out['loss'].detach().cpu())
            self.cache[split+'_preds'].append(out['pred'].detach())
            self.cache[split+'_labels'].append(out['label'].detach())
            return out["loss"]
        elif split=='predict': 
            return out["pred"], out["label"]
        else: 
            raise ValueError(f'invalid split {split}')

    def training_step(self, batch, batch_idx):
        return self.step(batch, 'train')

    def validation_step(self, batch, batch_idx):
        return self.step(batch, 'tune')

    def test_step(self, batch, batch_idx):
        return self.step(batch, 'test')
        
    def predict_step(self, batch, batch_idx):
        return self.step(batch, 'predict')

class PretrainingTask(L.LightningModule): 

    def __init__(self, cfg): 
        super().__init__()
        self.cfg = cfg
        self.model = hydra.utils.instantiate(cfg.model)
        self.save_hyperparameters(logger=False)
        self.cohort_logger_prefix = '' # if specified, it should end with _ 

    def configure_optimizers(self):
        """Currently this module uses the AdamW optimizer, with configurable weight_decay, with a learning rate
        warming up from 0 on a per-step manner to the configurable `self.cfg.optimizer.init_lr`, then
        undergoes polynomial decay as specified via `self.cfg.optimizer`.
        """
        opt = torch.optim.AdamW(
            params = self.parameters(),
            lr = self.cfg.optimizer.init_lr,
            betas = tuple(self.cfg.optimizer.betas),
            weight_decay = self.cfg.optimizer.weight_decay
        )
        scheduler = get_polynomial_decay_schedule_with_warmup(
            optimizer=opt,
            num_warmup_steps=self.cfg.optimizer.num_warmup_steps,
            num_training_steps=self.cfg.optimizer.num_training_steps,
            power=self.cfg.optimizer.lr_decay_power,
            lr_end=self.cfg.optimizer.end_lr,
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }

    def step(self, batch, split): 
        out = self.model(batch)
        if split in ('train','tune','test'): 
            self.log(f"{split}/loss", out["loss"], sync_dist=True)
            return out["loss"]
        elif split=='predict': 
            return out
        else: 
            raise ValueError(f'invalid split {split}')

    def training_step(self, batch, batch_idx):
        return self.step(batch, 'train')
        
    def predict_step(self, batch, batch_idx):
        return self.step(batch, 'predict')