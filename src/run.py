import math 
import hydra
import wandb
import torch
import lightning as L
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import WandbLogger
from dataset import SupervisedDataset
from models.mlp import MLP
from models.resnet import ResNet18, ResNet34
from models.dilated_resnet import drn_a_50
from lightning_modules import SupervisedTask, PretrainingTask
from models.ecg_models import (
    ECGtoLabel,
    ECGtoLabelviaDistrbution, 
    ECGandPriorLVEFtoLabel, 
    ECGandMeanPriorECGtoLabel,
    ECGandSequentialPriorECGtoLabel,
)
from models.ssl import PCLR
# from pytorch_lightning.plugins import DDPPlugin
import warnings, ipdb
warnings.simplefilter('ignore')
torch.set_default_dtype(torch.float64)
torch.set_float32_matmul_precision('medium')

def interpolate(cfg, pyd): 
    cfg.utils.output_dir = cfg.utils.output_dir.replace('Dropbox (Partners HealthCare)','dropbox')
    # cfg.model.encoder.num_channels = len(cfg.dataset.config.ecg.channels)
    callback_names = {'lightning.pytorch.callbacks.ModelCheckpoint', 'lightning.pytorch.callbacks.early_stopping.EarlyStopping'}
    if isinstance(pyd[0][cfg.dataset.config.label], bool): 
        cfg.model.config.classification = True
        cfg.model.objective._target_ = 'torch.nn.BCEWithLogitsLoss'
        for i in range(len(cfg.callbacks)): 
            if cfg.callbacks[i]['_target_'] in callback_names: 
                cfg.callbacks[i]['monitor'] = 'tune/AUROC'
                cfg.callbacks[i]['mode'] = 'max'
    else: 
        cfg.model.config.classification = False
        cfg.model.objective._target_ = 'torch.nn.MSELoss'
        for i in range(len(cfg.callbacks)): 
            if cfg.callbacks[i]['_target_'] in callback_names: 
                cfg.callbacks[i]['monitor'] = 'tune/loss'
                cfg.callbacks[i]['mode'] = 'min'
    assert (0.0 < cfg.optimizer.end_lr_frac_of_init_lr < 1.0)  
    cfg.optimizer.end_lr = cfg.optimizer.end_lr_frac_of_init_lr * cfg.optimizer.init_lr
    steps_per_epoch = int(math.ceil(len(pyd) / cfg.optimizer.batch_size))
    cfg.optimizer.num_training_steps = steps_per_epoch * cfg.trainer.max_epochs
    cfg.optimizer.num_warmup_steps = int(round(cfg.optimizer.lr_frac_warmup_steps * cfg.optimizer.num_training_steps))
    return cfg

@hydra.main(version_base=None, config_path="config", config_name="config")
def run_experiment(cfg):

    L.seed_everything(cfg.utils.seed)

    train_pyd = hydra.utils.instantiate(cfg.dataset, split='train')    
    cfg = interpolate(cfg, train_pyd)
    train_loader = torch.utils.data.DataLoader(
        dataset = train_pyd,
        batch_size = cfg.optimizer.batch_size,
        num_workers = cfg.optimizer.num_dataloader_workers,
        collate_fn = train_pyd.collate,
        shuffle=True,
        pin_memory=True,
    )
    
    tune_pyd = hydra.utils.instantiate(cfg.dataset, split='tune')
    tune_loader = torch.utils.data.DataLoader(
        dataset = tune_pyd,
        batch_size = cfg.optimizer.batch_size,
        num_workers = cfg.optimizer.num_dataloader_workers,
        collate_fn = tune_pyd.collate,
        shuffle=False,
        pin_memory=True,
    )

    LM = SupervisedTask(cfg)
    # pretrained_encoder = PretrainingTask.load_from_checkpoint('/storage/chandak/Silver/src/outputs/2024-03-20/16-28-20/checkpoints/epoch_076.ckpt').model.ecg_encoder
    # LM.model.ecg_encoder.load_state_dict(pretrained_encoder.state_dict())
    # for p in LM.model.ecg_encoder.parameters(): 
    #     p.requires_grad=False

    logger = hydra.utils.instantiate(cfg.wandb)
    logger.log_hyperparams( OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True) )
    callbacks = [hydra.utils.instantiate(cb) for cb in cfg.callbacks]
    trainer = hydra.utils.instantiate(
            cfg.trainer, 
            callbacks=callbacks, 
            logger=logger,
            strategy='ddp',
            # strategy=DDPPlugin(find_unused_parameters=False),
            # fast_dev_run=5, 
    )
    trainer.fit(LM, train_loader, tune_loader)

    ckpt_path = trainer.checkpoint_callback.best_model_path.replace('Dropbox (Partners HealthCare)','dropbox') if trainer.checkpoint_callback.best_model_path != "" else None 
    logger.log_hyperparams( {'best_model_path': ckpt_path})
    trainer.validate(LM, dataloaders=tune_loader, ckpt_path=ckpt_path)

    test_pyd = hydra.utils.instantiate(cfg.dataset, split='test')
    test_loader = torch.utils.data.DataLoader(
        dataset = test_pyd,
        batch_size = cfg.optimizer.batch_size,
        num_workers = cfg.optimizer.num_dataloader_workers,
        collate_fn = test_pyd.collate,
        shuffle=False,
        pin_memory=False,
    )
    trainer.test(LM, dataloaders=test_loader, ckpt_path=ckpt_path)

    wandb.finish()

if __name__ == "__main__":
    run_experiment()