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
from dataset import PretrainingDataset
from models.mlp import MLP
from models.resnet import ResNet18, ResNet34
from models.dilated_resnet import drn_a_50
from lightning_modules import PretrainingTask
from models.ssl import PCLR
# from pytorch_lightning.plugins import DDPPlugin
import warnings
import ipdb
warnings.simplefilter('ignore')
torch.set_default_dtype(torch.float64)

def interpolate(cfg, pyd): 
    cfg.utils.output_dir = cfg.utils.output_dir.replace('Dropbox (Partners HealthCare)','dropbox')
    assert (0.0 < cfg.optimizer.end_lr_frac_of_init_lr < 1.0)  
    cfg.optimizer.end_lr = cfg.optimizer.end_lr_frac_of_init_lr * cfg.optimizer.init_lr
    steps_per_epoch = int(math.ceil(len(pyd) / cfg.optimizer.batch_size))
    cfg.optimizer.num_training_steps = steps_per_epoch * cfg.trainer.max_epochs
    cfg.optimizer.num_warmup_steps = int(round(cfg.optimizer.lr_frac_warmup_steps * cfg.optimizer.num_training_steps))
    return cfg

@hydra.main(version_base=None, config_path="config", config_name="pretrain")
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

    LM = PretrainingTask(cfg)
    logger = hydra.utils.instantiate(cfg.wandb)
    logger.log_hyperparams( OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True) )
    callbacks = [hydra.utils.instantiate(cb) for cb in cfg.callbacks]
    trainer = hydra.utils.instantiate(
            cfg.trainer, 
            callbacks=callbacks, 
            logger=logger,
            strategy='ddp',
            # fast_dev_run=5, 
    )
    trainer.fit(LM, train_loader)

    ckpt_path = trainer.checkpoint_callback.best_model_path.replace('Dropbox (Partners HealthCare)','dropbox') if trainer.checkpoint_callback.best_model_path != "" else None 
    logger.log_hyperparams( {'best_model_path': ckpt_path})

if __name__ == "__main__":
    run_experiment()