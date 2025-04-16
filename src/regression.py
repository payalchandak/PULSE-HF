# import 
import numpy as np
from scipy.stats import chi2
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
import polars as pl
import pandas as pd
import seaborn as sns
import ipdb
import lightning as L
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, roc_curve
from omegaconf import OmegaConf, DictConfig
import hydra
import wandb
import os, h5py, hdf5plugin
import pywt, random, wfdb
from itertools import combinations
from sklearn.linear_model  import LogisticRegression
from sklearn.model_selection import GridSearchCV
import torch 
import numpy as np, pandas as pd, polars as pl
from dataset import SupervisedDataset
from lightning_modules import SupervisedTask
from models.ecg_models import *
from run import interpolate
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=Warning)
from sklearn.exceptions import ConvergenceWarning
warnings.simplefilter("ignore", category=ConvergenceWarning)
import os
os.environ["PYTHONWARNINGS"] = "ignore"
import logging
logging.getLogger("sklearn").setLevel(logging.ERROR)

# definitions 
label = 'future_1_365_any_below_40'
features = {
    'demographics':['sex', 'age'], 
    'meds':['num_meds', 'angio', 'betablocker', 'mra', 'diuretic'], 
    'comorbidities':['mean_annual_hospitalizations', 'diabetes_mellitus', 'hypertension', 'atheroscler', 'chronic_obstructive_pulmonary_disease', 'atrial_fibrillation'], 
    'lvef':['days_since_diagnosis', 'initial_lvef', 'preceding_lvef', 'prior_lvef_support', 'prior_lvef_mean', 'prior_lvef_min', 'prior_lvef_max', 'prior_lvef_std', 'prior_lvef_range', 'prior_lvef_any_hfref', 'prior_lvef_all_hfref', 'prior_lvef_any_hfpef', 'prior_lvef_all_hfpef', 'prior_lvef_any_hfmref', 'prior_lvef_all_hfmref'], 
    'ecg':['atrialrate', 'paxis', 'poffset', 'ponset', 'printerval', 'qoffset', 'qonset', 'qrscount', 'qrsduration', 'qtcorrected', 'qtinterval', 'toffset', 'ventricularrate'],
}

# functions 

def load_data(split):
    WANDB_RUN='payalchandak/SILVER/nuzrc47q'
    cfg = OmegaConf.create(wandb.Api().run(WANDB_RUN).config)
    L.seed_everything(cfg.utils.seed)
    cfg.dataset._target_ = 'dataset.SupervisedDataset'
    train_pyd = hydra.utils.instantiate(cfg.dataset, split='train')
    cfg = interpolate(cfg, train_pyd)
    cfg.optimizer.batch_size = 2048
    cfg.dataset.config.label = 'future_1_365_any_below_40'
    if split == 'mimic': 
        cfg.dataset.config.datadir = '/storage/shared/mimic/'
        cfg.dataset.config.ecg.storedir = '/storage/shared/mimic/raw/ecg/'
    pyd = hydra.utils.instantiate(cfg.dataset, split=split)
    pyd.data = pyd.data.reset_index(drop=1)
    return pyd.data

def add_ecg_features(row):
    date = row['ecg_date']
    if not isinstance(date, pd.Timestamp): date = pd.to_datetime(date)
    pth = os.path.join('/storage/shared/ecg/mgh', str(row['empi'])+'.hd5')
    f = h5py.File(row['ecg_path'], 'r')['ecg'][str(date).replace(' ','T')]
    for feat in features['ecg']: 
        for suffix in ['_md','_pc']:
            feat_find = feat+suffix 
            if feat_find in f.keys():
                row[feat] = float(f[feat_find][()])
                break
    return row 

# data
df_train = load_data('train').apply(add_ecg_features, axis=1).dropna(subset=features['ecg'], ignore_index=True)
df_test_mgh = load_data('test').apply(add_ecg_features, axis=1).dropna(subset=features['ecg'], ignore_index=True)
df_test_bwh = load_data('external').apply(add_ecg_features, axis=1).dropna(subset=features['ecg'], ignore_index=True)

# regression
feature_categories = ['ecg', 'lvef', 'demographics', 'meds', 'comorbidities']
combos = [list(combinations(feature_categories, r)) for r in range(1, len(feature_categories) + 1)]
combos = [*reversed([item for sublist in combos for item in sublist])]
combos = [x for x in combos if 'ecg' in x]

print("\n| " + " | ".join(feature_categories) + " | MGH All | BWH All | MGH Worse | BWH Worse |")
print("|" + "-" * (len(feature_categories) * 10 + 20) + "|")

for comb in combos:
    feat_included = {category: '1' if category in comb else '0' for category in feature_categories}
    feat = [features[k] for k in comb]
    feat = [item for sublist in feat for item in sublist]
    
    train = df_train.get(feat+[label]).fillna(0).drop_duplicates().reset_index(drop=True).astype(float)
    X_train = train.get(feat).values
    y_train = train.get(label).values
    search = GridSearchCV(
        estimator=LogisticRegression(),
        param_grid={"C": np.logspace(-3, 3, 7), "penalty": ["l1", "l2"], "solver": ["saga"], "max_iter": [1000]},
        scoring='roc_auc',
        n_jobs=20,
    )
    search.fit(X_train, y_train)
    clf = search.best_estimator_
    
    auc_all = {}
    for name, df in [('MGH', df_test_mgh), ('BWH', df_test_bwh)]:
        test = df.get(feat+[label]).fillna(0).drop_duplicates().reset_index(drop=True).astype(float)
        X_test = test.get(feat).values
        y_test = test.get(label).values
        prob = clf.predict_proba(X_test)[:, 1]
        auc_all[name] = round(roc_auc_score(y_test, prob), 3)

    auc_worsen = {}
    for name, df in [('MGH', df_test_mgh), ('BWH', df_test_bwh)]:
        test = df.query('preceding_lvef_above_40==True').get(feat+[label]).fillna(0).drop_duplicates().reset_index(drop=True).astype(float)
        X_test = test.get(feat).values
        y_test = test.get(label).values
        prob = clf.predict_proba(X_test)[:, 1]
        auc_worsen[name] = round(roc_auc_score(y_test, prob), 3)

    
    print("| " + " | ".join(feat_included[cat] for cat in feature_categories) + f" | {auc_all['MGH']} | {auc_all['BWH']} | {auc_worsen['MGH']} | {auc_worsen['BWH']} |")
