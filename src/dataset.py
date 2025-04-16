import os, h5py, hdf5plugin
import pywt, random, wfdb
import torch 
import numpy as np, pandas as pd, polars as pl

class SupervisedDataset(torch.utils.data.Dataset): 

    def __init__(self, config, split=None): 
        self.config = config
        self.data = self.read_data(split) 
        self.mimic = split=='mimic'
        pass

    def read_data(self, split) -> pd.DataFrame: 
        df = pl.read_parquet( os.path.join( self.config.datadir, 'data.parquet') )
        df = df.filter(pl.col(self.config.label).is_not_null())
        if split is not None: df = df.filter((pl.col('split')==split)).drop(['split'])
        df = df.to_pandas().reset_index(drop=1)
        return df 

    def get_ecg(self, pth, date, wavelet=False, samples=2500, convert_to_mV=True) -> np.ndarray:
        assert wavelet==False, 'wavelet transform not implemented' 
        if self.mimic:
            pth = os.path.join(self.config.ecg.storedir, pth)
            mimic_ecg = wfdb.rdrecord(pth)
        else:
            try: 
                h5py.File(pth, 'r')['ecg'][str(date).replace(' ','T')]
            except: 
                print(h5py.File(pth, 'r')['ecg'].keys(), date, str(date).replace(' ','T'))
            f = h5py.File(pth, 'r')['ecg'][str(date).replace(' ','T')]
        
        signal = []
        for l in self.config.ecg.channels:
            if self.mimic:
                i = [i for (i,x) in enumerate(mimic_ecg.sig_name) if x==l][0]
                s = mimic_ecg.p_signal[:,i].reshape(-1)
            else:
                s = f[l][()].reshape(-1)
            s = np.interp( np.linspace(0, 1, samples), np.linspace(0, 1, len(s)), s)
            if convert_to_mV: 
                if self.mimic: assert mimic_ecg.units[i]=='mV'
                else: s = s/1000
            signal.append(s)
        if np.isnan(np.vstack(signal)).any():
            print(f'NaN in {pth}')
        signal = np.vstack(signal).tolist()
        return signal

    def __len__(self): 
        return self.data.shape[0] 

    def __getitem__(self, idx) -> dict: 
        item = self.data.loc[idx,:].to_dict()
        for k in item.keys():
            if item[k] is None: item[k] = 0
        item['ecg'] = self.get_ecg(pth=item['ecg_path'], date=item['ecg_date'], wavelet=self.config.ecg.wavelet)
        return item

    def collate(self, batch:dict): 
        collated = {
            'empi': torch.tensor([x['empi'] for x in batch], dtype=torch.int64),
            'label': torch.tensor([x[self.config.label] for x in batch], dtype=torch.float64),
        }
        cols = ['ecg', self.config.label] + self.config.lvef.prior
        for c in cols: 
            data = torch.tensor([x[c] for x in batch], dtype=torch.float64)
            if c in self.config.lvef.prior: # allow missingness 
                data = torch.nan_to_num(data, nan=0)
            assert torch.isfinite(data).all(), f'{c} nans:{data.isnan().sum()} infs:{data.isinf().sum()}'
            collated[c] = data
        return collated

class PretrainingDataset(torch.utils.data.Dataset): 

    def __init__(self, config, split=None, cohort=None): 
        self.config = config
        self.data = self.read_data(split, cohort) 

    def read_data(self, split, cohort) -> pd.DataFrame: 
        file = os.path.join( self.config.datadir, 'ecg_lvef.parquet')
        if self.config.outpatient_lvef_only: 
            file = os.path.join( self.config.datadir, 'ecg_lvef_outpatient.parquet')
        df = pl.read_parquet(file, columns=['id', 'ecg_date', 'switch_patient', 'split'])
        if split is not None: df = df.filter((pl.col('split')==split)).drop(['split'])
        if cohort=='switch': df = df.filter((pl.col('switch_patient'))).drop(['switch_patient'])
        has_more_than_2_ecg = set(df.group_by('id').count().filter(pl.col('count')>1).select('id').to_numpy().reshape(-1))
        df = df.filter(pl.col('id').is_in(has_more_than_2_ecg))
        df = df.to_pandas().reset_index(drop=1)
        return df  

    def get_ecg(self, id, date, wavelet=False, samples=2500, convert_to_mV=True) -> np.ndarray:
        assert wavelet==False, 'wavelet transform not implemented' 
        if not isinstance(date, pd.Timestamp): date = pd.to_datetime(date)
        pth = os.path.join(self.config.ecg.storedir, str(id)+'.hd5')
        f = h5py.File(pth, 'r')['ecg'][str(date).replace(' ','T')]
        signal = []
        for l in self.config.ecg.channels: 
            s = f[l][()].reshape(-1)
            s = np.interp( np.linspace(0, 1, samples), np.linspace(0, 1, len(s)), s)
            if convert_to_mV: s = s/1000
            signal.append(s)
        signal = np.vstack(signal).tolist()
        return signal

    def __len__(self): 
        return self.data.shape[0] 

    def __getitem__(self, idx) -> dict: 
        item = self.data.loc[idx,:].to_dict()
        item['ecg'] = self.get_ecg(id=item['id'], date=item['ecg_date'], wavelet=self.config.ecg.wavelet)
        item['other_ecg_date'] = random.choice(list(set(self.data.query("id==@item['id']").get('ecg_date').values) - set([item['ecg_date']])))
        item['other_ecg'] = self.get_ecg(id=item['id'], date=item['other_ecg_date'], wavelet=self.config.ecg.wavelet)
        return item

    def collate(self, batch:dict): 
        collated = {
            'id': torch.tensor([x['id'] for x in batch], dtype=torch.int64),
            'switch': torch.tensor([x['switch_patient'] for x in batch], dtype=torch.int64),
            'x_i': torch.tensor([x['ecg'] for x in batch], dtype=torch.float64),
            'x_j': torch.tensor([x['other_ecg'] for x in batch], dtype=torch.float64),
        }
        for k in collated.keys(): 
            data = collated[k]
            assert torch.isfinite(data).all(), f'{k} nans:{data.isnan().sum()} infs:{data.isinf().sum()}'
        return collated


class BaselineDataset(): 

    def __init__(self, features=None, target=None): 
        datafile = '/storage/chandak/Silver/data/baseline.parquet'
        if not os.path.exists(datafile): self.build_dataset(datafile)
        self.df = pl.read_parquet(datafile)
        features = [f for f in features if f in self.df.columns]
        print(f'missing features: {[f for f in features if f not in self.df.columns]}')
        self.df = self.df.filter(pl.col(target).is_not_null()).fill_null(-1)
        self.features = features 
        self.target = target 

    def build_dataset(self, datafile): 
        df = pl.read_parquet('/storage/chandak/Silver/data/ecg_lvef_1_365.parquet')
        ecg_md_features = ['atrialrate_md', 'paxis_md', 'poffset_md', 'ponset_md', 'printerval_md', 'qoffset_md', 'qonset_md', 'qrscount_md', 'qrsduration_md', 'qtcorrected_md', 'qtinterval_md', 'raxis_md','taxis_md', 'toffset_md', 'ventricularrate_md']
        ecg_pc_features = ['atrialrate_pc', 'paxis_pc', 'poffset_pc', 'ponset_pc', 'printerval_pc', 'qoffset_pc', 'qonset_pc', 'qrscount_pc', 'qrsduration_pc', 'qtcorrected_pc', 'qtinterval_pc', 'raxis_pc' 'taxis_pc', 'toffset_pc', 'ventricularrate_pc']
        ecg_features = ecg_md_features + ecg_pc_features
        def add_ecg_features(row):
            date = row['ecg_date']
            if not isinstance(date, pd.Timestamp): date = pd.to_datetime(date)
            pth = os.path.join('/storage/shared/ecg/mgh', str(row['id'])+'.hd5')
            f = h5py.File(pth, 'r')['ecg'][str(date).replace(' ','T')]
            for feat in ecg_features: 
                if feat in f.keys(): row[feat] = float(f[feat][()])
            return row 
        ecg_data = df.select(['id','ecg_date']).to_pandas().apply(add_ecg_features, axis=1)
        ecg_data = pl.from_pandas(ecg_data).with_columns(pl.col('ecg_date').cast(df.select('ecg_date').dtypes[0]))
        df = df.join(ecg_data, on=['id','ecg_date'], how='left')
        df.write_parquet(datafile)

    def get_train_data(self, unique=False): 
        df = self.df.filter(pl.col('split')=='train')
        if unique: df = df.unique(self.features + [self.target])
        df = df.select(self.features + [self.target]).to_pandas()
        X_train, y_train = df.get(self.features).values, df.get(self.target).values
        return X_train, y_train
    
    def get_test_data(self, filters=[]): 
        df = self.df.filter(pl.col('split')=='test')
        for filter in filters: 
            if filter in [
                'recent_hfref','recent_hfmref','recent_hfpef',
                'initial_hfref','initial_hfmref','initial_hfpef',
            ]: 
                df = df.filter(pl.col(filter)==1)
            elif filter=='no_switches_so_far':
                df = df.filter(pl.col('switches_so_far')==0)
            else: raise ValueError(filter)
        df = df.select(self.features + [self.target]).to_pandas()
        X_test, y_test = df.get(self.features).values, df.get(self.target).values
        return X_test, y_test
        


# def wavelet(): 
#     if self.ecg.wavelet: 
#         wavelet_transform = []
#         for i in range(len(self.ecg.channels)):
#             wavelet = 'morl' 
#             fs = 250 
#             dt = 1/fs
#             widths = np.logspace(-1, 2.75, num=1028) 
#             frequencies = pywt.scale2frequency(wavelet, widths) / dt 
#             upper = ([x for x in range(len(widths)) if frequencies[x] > 50])[-1]
#             lower = ([x for x in range(len(widths)) if frequencies[x] < 0.5])[0]
#             if lower-upper>512:
#                 widths = widths[upper:upper+512] 
#             else: widths = widths[upper:lower] 
#             [coefficients, frequencies] = pywt.cwt(signal[i], widths, wavelet = wavelet, sampling_period=dt)
#             magnitude = np.abs(coefficients)
#             idx = np.random.choice(2500-512-250-250) + 250
#             sampled = magnitude[:, idx:idx+512]
#             wavelet_transform.append(sampled)
#         wavelet_transform = np.stack(wavelet_transform, axis=0)
#         wavelet_transform = wavelet_transform / np.max(wavelet_transform)
#         wavelet_transform = torch.Tensor(wavelet_transform).type('torch.FloatTensor')
