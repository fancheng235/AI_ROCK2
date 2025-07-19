import numpy as np
from tqdm import tqdm
from lstm_chem_pt.utils.smiles_tokenizer import SmilesTokenizer

import torch
from torch.utils.data import Dataset

class SMILESDataSet(Dataset):
    def __init__(self, config, data_type='train') -> None:
        self.config = config
        self.data_type = data_type
        assert self.data_type in ['train', 'valid', 'finetune']

        self.max_len = 0

        if self.data_type in ['train', 'valid']:
            self.smiles = self._load(self.config.data_filename)
        else:
            self.smiles = self._load(self.config.finetune_data_filename)

        self.st = SmilesTokenizer()
        self.token_to_idx = self.st.token_to_idx
        self.tokenized_smiles = self._tokenize(self.smiles)

        if self.data_type in ['train', 'valid']:
            self.idx = np.arange(len(self.tokenized_smiles))
            self.valid_size = int(
                np.ceil(
                    len(self.tokenized_smiles) * self.config.validation_split))
            np.random.seed(self.config.seed)
            np.random.shuffle(self.idx)
        
        self.target_tokenized_smiles = self._set_data()

    def _set_data(self):
        if self.data_type == 'train':
            ret = [
                self.tokenized_smiles[self.idx[i]]
                for i in self.idx[self.valid_size:]
            ]
        elif self.data_type == 'valid':
            ret = [
                self.tokenized_smiles[self.idx[i]]
                for i in self.idx[:self.valid_size]
            ]
        else:
            ret = self.tokenized_smiles
        return ret

    def __len__(self):
        ret = len(self.target_tokenized_smiles)
        return ret

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        data = self.target_tokenized_smiles[idx]
        data = self._pad(data)

        self.X = [self.token_to_idx[symbol] for symbol in data[:-1]]
        self.y = [self.token_to_idx[symbol] for symbol in data[1:]]
        
        self.X = torch.from_numpy(np.array(self.X, dtype=np.float32)) #X.shape (max_len+1,)
        self.y = torch.from_numpy(np.array(self.y, dtype=np.float32))

        sample = {"input_smi":self.X, "label_smi":self.y}

        return sample

    def _load(self, data_filename):
        length = self.config.data_length
        print('loading SMILES...')
        with open(data_filename) as f:
            smiles = [s.rstrip() for s in f]
        if length != 0:
            smiles = smiles[:length]
        print('done.')
        return smiles

    def _tokenize(self, smiles):
        assert isinstance(smiles, list)
        print('tokenizing SMILES...')
        tokenized_smiles = [self.st.tokenize(smi) for smi in tqdm(smiles)]

        if self.data_type in ['train', 'valid']:
            for tokenized_smi in tokenized_smiles:
                length = len(tokenized_smi)
                if self.max_len < length:
                    self.max_len = length
            if self.data_type == 'train':
                self.config.train_smi_max_len = self.max_len
            else:
                self.config.valid_smi_max_len = self.max_len
        else:
            for tokenized_smi in tokenized_smiles:
                length = len(tokenized_smi)
                if self.max_len < length:
                    self.max_len = length
            self.config.finetune_smi_max_len = self.max_len
            
        print('done.')
        return tokenized_smiles

    def _pad(self, tokenized_smi):
        return ['G'] + tokenized_smi + ['E'] + [
            'A' for _ in range(self.max_len - len(tokenized_smi))
        ]
