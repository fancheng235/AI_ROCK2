from lstm_chem_pt.utils.smiles_tokenizer import SmilesTokenizer
from tqdm import tqdm
import torch
import numpy as np
from torch.nn import functional as F
import time

class LSTMChemGenerator(object):
    def __init__(self, modeler):
        self.session = modeler.session
        self.model = modeler.model
        self.config = modeler.config
        self.st = SmilesTokenizer()
    
    def _generate(self, sequence):
        while (sequence[-1] != 'E') and (len(self.st.tokenize(sequence)) <=
                                         self.config.smiles_max_length):
            x = self.st.tokens_to_ndarray(self.st.tokenize(sequence))#x.shape[1,num_tokens]
            x = torch.from_numpy(x).float().to(torch.device("cuda"))
            preds = F.softmax(self.model(x)[0][-1], dim=0).detach().cpu().numpy()
            next_idx = self.sample_with_temp(preds)
            sequence += self.st.table[next_idx]
        
        sequence = sequence[1:].rstrip('E')
        return sequence

    def sample_with_temp(self, preds):
        streched = np.log(preds) / self.config.sampling_temp
        streched_probs = np.exp(streched) / np.sum(np.exp(streched))
        return np.random.choice(range(len(streched)), p=streched_probs)
    
    def sample(self, num=1, start='G'):
        sampled = []
        start_time = time.time()
        if self.session == 'generate':
            for _ in tqdm(range(num)):
                sampled.append(self._generate(start))
            return sampled
        else:#finetune control validity
            from rdkit import Chem, RDLogger
            RDLogger.DisableLog('rdApp.*')
            while len(sampled) < num:
                sequence = self._generate(start)
                mol = Chem.MolFromSmiles(sequence)
                if mol is not None:
                    canon_smiles = Chem.MolToSmiles(mol)
                    sampled.append(canon_smiles)
                if len(sampled) % 1000 == 0 and len(sampled) != 0:
                    now_time = time.time()
                    print(f"{len(sampled)} molecules sampled, {now_time - start_time}s elapsed")
            return sampled