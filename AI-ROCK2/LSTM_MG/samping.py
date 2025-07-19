#!/usr/bin/env python
from torch.utils.data import DataLoader
import argparse
import time
from lstm_chem_pt.utils.config import process_config
from lstm_chem_pt.model import LSTMChem
from lstm_chem_pt.data_set import SMILESDataSet
from lstm_chem_pt.finetuner import LSTMChemFinetuner

def main():
    sample_size = 30000
    config = process_config('experiments/2022-12-29/rock2_f5_ft/config.json')

    modeler = LSTMChem(config, session='finetune')
    finetuner = LSTMChemFinetuner(modeler, None)

    print('start sampling')
    since = time.time()
    finetuned_smiles = finetuner.sample(num=sample_size)
    print(f'sampling finished, {round((time.time() - since), 2)}s elapsed')
    
    with open("sampled_SMILES.smi", "w") as f:
        for smi in finetuned_smiles:
            f.write(smi + "\n")

if __name__ == "__main__":
    main()