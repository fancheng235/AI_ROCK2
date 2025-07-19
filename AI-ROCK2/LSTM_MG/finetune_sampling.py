#!/usr/bin/env python
from torch.utils.data import DataLoader
import argparse
import time
from lstm_chem_pt.utils.config import process_config
from lstm_chem_pt.model import LSTMChem
from lstm_chem_pt.data_set import SMILESDataSet
from lstm_chem_pt.finetuner import LSTMChemFinetuner


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LSTMChem-Finetuning and Sampling")
    parser.add_argument("--epochs", help="number of epochs for finetuning", default=5)
    parser.add_argument("--size", help="number of sampled SMILES", default=256)
    args = parser.parse_args()

    epochs = int(args.epochs)
    size = int(args.size)

    config = process_config('experiments/2022-10-18/ChemDiv_Specs_20221018/config.json')
    config.finetune_epochs = epochs

    modeler = LSTMChem(config, session='finetune')
    finetune_ds = SMILESDataSet(config, data_type="finetune")
    finetune_dl = DataLoader(
        dataset=finetune_ds,
        batch_size=config.finetune_batch_size,
        shuffle=True,
        pin_memory=False,
        num_workers=config.num_workers)

    finetuner = LSTMChemFinetuner(modeler, finetune_dl)
    print('start finetuning the model')
    finetuner.finetune()
    print('finetuning finished')

    print('start sampling')
    since = time.time()
    finetuned_smiles = finetuner.sample(num=size)
    print(f'sampling finished, consuming {round((time.time() - since), 2)}s')
    
    with open("sampled_SMILES.smi", "w") as f:
        for smi in finetuned_smiles:
            f.write(smi + "\n")