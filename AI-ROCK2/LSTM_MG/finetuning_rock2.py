#!/usr/bin/env python
import os
import torch
from torch.utils.data import DataLoader

from lstm_chem_pt.utils.config import process_config
from lstm_chem_pt.model import LSTMChem
from lstm_chem_pt.data_set import SMILESDataSet
from lstm_chem_pt.finetuner import LSTMChemFinetuner

def main():

    config = process_config('experiments/2022-10-18/ChemDiv_Specs_20221018/config.json')
    config.finetune_epochs = 12
    config.exp_name = "rock2_ligands_ft"
    config.finetune_data_filename = "datasets/rock2_ligands_prep.smi"
    config.exp_dir = "experiments/2022-12-29/rock2_ligands_ft"

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

    print('saving model ...')
    finetuner.config.model_path = "experiments/2022-12-29/rock2_ligands_ft/rock2_ligands.pkl"
    torch.save(finetuner.model, finetuner.config.model_path) 
    print('model saved.')

    with open(os.path.join(finetuner.config.exp_dir, 'config.json'), 'w') as f:
        f.write(finetuner.config.toJSON(indent=2))

if __name__ == "__main__":
    main()