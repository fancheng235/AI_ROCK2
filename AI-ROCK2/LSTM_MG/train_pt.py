#!/usr/bin/env python3
from lstm_chem_pt.utils.config import process_config
from lstm_chem_pt.utils.dirs import create_dirs
from lstm_chem_pt.data_set import SMILESDataSet
from lstm_chem_pt.trainner import train_model
from lstm_chem_pt.model import LSTMChem
import os
from torch.utils.data import DataLoader
import torch
from torch import nn
import torch.optim as optim


CONFIG_FILE = 'base_config.json'
CTX = torch.device("cuda")
def main():
    config = process_config(CONFIG_FILE)

    create_dirs(
        [config.exp_dir, config.tensorboard_log_dir])

    print('Create the data generator.')
    train_ds = SMILESDataSet(config, data_type="train")
    train_dl = DataLoader(
        dataset=train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        pin_memory=False,
        num_workers=config.num_workers)

    valid_ds = SMILESDataSet(config, data_type="valid")
    valid_dl = DataLoader(
        dataset=valid_ds,
        batch_size=config.batch_size,
        shuffle=True,
        pin_memory=False,
        num_workers=config.num_workers)

    print('Create the model.')
    lstm = LSTMChem(config, session='train')
    lstm.model.to(CTX)
    optimizer = optim.Adam(lstm.model.parameters())
    loss_func = nn.CrossEntropyLoss()
    print('Start training the model.')
    best_model_wts = train_model(lstm.model, train_dl, valid_dl, loss_func, optimizer, lstm.config, config.num_epochs)
    lstm.model.load_state_dict(best_model_wts)
    print('Saving model ...')
    torch.save(lstm.model, lstm.config.model_path)
    print('model saved.')

    with open(os.path.join(lstm.config.exp_dir, 'config.json'), 'w') as f:
        f.write(lstm.config.toJSON(indent=2))

if __name__ == '__main__':
    main()
