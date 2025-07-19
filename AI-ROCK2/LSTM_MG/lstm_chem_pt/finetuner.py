from lstm_chem_pt.utils.smiles_tokenizer import SmilesTokenizer
from lstm_chem_pt.generator import LSTMChemGenerator
import time 
from tqdm import tqdm

import torch
from torch import nn
import torch.optim as optim


class LSTMChemFinetuner(LSTMChemGenerator):
    def __init__(self, modeler, finetune_data_loader):
        self.session = modeler.session
        self.model = modeler.model.to(torch.device("cuda"))
        self.config = modeler.config
        self.finetune_data_loader = finetune_data_loader
        self.st = SmilesTokenizer()
    
    def finetune(self):
        train_loss_all = []
        train_acc_all = []
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), 1e-4)
        num_epochs = self.config.finetune_epochs
        self.model.train()
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.95)
        for epoch in range(num_epochs):
            since = time.time()
            print("-"*30)
            print("Epoch {}/{}".format(epoch, num_epochs-1))
            train_loss = 0.0
            train_corrects = 0
            train_num = 0
            for step, batch in enumerate(tqdm(self.finetune_data_loader, ascii=True, desc="finetuning")):
                data_smi, target_smi = batch["input_smi"], batch["label_smi"]
                data_smi, target_smi = data_smi.cuda(non_blocking=False), target_smi.cuda(non_blocking=False)
                out = self.model(data_smi)
                out = out.reshape((-1, out.shape[-1]))
                pre_lab = torch.argmax(out, 1)
                target = target_smi.reshape(-1)

                loss = criterion(out, target.long()).mean()   
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                #torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()
                train_loss += loss.item() * len(target_smi)
                train_corrects += (torch.sum(pre_lab == target) / data_smi.shape[1])
                train_num += len(target_smi)
            train_loss_all.append(train_loss / train_num)
            train_acc_all.append(train_corrects.double().item()/train_num)
            train_time = time.time()
            epoch_time_train = train_time - since
            print("Finetuning Loss: {:.4f} Finetuning Acc: {:.4f}, Epoch Time(Finetune): {}s".format(
                train_loss_all[-1], train_acc_all[-1], epoch_time_train
            ))
            scheduler.step()