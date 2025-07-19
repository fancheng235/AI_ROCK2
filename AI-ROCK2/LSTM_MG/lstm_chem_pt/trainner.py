import time
from tqdm import tqdm
import torch
import copy
from tensorboardX import SummaryWriter


def train_model(model, traindataloader, 
        validdataloader, criterion, 
        optimizer, config, num_epochs=22):
    sumwriter = SummaryWriter(log_dir=config.tensorboard_log_dir)
    val_loss_0 = 1.5
    train_loss_accu = 0
    train_loss_all = []
    train_acc_all = []
    valid_loss_accu = 0
    valid_loss_all = []
    valid_acc_all = []
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.95)
    for epoch in range(num_epochs):
        since = time.time()
        print("-"*30)
        print("Epoch {}/{}".format(epoch, num_epochs-1))
        train_loss = 0.0
        train_corrects = 0
        train_num = 0
        valid_loss = 0.0
        valid_corrects = 0
        valid_num = 0
        model.train()    
        for step, batch in enumerate(tqdm(traindataloader, ascii=True, desc="training")):
            data_smi, target_smi = batch["input_smi"], batch["label_smi"]
            data_smi, target_smi = data_smi.cuda(non_blocking=False), target_smi.cuda(non_blocking=False)
            out = model(data_smi)
            out = out.reshape((-1, out.shape[-1]))
            pre_lab = torch.argmax(out, 1)
            target = target_smi.reshape(-1)

            loss = criterion(out, target.long()).mean()   
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            train_loss += loss.item() * len(target_smi)
            train_loss_accu += loss.item()
            train_corrects += (torch.sum(pre_lab == target) / data_smi.shape[1])
            train_num += len(target_smi)
            niter = epoch * len(traindataloader) + step + 1
            if niter % 100 == 0:
                sumwriter.add_scalar("train loss", train_loss_accu / niter, niter)
                sumwriter.add_scalar("train acc", train_corrects.double().item() / train_num, niter)
        train_loss_all.append(train_loss / train_num)
        train_acc_all.append(train_corrects.double().item()/train_num)
        train_time = time.time()
        epoch_time_train = train_time - since
        print("Train Loss: {:.4f} Train Acc: {:.4f}, Epoch Time(Train): {}s".format(
            train_loss_all[-1], train_acc_all[-1], epoch_time_train
        ))
        scheduler.step()
        model.eval()
        for step, batch in enumerate(tqdm(validdataloader, ascii=True, desc="validating")):
            data_smi, target_smi = batch["input_smi"], batch["label_smi"]
            data_smi, target_smi = data_smi.cuda(), target_smi.cuda()
            out = model(data_smi)
            out = out.reshape((-1, out.shape[-1]))
            pre_lab = torch.argmax(out, 1)
            target = target_smi.reshape(-1)

            loss = criterion(out, target.long()).mean()   
            valid_loss += loss.item() * len(target_smi)
            valid_loss_accu += loss.item()
            valid_corrects += (torch.sum(pre_lab == target) / data_smi.shape[1])
            valid_num += len(target_smi)
            niter = epoch * len(validdataloader) + step + 1
            if niter % 10 == 0:
                sumwriter.add_scalar("valid loss", valid_loss_accu / niter, niter)
                sumwriter.add_scalar("valid acc", valid_corrects.double().item() / valid_num, niter)            
        valid_loss_all.append(valid_loss / valid_num)
        valid_acc_all.append(valid_corrects.double().item()/valid_num)
        valid_time = time.time()
        epoch_time_valid = valid_time - train_time
        print("Valid Loss: {:.4f} Valid Acc: {:.4f}, Epoch Time(Valid): {}s".format(
            valid_loss_all[-1], valid_acc_all[-1], epoch_time_valid
        ))

        if valid_loss_all[-1] < val_loss_0:
            val_loss_0 = valid_loss_all[-1]
            best_model_wts = copy.deepcopy(model.state_dict())
    return best_model_wts
