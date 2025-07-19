from torch.utils.data import DataLoader
from tqdm import tqdm
import time
import torch
from lstm_chem_pt.utils.config import process_config
from lstm_chem_pt.model import LSTMChem
from lstm_chem_pt.data_set import SMILESDataSet


if __name__ == "__main__":
    config = process_config('experiments/2022-10-18/ChemDiv_Specs_20221018/config.json')
    modeler = LSTMChem(config, session='finetune')
    modeler.model = torch.load(config.model_path)
    test_ds = SMILESDataSet(config, data_type="train")
    test_dl = DataLoader(
        dataset=test_ds,
        batch_size=256,
        shuffle=True,
        pin_memory=False,
        num_workers=config.num_workers)
    test_acc_all = []
    test_corrects = 0
    test_num = 0
    modeler.model.eval()
    since = time.time()
    for step, batch in enumerate(tqdm(test_dl, ascii=True, desc="testing")):
        data_smi, target_smi = batch["input_smi"], batch["label_smi"]
        data_smi, target_smi = data_smi.cuda(), target_smi.cuda()
        out = modeler.model(data_smi)
        out = out.reshape((-1, out.shape[-1]))
        pre_lab = torch.argmax(out, 1)
        target = target_smi.reshape(-1)

        test_corrects += (torch.sum(pre_lab == target) / data_smi.shape[1])
        test_num += len(target_smi)          
    test_acc_all.append(test_corrects.double().item()/test_num)
    test_time = time.time()
    epoch_time_test = test_time - since
    print("test Acc: {:.4f}, Epoch Time(test): {}s".format(
            test_acc_all[-1], epoch_time_test
        ))