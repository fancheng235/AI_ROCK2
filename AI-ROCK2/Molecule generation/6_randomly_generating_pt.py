#!/usr/bin/env python
from lstm_chem_pt.utils.config import process_config
from lstm_chem_pt.model import LSTMChem
from lstm_chem_pt.generator import LSTMChemGenerator

def main():
    CONFIG_FILE = 'experiments/2022-10-15/LSTM_Chem/config.json'
    config = process_config(CONFIG_FILE)

    modeler = LSTMChem(config, session="generate")
    generator = LSTMChemGenerator(modeler)

    print("start generating ...")
    sampled_smiles = generator.sample(num=1000)
    print("generating finished")

    with open("randomly_generating.smi", "w") as f:
        for smi in sampled_smiles:
            f.write(smi + "\n")

if __name__ == "__main__":
    main()