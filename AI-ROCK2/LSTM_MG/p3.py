#!/usr/bin/env python
from RAscore import RAscore_NN
from rdkit import Chem
from tqdm import tqdm
from rdkit.Chem.QED import qed
import os


with open("1012smi_p2.smi", "r") as f:
    list_smiles = [l.rstrip() for l in f]

writer = Chem.SDWriter("final_lstm.sdf")
nn_scorer = RAscore_NN.RAScorerNN()

for i, smi in enumerate(tqdm(list_smiles, ascii=True, desc="deposite with QED and RA score")):
    mol = Chem.MolFromSmiles(smi)
    mol.SetProp("_Name", "molecule" + str(i+1))
    mol.SetProp("SMILES", smi)
    mol.SetProp("QED", str(round(qed(mol), 2)))
    mol.SetProp("RA score", str(round(nn_scorer.predict(smi), 2)))
    writer.write(mol)
writer.close()

os.remove("1012smi_p1.smi")
os.remove("1012smi_p2.smi")
os.remove("datasets/1012cleaned.smi")