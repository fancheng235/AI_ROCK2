#!/usr/bin/env python
from rdkit import Chem
from rdkit.Chem import MACCSkeys, DataStructs
from adme_pred import ADME
from tqdm import tqdm
import os


with open("datasets/1012.smi", "r") as f:
    list_smiles0 = [l.rstrip() for l in f]
list_m0 = [Chem.MolFromSmiles(x) for x in list_smiles0]#known actives
list_maccs0 = [MACCSkeys.GenMACCSKeys(x) for x in list_m0]

with open("1012smi_p1.smi", "r") as f:
    list_smiles1 = [l.rstrip() for l in f] #generated smis
list_del_smiles = []
list_retain_smiles = []
for i in tqdm(range(0, len(list_smiles1)), ascii=True, desc="filtering by MACCS FP Similaity 0.75"): #filter by fp sims, threshold 0.75
    m1 = Chem.MolFromSmiles(list_smiles1[i])
    maccs1 = MACCSkeys.GenMACCSKeys(m1)
    flag = 1
    for j in range(0, len(list_smiles0)):
        # calcualte pairwise FingerprintSimilarity between ligand 0 and ligand1
        similarity = DataStructs.FingerprintSimilarity(list_maccs0[j], maccs1)
            # If there is a similar structure, retain the molecule
        if similarity >= 0.75:
                flag = 0
                list_del_smiles.append(list_smiles1[i]) #若有相似结构，那么放入del删除列表
                break
        # Otherwise, delete the molecule
    if flag == 1:
        list_retain_smiles.append(list_smiles1[i]) #若没有相似结构，那么放入remain保留列表
print("Num. of SMILES after preprocess step2.1: ", len(list_retain_smiles))

l_p_retain_smiles = []
for smi in tqdm(list_retain_smiles, ascii=True, desc="filtering by ADME"):
    mol = ADME(smi)
    if mol.druglikeness_lipinski and mol.pains:
        l_p_retain_smiles.append(smi)

with open("1012smi_p2.smi",'w') as infile:
    for smi in l_p_retain_smiles:
        infile.write(smi + '\n')
print("Num. of SMILES after preprocess step2.2: ", len(l_p_retain_smiles))
