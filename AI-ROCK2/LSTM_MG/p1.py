#!/usr/bin/env python
from rdkit import Chem


rawsmi_input = 'sampled_SMILES.smi'
known_input = 'datasets/1012cleaned.smi'
smi_output = '1012smi_p1.smi'

smi_l1 = []
smi_l2 = []
smi_l3 = []
valid_mols = []
valid_mols_smi_l = []
final = []

with open(rawsmi_input) as file_object:
    lines = file_object.readlines()
for line in lines:
    smi_l1.append(line.rstrip())

#去除不合理分子
for l in smi_l1:
    mol = Chem.MolFromSmiles(l) 
    if mol:
        try:
            Chem.SanitizeMol(mol) #test whether mol is valid
        except ValueError:
            print(f"{l} is not valid !")
        else:
            valid_mols.append(mol)  
print(f'Validity: {len(valid_mols) / len(smi_l1):.2%}')

for m in valid_mols:
    valid_mols_smi = Chem.MolToSmiles(m)
    valid_mols_smi_l.append(valid_mols_smi)

#去除内部重复分子
smi_l2 = list(set(valid_mols_smi_l))
smi_l2.sort(key = valid_mols_smi_l.index)
print(f'Uniquness: {len(smi_l2) / len(valid_mols_smi_l):.2%}')

#去除已知重复分子
with open(known_input) as file_object:
    lines2 = file_object.readlines()
for line2 in lines2:
    smi_l3.append(line2.rstrip())

x = set(smi_l2)
y = set(smi_l3)
fin =list(x - y)
print(f'Novelty: {len(fin) / len(x):.2%}')
fin.sort(key = smi_l2.index)

with open(smi_output,'w') as infile:
    for smi in fin:
        infile.write(smi + '\n')
print("Num. of SMILES after preprocess step1: ", len(fin))