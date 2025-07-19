#!/usr/bin/env python
from rdkit import Chem
from adme_pred import ADME
from rdkit.Chem.QED import qed
from tqdm import tqdm
import sascorer

def main():
    with open("sampled_preped.smi") as f:
        smis_preped = [l.rstrip() for l in f]

    #filter1: retain compounds obeying Ro5 and remove PAINS
    smis_filtered_1 = []
    for smi in smis_preped:
        mol = ADME(smi)
        if mol.druglikeness_lipinski() and not mol.pains():
            smis_filtered_1.append(smi)

    #filter2:QED>0.3, SAscore<5
    mols_filtered_2 = []
    for smi in tqdm(smis_filtered_1):
        mol = Chem.MolFromSmiles(smi)
        qed_mol = qed(mol)
        sascore_mol = sascorer.calculateScore(mol)
        if qed_mol>0.3 and sascore_mol<5:
            mols_filtered_2.append(mol)

    writer = Chem.SDWriter("sampled_fordock.sdf")
    for i, mol in enumerate(tqdm(mols_filtered_2)):
        mol.SetProp("_Name", "molecule" + str(i+1))
        writer.write(mol)
    writer.close()

if __name__ == "__main__":
    main()