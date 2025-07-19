#!/usr/bin/env python
from rdkit import Chem
from molvs import Standardizer

def main():
    with open("sampled_SMILES.smi") as f:
        smis_sampled = [l.rstrip() for l in f]

    #remove invalid SMILES
    valid_mols = []
    for l in smis_sampled:
        mol = Chem.MolFromSmiles(l) 
        if mol:
            try:
                Chem.SanitizeMol(mol)
            except ValueError:
                print(f"{l} is not valid !")
            else:
                if len(Chem.MolToSmiles(mol)) != 0:
                    valid_mols.append(mol)  
    print(f'Validity: {len(valid_mols) / len(smis_sampled):.2%}')

    #remove internal duplicates
    valid_smis = [Chem.MolToSmiles(mol) for mol in valid_mols]
    valid_smis_uniq = list(set(valid_smis))
    valid_smis_uniq.sort(key=valid_smis.index)
    print(f'Uniquness: {len(valid_smis_uniq) / len(valid_smis):.2%}')

    #remove duplicates from datasets of transfer learning
    with open("datasets/rock2_ligands_prep_can.smi", "r") as f:
        smis_tf = [l.rstrip() for l in f]
    valid_smis_uniq_notl = list(set(valid_smis_uniq) - set(smis_tf))
    valid_smis_uniq_notl.sort(key=valid_smis_uniq.index)
    print(f'Novelty: {len(valid_smis_uniq_notl) / len(valid_smis_uniq):.2%}')

    #standardize
    mols_preped = []
    s = Standardizer()
    for smi in valid_smis_uniq_notl:
        mol = Chem.MolFromSmiles(smi)
        mol = s.standardize(mol)
        mols_preped.append(mol)

    with open("sampled_preped.smi", "w") as f:
        for mol in mols_preped:
            smi = Chem.MolToSmiles(mol)
            f.write(smi + "\n")

if __name__ == "__main__":
    main()