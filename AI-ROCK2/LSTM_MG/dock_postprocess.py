#!/usr/bin/env python
import re
import pandas as pd
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem.QED import qed
from RAscore import RAscore_NN
from tqdm import tqdm

def main():
    mols_vl = Chem.SDMolSupplier("sampled_fordock.sdf")

    df_scores = pd.read_csv("vina_scores/scores.txt", sep="\t")
    df_scores_sorted = df_scores.sort_values(by=['scores'], ascending=True)
    df_scores_sorted_btf5 = df_scores_sorted[df_scores_sorted['scores'] < -11.8]
    df_hits = df_scores_sorted_btf5.reset_index(drop=True)

    hits_names = list(df_scores_sorted_btf5['ligand_name'])
    hits_mol = []
    for n in hits_names:
        idx = re.search('mol(\d+)', n).group(1)
        hits_mol.append(mols_vl[int(idx) - 1])

    f5_mol = Chem.MolFromSmiles("C(=O)(N1CCC(C(=O)N2CC=C(c3ccccc3)CC2)CC1)c1n[nH]c(-c2ncccc2)c1")
    f5_fp = AllChem.GetMorganFingerprintAsBitVect(f5_mol, 3, nBits=1024, useFeatures=True)
    nn_scorer = RAscore_NN.RAScorerNN()

    writer = Chem.SDWriter("rock2_f5_hits.sdf")
    for i, mol in enumerate(tqdm(hits_mol)):
        mol_fp = AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=1024, useFeatures=True)
        sims = DataStructs.FingerprintSimilarity(f5_fp, mol_fp)
        mol.SetDoubleProp("sims(FCFP_6)", sims)

        mol.SetDoubleProp("QED", qed(mol))

        smi = Chem.MolToSmiles(mol)
        mol.SetDoubleProp("RA score", float(nn_scorer.predict(smi)))

        mol.SetDoubleProp("vina score", df_hits.iloc[i, 1])
        writer.write(mol)
    writer.close()

if __name__ == "__main__":
    main()