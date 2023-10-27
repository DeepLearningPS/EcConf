from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np 

def calc_performance_stats(true_mols, model_mols):

    threshold = np.arange(0, 2.0, 0.25) # threshold = np.arange(0, 2.0, .25) #看1.25的下标 a[2] = 0.5; a[5] = 1.25
    rmsd_list = []
    for tc in true_mols:
        for mc in model_mols:

            try:
                rmsd_val = AllChem.GetBestRMS(Chem.RemoveHs(tc), Chem.RemoveHs(mc))
            except RuntimeError:
                return None
            rmsd_list.append(rmsd_val)

    rmsd_array = np.array(rmsd_list).reshape(len(true_mols), len(model_mols))

    coverage_recall = np.sum(rmsd_array.min(axis=1, keepdims=True) < threshold, axis=0) / len(true_mols)
    amr_recall = rmsd_array.min(axis=1).mean()

    coverage_precision = np.sum(rmsd_array.min(axis=0, keepdims=True) < np.expand_dims(threshold, 1), axis=1) / len(model_mols)
    amr_precision = rmsd_array.min(axis=0).mean()

    return coverage_recall, amr_recall, coverage_precision, amr_precision

