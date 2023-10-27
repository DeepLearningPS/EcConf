from rdkit.Geometry.rdGeometry import Point3D
from rdkit import Chem 
from rdkit.Chem import AllChem,rdmolfiles,rdFMCS,Draw
import copy 

def Neutralize_atoms(mol):
    pattern = Chem.MolFromSmarts("[+1!h0!$([*]~[-1,-2,-3,-4]),-1!$([*]~[+1,+2,+3,+4])]")
    at_matches = mol.GetSubstructMatches(pattern)
    at_matches_list = [y[0] for y in at_matches]
    if len(at_matches_list) > 0:
        for at_idx in at_matches_list:
            atom = mol.GetAtomWithIdx(at_idx)
            chg = atom.GetFormalCharge()
            hcount = atom.GetTotalNumHs()
            atom.SetFormalCharge(0)
            atom.SetNumExplicitHs(hcount - chg)
            atom.UpdatePropertyCache()
    return mol


def prepare_mols_from_smi(smiles,savepath='./datasets/mols.smi'):
    with open(savepath,'w') as f:
        mols=[]
        for smi in tqdm(smiles):
            if smi:
                try:
                    mol=Chem.MolFromSmiles(smi)
                    mol=Neutralize_atoms(mol)
                    Chem.Kekulize(mol)
                    if mol:
                        flag=molfielter(mol)
                        if flag:
                            mols.append(mol)
                            f.write(Chem.MolToSmiles(mol)+'\n')
                except Exception as e:
                    print (smi,e)
    return mols 

def prepare_mols_from_sdf(sdfname,smi_path):
    mols=[]
    supp=Chem.rdmolfiles.SDMolSupplier(sdfname)
    with open(f'{smi_path}','w') as f:
        print (f'Prepare mols from {sdfname}')
        for mol in tqdm(supp):
            try:
                mol=Neutralize_atoms(mol)
                Chem.Kekulize(mol)
                if mol:
                    flag=molfielter(mol)
                    if flag:
                        mols.append(mol)
                        f.write(Chem.MolToSmiles(mol)+'\n')
                    #else:
                        #print (Chem.MolToSmiles(mol)+' is not allowed')
            except Exception as e:
                print (e)
                pass
    return mols 

def find_similar_molecules(target_smi,smis):
    target_mol=Chem.MolFromSmiles(target_smi)
    target_mol=Neutralize_atoms(target_mol)
    Chem.Kekulize(target_mol)
    collect_mols=[target_mol]
    for smi in tqdm(smis):
        mol=Chem.MolFromSmiles(smi)
        if mol:
            mol=Neutralize_atoms(mol)
            Chem.Kekulize(mol)
            simi=tanimoto_similarities(target_mol,mol)
            if simi>0.7:
                collect_mols.append(mol)
    return collect_mols

def Change_mol_xyz(rdkitmol,coords):
    molobj=copy.deepcopy(rdkitmol)
    conformer=molobj.GetConformer()
    id=conformer.GetId()
    for cid,xyz in enumerate(coords):
        conformer.SetAtomPosition(cid,Point3D(float(xyz[0]),float(xyz[1]),float(xyz[2])))
    conf_id=molobj.AddConformer(conformer)
    molobj.RemoveConformer(id)
    return molobj