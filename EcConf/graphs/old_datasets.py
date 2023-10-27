import pickle 
import os 
import numpy as np 
from rdkit import Chem
from EcConf.graphs import * 
from EcConf.utils import *
import random 
import math 
from tqdm import tqdm 
from torch.utils.data import Dataset,DataLoader
from ..comparm import *
import copy

def Deal_GEOM_Dataset(flist,max_conf_per_conf,save_path='molgraphs.pickle'):
    molgraphs=[]
    for fname in tqdm(flist):
        with open(f'{fname}','rb') as f:
            try:
                a=pickle.load(f)
                conformers=a['conformers']
                smiles=[]
                energies=[]
                for conf in conformers:
                    energies.append(conf['totalenergy'])
                energies=np.array(energies)
                lowest_ids=np.argsort(energies)[:max(max_conf_per_conf,len(conformers))]
                selected_mols=[conformers[i]['rd_mol'] for i in lowest_ids]
                for mol in selected_mols:
                    #try:
                        mol_noH=Chem.rdmolops.RemoveHs(mol,sanitize=True)
                        mol_noH=Neutralize_atoms(mol_noH)
                        Chem.Kekulize(mol_noH)
                        atoms=[atom.GetAtomicNum() for atom in mol_noH.GetAtoms()]
                        natoms=sum([1 for i in atoms if i!=1])
                        smi=Chem.MolToSmiles(mol_noH)
                        if natoms>3 and '.' not in smi:
                            molgraph=Molgraph(mol_noH,smiles=smi)
                            molgraphs.append(molgraph)
                            if molgraph.natoms!=len(molgraph.atoms):
                                print (e,smi,fname)
                    #except Exception as e:
                        #pass
            except:
                print (fname)
    with open(save_path,'wb') as f:
        pickle.dump(molgraphs,f)
    return 

def Multi_Process_Creat_GEOM_Molgraphs(flist,nmols_per_process,max_conf_per_mol,savepath='./molgraphs',nprocs=14):
    from multiprocessing import Pool,Queue,Manager,Process
    manager=Manager()
    DQueue=manager.Queue()
    nmols=len(flist)
    njobs=math.ceil(nmols/nmols_per_process)

    if not os.path.exists(savepath):
        os.system(f'mkdir -p {savepath}')
    with open(f'{savepath}/flist.csv','w') as f:
        for fname in flist:
            f.write(fname+'\n')

    p=Pool(nprocs)
    resultlist=[]
    for i in range(njobs):
        result=p.apply_async(Deal_GEOM_Dataset,(flist[i*nmols_per_process:(i+1)*nmols_per_process],max_conf_per_mol,f'{savepath}/part_{i}.pickle'))
        resultlist.append(result)
    for i in range(len(resultlist)):
        tmp=resultlist[i].get()
        print (tmp)
    p.terminate()
    p.join()
    print (f'Mols have all trans to Molgraphs in {savepath}')
    return 

def Statistic_GPARAMS(MGFiles):
    params={'atom_types':[],'max_atoms':0}
    Hmols=0
    for fname in MGFiles:
        with open(fname,'rb') as f:
            mgs=pickle.load(f)
        for mg in tqdm(mgs):
            if mg.natoms > params["max_atoms"]:
                params["max_atoms"]=mg.natoms
            if 1 in mg.atoms:
                Hmols+=1
            for a in mg.atoms:
                if a not in params["atom_types"]:
                    params["atom_types"].append(a)
    params["atom_types"]=np.sort(params["atom_types"])
    print (Hmols)
    return params


class MG_Dataset(Dataset):
    def __init__(self,MGlist,name):
        super(Dataset,self).__init__()
        self.mglist=MGlist
        self.name=name
        self.nmols=len(self.mglist)
        self.max_atoms=GP.max_atoms
        return 
    def __len__(self):
        return len(self.mglist)
    def __getitem__(self,idx):
        return self.getitem__(idx)
    def getitem__(self,idx):
        mg=copy.deepcopy(self.mglist[idx])
        #mg.RemoveHs()
        atoms,chiraltags,adjs,coords,zmats,masks=mg.Get_3D_Graph_Tensor(max_atoms=self.max_atoms)
        if not GP.if_chiral:
            feats=atoms
        else:
            feats=torch.concat((atoms,chiraltags),axis=-1)
        return {
                'Feats':feats,
                "Adjs":adjs,
                "Coords":coords,
                "Zmats":zmats,
                "Masks":masks
                }



        