from rdkit import Chem
from rdkit.Chem.rdchem import ChiralType
import json 

class GPARMAS:
    def __init__(self):
        self.atom_types=[1,6,7,8,9,15,16,17,35,53]
        self.bond_types=[Chem.BondType.SINGLE,Chem.BondType.DOUBLE,Chem.BondType.TRIPLE,Chem.BondType.AROMATIC]
        self.if_chiral=True
        self.chiral_types=[ ChiralType.CHI_UNSPECIFIED, ChiralType.CHI_TETRAHEDRAL_CW, ChiralType.CHI_TETRAHEDRAL_CCW, ChiralType.CHI_OTHER,
                            ChiralType.CHI_TETRAHEDRAL, ChiralType.CHI_ALLENE, ChiralType.CHI_SQUAREPLANAR, ChiralType.CHI_TRIGONALBIPYRAMIDAL,ChiralType.CHI_OCTAHEDRAL]
        self.max_atoms=9
        self.batchsize=50
        self.device='cuda'
        self.dim=(16,16)
        self.dim_head=(16,16)
        self.heads=(8,4)
        self.num_linear_att_heads=0
        self.num_degrees=2
        self.depth=6
        self.consistency_training_steps=150
        self.sigma_min=0.002
        self.sigma_max=80.0
        self.rho=7.0
        self.sigma_data=0.5
        self.initial_timesteps=2
        self.final_timesteps=150
        self.lr_patience=100
        self.lr_cooldown=100
        self.n_workers=20
        self.multi_step = 0
            
def Loaddict2obj(dict,obj):
    objdict=obj.__dict__
    for i in dict.keys():
        if i not in objdict.keys():
            print ("%s not is not a standard setting option!"%i)
        objdict[i]=dict[i]
    obj.__dict__==objdict

def Update_GPARAMS(jsonfile):
    with open(jsonfile,'r') as f:
        jsondict=json.load(f)
        Loaddict2obj(jsondict,GP)
    return 

GP=GPARMAS()
