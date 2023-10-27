import pickle 
import os 
import numpy as np 
from rdkit import Chem
from EcConf.graphs import * 
from EcConf.utils import *
from EcConf.model import * 
import random 
import math 
from tqdm import tqdm 
import torch

np.random.seed(2023)
torch.manual_seed(2023)
random.seed(2023)
torch.cuda.manual_seed_all(2023)

recover = False #Whether to resume training

local_rank=int(os.environ["LOCAL_RANK"])
Update_GPARAMS('./ctrl.json')
print(torch.cuda.device_count()) 

train_MGFiles=Find_Files('../GEMO/Drugs/datasets/GeoDiff_Drugs/train','ecconf_part')
valid_MGFiles=Find_Files('../GEMO/Drugs/datasets/GeoDiff_Drugs/valid','ecconf_part')


if recover:
    print('reload model')
    model=Equi_Consistency_Model_Parallel(modelname='Equi_Consis_Model',loadtype='Perepoch',local_rank=local_rank)
else:
    print('trian model')
    model=Equi_Consistency_Model_Parallel(local_rank=local_rank)

model.Fit(train_MGFiles,valid_MGFiles,Epochs=1000000)


