import pickle 
import os 
import numpy as np 
from rdkit import Chem
from EcConf.graphs import * 
from EcConf.utils import *
import random 
import math 
from tqdm import tqdm 
"""
datapath='/home/myxu/Workspace2/Dataset/rdkit_folder/drugs'

#activate G2
flist=Find_Files(start_path=datapath,name='pickle')
#random.shuffle(flist)
cut_ratio=0.95
mol_num=0
#print(flist)
trainlist=flist[:int(len(flist)*0.9)]
validlist=flist[int(len(flist)*0.9):int(len(flist)*0.95)]
testlist=flist[int(len(flist)*0.95):]
#print (testlist)
"""
"""
Deal_GEOM_Dataset(trainlist,max_conf_per_conf=5,save_path='molgraphs.pickle')
"""

#with open(flist_file, 'rb')as f:
    #flist = pickle.load(f)
    
'''
with open('datasets/train/flist.csv','r') as f:
    #trainlist=[line.strip() for line in f.readlines()]
    trainlist = '/data/fzg/3DConformersGen/Ch_EcConf/scripts/QM9_chiraltag/datasets/test/QM9/train_data_40k.pkl'
with open('datasets/valid/flist.csv','r') as f:
    #validlist=[line.strip() for line in f.readlines()]
    validlist = '/data/fzg/3DConformersGen/Ch_EcConf/scripts/QM9_chiraltag/datasets/test/QM9/val_data_5k.pkl'
with open('datasets/test/test_data_2971.txt','r') as f:
    #testlist=['/data/fzg/3DConformersGen/rdkit_folder/' + line.strip() for line in f.readlines()]
    testlist = '/data/fzg/3DConformersGen/Ch_EcConf/scripts/QM9_chiraltag/datasets/test/QM9/test_data_1k.pkl'
'''

def read(file):
    with open(file, 'rb')as f:
        flist = pickle.load(f)
        print('type(flist)):', type(flist))
    return flist

root_path  = '/home/bingxing2/home/scx6266'  #替换成你的绝对路径


train_file = f'{root_path}/Ec-Conf/scripts/new_GEMO/QM9/datasets/Origin_GeoDiff_QM9/train_data_40k.pkl'  
trainlist  = read(train_file)

valid_file = f'{root_path}/Ec-Conf/scripts/new_GEMO/QM9/datasets/Origin_GeoDiff_QM9/val_data_5k.pkl'
validlist  = read(valid_file) 

test_file  = f'{root_path}/Ec-Conf/scripts/new_GEMO/QM9/datasets/Origin_GeoDiff_QM9/test_data_1k.pkl'
testlist   = read(test_file)


save_name = 'GeoDiff_QM9' #/home/bingxing2/home/scx6266/Ec-Conf/scripts/GEMO/QM9/datasets

'''
Multi_Process_Creat_GEOM_Molgraphs(trainlist,nmols_per_process=1000000,max_conf_per_mol=5,savepath=f'{root_path}/Ec-Conf/scripts/GEMO/QM9/datasets/{save_name}/train',nprocs=14)
Multi_Process_Creat_GEOM_Molgraphs(validlist,nmols_per_process=1000000,max_conf_per_mol=5,savepath=f'{root_path}/Ec-Conf/scripts/GEMO/QM9/datasets/{save_name}/valid',nprocs=14)
Multi_Process_Creat_GEOM_Molgraphs(testlist,nmols_per_process=1000000,max_conf_per_mol=5,savepath=f'{root_path}/Ec-Conf/scripts/GEMO/QM9/datasets/{save_name}/test',nprocs=14)
#Multi_Process_Creat_Test_GEOM_Molgraphs(testlist,nmols_per_process=10000,max_conf_per_mol=5,savepath='./datasets/test',nprocs=14)
'''     
Multi_Process_Creat_GEOM_Molgraphs(trainlist,nmols_per_process=1000000,max_conf_per_mol=5,savepath=f'GEMO/QM9/datasets/{save_name}/train',nprocs=14)
Multi_Process_Creat_GEOM_Molgraphs(validlist,nmols_per_process=1000000,max_conf_per_mol=5,savepath=f'GEMO/QM9/datasets/{save_name}/valid',nprocs=14)
Multi_Process_Creat_GEOM_Molgraphs(testlist,nmols_per_process=1000000,max_conf_per_mol=5,savepath=f'GEMO/QM9/datasets/{save_name}/test',nprocs=14)
#Multi_Process_Creat_Test_GEOM_Molgraphs(testlist,nmols_per_process=10000,max_conf_per_mol=5,savepath='./datasets/test',nprocs=14) 
