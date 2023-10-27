import pickle 
import os 
import numpy as np 
from rdkit import Chem
from EcConf.graphs import * 
from EcConf.utils import *
from EcConf.model import * 
from EcConf import * 
import random 
import math 
from tqdm import tqdm 
import argparse as arg
from rdkit import Chem
from EcConf.graphs import  *
from EcConf.utils import *
from EcConf.comparm import *
from EcConf import * 
import pickle 
import json

from collections import defaultdict

#evaluate 包
import numpy as np 
import pickle
import os 
from EcConf.graphs import * 
from EcConf.utils import *
from rdkit import Chem
from rdkit.Chem import *
from tqdm import tqdm

import shutil


dirs = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
#dirs = '2023_05_12_23_18_54'


def sample(step, num):

    #local_rank=int(os.environ["LOCAL_RANK"])
    Update_GPARAMS('./ctrl.json')
    GP.bond_types=[Chem.BondType.SINGLE,Chem.BondType.DOUBLE,Chem.BondType.TRIPLE,Chem.BondType.AROMATIC]

    GP.final_timesteps = step#设置步长5,10,25,50,100
    print('GP.multi_step:', GP.multi_step)

    #dirs = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
    step_path = {} #为每一个步长创建目录路径
    for stp in np.arange(step) + 1 :
        path = os.path.join(f'GeoDiff_Drugs_evaluation_{dirs}', f'step{stp}', 'conformation')
        os.makedirs(path, exist_ok = True)
        step_path[stp] = path
    
    #将当前的模型也保存复制一份
    shutil.copy('Equi_Consis_Model.zip', f'GeoDiff_Drugs_evaluation_{dirs}')  #shutil.copy('源文件路径', '目标文件路径')

    print(torch.cuda.device_count()) 

    film = [  0,   2,   8,  11,  16,  18,  20,  29,  31,  33,  34,  35,  44,
            49,  52,  54,  57,  58,  60,  64,  74,  79,  80,  84, 104, 107,
        110, 117, 120, 123, 126, 141, 143, 145, 147, 154, 161, 165, 166,
        167, 170, 175, 177, 181, 186, 191, 192, 193, 199]


    model=Equi_Consistency_Model(modelname='Equi_Consis_Model',loadtype='Perepoch')
    with open('../GEMO/Drugs/datasets/GeoDiff_Drugs/test/ecconf_all.pickle','rb') as f:
        mg_list=pickle.load(f)
        
    
    print('mg_list:', len(mg_list)) #mg_list: 39782
        
    old_mgdict = defaultdict(list)
    
    for conf in mg_list:
        old_mgdict[conf.smiles].append(conf)
    
    print('old_mgdict:', len(old_mgdict)) #old_mgdict: 344
    
    new_mgdict = {}
    for idx, k in enumerate(old_mgdict):
        #去手性，去掉带有手性构象的分子
        #if idx not in film: mgdict[k] = old_mgdict[k]
        
        #if idx in film: continue
        
        new_mgdict[k] = old_mgdict[k]

    mgdict = {}
    for k in list(new_mgdict.keys())[:num]:
        mgdict[k] = new_mgdict[k]
    
    print('new_mgdict: {}'.format(len(new_mgdict)))   
    print('mgdict: {}'.format(len(mgdict)))

    sampled_mgdict={}
    target_mgdict={}
    step_target_error_dict = defaultdict(int)
    step_gen_error_dict = defaultdict(int)
    kid=0
    
    coverage_recall_dict,amr_recall_dict,coverage_precision_dict, amr_precision_dict=defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
    for mol_i, key in tqdm(enumerate(list(mgdict.keys()))):
        confnum=len(mgdict[key])
        print (confnum)
        mgs=mgdict[key]
        if GP.multi_step == 1:
            mols_dict=model.Sample(mgs,conf_num_per_mol=2)
        else:
            mols=model.Sample(mgs,conf_num_per_mol=2)
        
        flag = 0
    
        for k in list(step_path.keys())[:]:
            path = step_path[k]
            mols = mols_dict[k]
            
            target_mgdict[key]=[]


            try:
                for mg in mgdict[key]:
                    target_mgdict[key].append(mg.Trans_to_Rdkitmol()) #保存真实构象,会报错误，rdkit.Chem.rdchem.AtomValenceException: Explicit valence for atom # 16 N, 4, is greater than permitted
            except Exception:
                #error_sm_list.append(key)
                step_target_error_dict[k] += 1
                flag = 1
                continue #如果有问题，则过滤该分子


            sampled_mgdict[key]=mols
            supp=Chem.SDWriter(f'{path}/{kid}_ecconf.sdf') #保存的sdf文件的构象去除了H氢原子
            for mol in mols:
                supp.write(mol)
            supp.close()
            supp=Chem.SDWriter(f'{path}/{kid}_target.sdf')
            for mol in target_mgdict[key]:
                supp.write(mol)
            supp.close()
            
            #coverage_recall, amr_recall, coverage_precision, amr_precision=calc_performance_stats(target_mgdict[key], sampled_mgdict[key])
            try:
                cr,ar,cp,ap=calc_performance_stats(target_mgdict[key], sampled_mgdict[key]) #这里可能会出现None的数据，即生成的数据不好，经过取氢等操作后为None
            except Exception:
                #count_error += 1
                step_gen_error_dict[k] += 1
                continue
            coverage_recall_dict[k].append(cr)
            amr_recall_dict[k].append(ar)
            coverage_precision_dict[k].append(cp)
            amr_precision_dict[k].append(ap)
            #print (coverage_recall, amr_recall, coverage_precision, amr_precision)
            
        if flag == 0:
            kid+=1
            
    for k in list(step_path.keys())[:]:
        path = step_path[k]
        
        coverage_recall, amr_recall, coverage_precision, amr_precision = coverage_recall_dict[k], amr_recall_dict[k], coverage_precision_dict[k], amr_precision_dict[k]
        #np.save(f'{path}/stats.npy',[coverage_recall, amr_recall, coverage_precision, amr_precision])
        #np.savetext
        coverage_recall_vals = [stat[5] for stat in coverage_recall]
        coverage_precision_vals = [stat[5] for stat in coverage_precision]
        #print(f'Recall Coverage: Mean = {np.mean(coverage_recall_vals)*100:.2f}, Median = {np.median(coverage_recall_vals)*100:.2f}')
        #print(f'Recall AMR: Mean = {np.nanmean(amr_recall):.4f}, Median = {np.nanmedian(amr_recall):.4f}')
        #print()
        #print(f'Precision Coverage: Mean = {np.mean(coverage_precision_vals)*100:.2f}, Median = {np.median(coverage_precision_vals)*100:.2f}')
        #print(f'Precision AMR: Mean = {np.nanmean(amr_precision):.4f}, Median = {np.nanmedian(amr_precision):.4f}')



        r_dict = {}
        r_dict['Recall_Coverage_Mean'] = f'{np.mean(coverage_recall_vals):.4f}'
        r_dict['Recall_Coverage_Median'] = f'{np.median(coverage_recall_vals):.4f}'
        r_dict['Recall_AMR_Mean'] = f'{np.nanmean(amr_recall):.4f}'
        r_dict['Recall_AMR_Median'] = f'{np.nanmedian(amr_recall):.4f}'

        r_dict['Precision_Coverage_Mean'] = f'{np.mean(coverage_precision_vals):.4f}'
        r_dict['Precision_Coverage_Median'] = f'{np.median(coverage_precision_vals):.4f}'
        r_dict['Precision_AMR_Mean'] = f'{np.nanmean(amr_precision):.4f}'
        r_dict['Precision_AMR_Median'] = f'{np.nanmedian(amr_precision):.4f}'


        with open(f'{path}/resault.json', 'w')as f:
            json.dump(r_dict, f, indent=4)

        with open(f'{path}/resault.txt', 'w')as f:
            f.write(f'Recall Coverage: Mean = {np.mean(coverage_recall_vals):.4f}, Median = {np.median(coverage_recall_vals):.4f}\n')
            f.write(f'Recall AMR: Mean = {np.nanmean(amr_recall):.4f}, Median = {np.nanmedian(amr_recall):.4f}\n')
            f.write(f'Precision Coverage: Mean = {np.mean(coverage_precision_vals):.4f}, Median = {np.median(coverage_precision_vals):.4f}\n')
            f.write(f'Precision AMR: Mean = {np.nanmean(amr_precision):.4f}, Median = {np.nanmedian(amr_precision):.4f}\n')


        json_str = json.dumps(r_dict, indent=4)
        print(json_str)
        current_directory = os.path.dirname(os.path.abspath('__file__')) #获取当前路径
        print('file_path:', current_directory + '/' + path)
        print('target_error num:', step_target_error_dict[k])
        print('gen_error num:', step_gen_error_dict[k])


if __name__ == '__main__':
    #shutil.make_archive('Equi_Consis_Model',"zip",'Equi_Consis_Model') #打包模型
    np.random.seed(2023)
    torch.manual_seed(2023)
    random.seed(2023)
    
    torch.cuda.manual_seed_all(2023)
    GP.multi_step = 1
    num = 200 
    sample(25, num)