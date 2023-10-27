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
from rdkit.Chem.rdForceFieldHelpers import MMFFOptimizeMolecule

from collections import defaultdict

import shutil
#shutil.make_archive('Equi_Consis_Model',"zip",'Equi_Consis_Model') #打包模型

#dirs = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
#dirs = '2023_05_12_23_18_54'

def ff(mol_list):
    #像rdkit一样使用经验力场来优化生成的分子
    opt_rdmols_list = []
    for mol in mol_list: #遍历每一个构象mol 
        opt_mol = copy.deepcopy(mol)
        MMFFOptimizeMolecule(opt_mol)
        opt_rdmols_list.append(opt_mol)
    return opt_rdmols_list 


def sample(step, num):

    #local_rank=int(os.environ["LOCAL_RANK"])
    Update_GPARAMS('./ctrl.json')
    GP.bond_types=[Chem.BondType.SINGLE,Chem.BondType.DOUBLE,Chem.BondType.TRIPLE,Chem.BondType.AROMATIC]

    GP.final_timesteps = step#设置步长5,10,25,50,100.

    #dirs = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
    path = os.path.join(f'GeoDiff_Drugs_evaluation_{dirs}', f'step{step}', 'conformation')
    os.makedirs(path, exist_ok = True)
    
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
    
    '''
    mgdict = {}
    count = 0
    for k in list(old_mgdict.keys()):  
        mgdict[k] = old_mgdict[k]
        count += 1
        if count == 200:
            break
    '''
        
    
    #print('mgdict:', len(mgdict)) #mgdict: 200 
    #raise Exception('test')
    


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
    error_sm_list = []
    kid=0
    count_error = 0
    coverage_recall,amr_recall,coverage_precision, amr_precision=[],[],[],[]
    for key in tqdm(list(mgdict.keys())):
        confnum=len(mgdict[key])
        print (confnum)
        mgs=mgdict[key] #构象列表
        mols=model.Sample(mgs,conf_num_per_mol=2) #conf_num_per_mol每一个构象复制2份，返回的是坐标更新后的构象
        if use_ff:
            mols = ff(mols) #使用FF经验力场看看效果，估计着意义不大，这种方法当生成的构象很不好的时候才有用，QM9应该作用不大，到是Drugs可能有用
        
        target_mgdict[key]=[]
        
        try:
            for mg in mgdict[key]:
                target_mgdict[key].append(mg.Trans_to_Rdkitmol()) #保存真实构象,会报错误，rdkit.Chem.rdchem.AtomValenceException: Explicit valence for atom # 16 N, 4, is greater than permitted
        except Exception:
            error_sm_list.append(key)
            continue #如果有问题，则过滤该分子
            
        
        sampled_mgdict[key]=mols  #采样出来的预测构象
        supp=Chem.SDWriter(f'{path}/{kid}_ecconf.sdf') #保存的sdf文件的构象去除了H氢原子
        for mol in mols:
            supp.write(mol)
        supp.close()    #需要手动关闭
        supp=Chem.SDWriter(f'{path}/{kid}_target.sdf')
        for mol in target_mgdict[key]:
            supp.write(mol)
        supp.close()
        
        #coverage_recall, amr_recall, coverage_precision, amr_precision=calc_performance_stats(target_mgdict[key], sampled_mgdict[key])
        try:
            cr,ar,cp,ap=calc_performance_stats(target_mgdict[key], sampled_mgdict[key]) #这里可能会出现None的数据，即生成的数据不好，经过取氢等操作后为None
        except Exception:
            count_error += 1
            continue

        coverage_recall.append(cr)
        amr_recall.append(ar)
        coverage_precision.append(cp)
        amr_precision.append(ap)
        #print (coverage_recall, amr_recall, coverage_precision, amr_precision)
        kid+=1
        
    with open('geodiff_error_sm_list.txt', 'w')as f:
        for i in error_sm_list:
            f.write(str(i) + '\n')
            
    #np.save(f'{path}/stats.npy',[coverage_recall, amr_recall, coverage_precision, amr_precision])
    #np.savetext
    '''
    for stat in coverage_recall:
        print('stat:', stat)
        exit()
        if stat[5] == 1.25:
            pass
        else:
            raise Exception('stat[5] != 1.25')
    '''
    coverage_recall_vals = [stat[5] for stat in coverage_recall] #stat[2]用来设置COV节点的位置，如0.5,1.25
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
    print('target_error num:', len(error_sm_list))
    print('gen_error num:', count_error)
    return json_str, path

if __name__ == '__main__':
    np.random.seed(2023)
    torch.manual_seed(2023)
    random.seed(2023)
    torch.cuda.manual_seed_all(2023)
    
    #shutil.make_archive('Equi_Consis_Model',"zip",'Equi_Consis_Model') #打包模型
    
    num = 200
    #num = 1
    
    #use_ff = False
    #dirs = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
    #json_str = sample(15, num)
    
    s_dict = {}
    #for s in [1, 2, 5, 10, 15, 25, 30, 35, 40, 45, 50][:-4]: #
    #for s in [15, 25, 50, 100]:
    for s in [25]:
        use_ff = False
        dirs = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
        json_str = sample(s, num)
        s_dict[s] = json_str
    print('\n----------------------------------------------------------------')
    
    for s in s_dict.keys():
        print('Step:', s)
        print(s_dict[s][0])
        current_directory = os.path.dirname(os.path.abspath('__file__')) #获取当前路径
        print('file_path:', current_directory + '/' + s_dict[s][1])
        print('====================================================\n')