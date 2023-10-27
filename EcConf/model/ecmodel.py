import torch
from .equiformer import * 
from .consistency import * 
import pickle,os,tempfile, shutil, zipfile, time, math, tqdm 
from datetime import datetime 
from ..comparm import * 
from ..utils.utils_torch import *
from torch.optim import Adam
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR, ExponentialLR, ReduceLROnPlateau
from ..graphs.datasets import *
from tqdm import tqdm 
from collections import defaultdict
import random
import numpy as np

np.random.seed(2023)
torch.manual_seed(2023)
random.seed(2023)
torch.cuda.manual_seed_all(2023)

def parallel_cpkt_to_single(cpkt):
    state_dict=cpkt["state_dict"]
    new_state_dict={}
    for key in state_dict.keys():
        if "module" in key:
            new_state_dict[key[7:]]=state_dict[key]
        else:
            new_state_dict[key]=state_dict[key]
    return new_state_dict
class Equi_Consistency_Model:
    def __init__(self,**kwargs):
        epochs=kwargs.get('start')
        self.device=GP.device
        loadtype=kwargs.get('loadtype','Perepoch')
        if "modelname" not in kwargs:
            self.mode="train"
            self.modelname='Equi_Consis_Model'
            self.online_model=None
            self.ema_model=None
            self.optim=None
            self.lr_scheduler=None

            self.min_train_loss_batch=1e20
            self.train_batch_loss=0
            self.train_epoch_loss=0
            self.min_valid_loss_batch=1e20
            self.valid_batch_loss=0
            self.valid_epoch_loss=0
            
        else:
            self.mode='test'
            self.modelname=kwargs.get('modelname')

        if not os.path.exists(f'./{self.modelname}/model'):
            os.system(f'mkdir -p ./{self.modelname}/model')

        if self.mode=="train":
            pickle.dump(self.__dict__,open(self.modelname+"/modelsetting.pickle", "wb"))
            self.__build_model()
            self.logger=open(f'./{self.modelname}/Training.log','a')
            self.logger.write('='*40+datetime.now().strftime("%d/%m/%Y %H:%M:%S")+'='*40+'\n') 
            self.logger.flush()
        else:
            self.loadtype=kwargs.get("loadtype")
            self.Load(self.modelname,loadtype)
            self.logger=open(f'./{self.modelname}/Training.log','a')
            now = datetime.now()
            self.logger.write('='*40+now.strftime("%d/%m/%Y %H:%M:%S")+'='*40+'\n')
            self.logger.flush()
        if epochs:
            self.epochs=epochs 
        self.batchsize=GP.batchsize
        return
    
    def __build_model(self):
        if not GP.if_chiral:
            self.online_model = Equiformer_Consistency(  
                                                num_tokens = len(GP.atom_types)+1,
                                                num_edge_tokens=len(GP.bond_types)+1,
                                                edge_dim=4,
                                                num_sigmas=2,
                                                dim = GP.dim,               # dimensions per type, ascending, length must match number of degrees (num_degrees)
                                                dim_head = GP.dim_head,          # dimension per attention head
                                                heads = GP.heads,             # number of attention heads
                                                num_linear_attn_heads = GP.num_linear_att_heads,     # number of global linear attention heads, can see all the neighbors
                                                num_degrees = GP.num_degrees,               # number of degrees
                                                depth = GP.depth,                     # depth of equivariant transformer
                                                attend_self = True,            # attending to self or not
                                                reduce_dim_out = True,         # whether to reduce out to dimension of 1, say for predicting new coordinates for type 1 features
                                                l2_dist_attention = False      # set to False to try out MLP attention
                                                )
        
            self.ema_model = Equiformer_Consistency(  
                                                num_tokens = len(GP.atom_types)+1,
                                                num_edge_tokens=len(GP.bond_types)+1,
                                                edge_dim=4,
                                                num_sigmas=2,
                                                dim = GP.dim,               # dimensions per type, ascending, length must match number of degrees (num_degrees)
                                                dim_head = GP.dim_head,          # dimension per attention head
                                                heads = GP.heads,             # number of attention heads
                                                num_linear_attn_heads = GP.num_linear_att_heads,     # number of global linear attention heads, can see all the neighbors
                                                num_degrees = GP.num_degrees,               # number of degrees
                                                depth = GP.depth,                     # depth of equivariant transformer
                                                attend_self = True,            # attending to self or not
                                                reduce_dim_out = True,         # whether to reduce out to dimension of 1, say for predicting new coordinates for type 1 features
                                                l2_dist_attention = False      # set to False to try out MLP attention
                                                )
        else:
            self.online_model = Equiformer_Consistency(  
                                                num_edge_tokens=len(GP.bond_types)+1,
                                                token_dim=len(GP.atom_types)+len(GP.chiral_types),
                                                edge_dim=4,
                                                num_sigmas=2,
                                                dim = GP.dim,               # dimensions per type, ascending, length must match number of degrees (num_degrees)
                                                dim_head = GP.dim_head,          # dimension per attention head
                                                heads = GP.heads,             # number of attention heads
                                                num_linear_attn_heads = GP.num_linear_att_heads,     # number of global linear attention heads, can see all the neighbors
                                                num_degrees = GP.num_degrees,               # number of degrees
                                                depth = GP.depth,                     # depth of equivariant transformer
                                                attend_self = True,            # attending to self or not
                                                reduce_dim_out = True,         # whether to reduce out to dimension of 1, say for predicting new coordinates for type 1 features
                                                l2_dist_attention = False      # set to False to try out MLP attention
                                                )
            self.ema_model = Equiformer_Consistency(  
                                                num_edge_tokens=len(GP.bond_types)+1,
                                                token_dim=len(GP.atom_types)+len(GP.chiral_types),
                                                edge_dim=4,
                                                num_sigmas=2,
                                                dim = GP.dim,               # dimensions per type, ascending, length must match number of degrees (num_degrees)
                                                dim_head = GP.dim_head,          # dimension per attention head
                                                heads = GP.heads,             # number of attention heads
                                                num_linear_attn_heads = GP.num_linear_att_heads,     # number of global linear attention heads, can see all the neighbors
                                                num_degrees = GP.num_degrees,               # number of degrees
                                                depth = GP.depth,                     # depth of equivariant transformer
                                                attend_self = True,            # attending to self or not
                                                reduce_dim_out = True,         # whether to reduce out to dimension of 1, say for predicting new coordinates for type 1 features
                                                l2_dist_attention = False      # set to False to try out MLP attention
                                                )
        
        
        self.consistency_training=ConsistencyTraining(
            sigma_min=GP.sigma_min,
            sigma_max=GP.sigma_max,
            sigma_data=GP.sigma_data,
            rho=GP.rho,
            initial_timesteps=GP.initial_timesteps,
            final_timesteps=GP.final_timesteps
            )
        self.consistency_sampling_and_editing = ConsistencySamplingAndEditing(
                        sigma_min = GP.sigma_min, # minimum std of noise
                        sigma_data = GP.sigma_data, # std of the data
                        )
        
        if self.device=='cuda':
            self.online_model.cuda()
            self.ema_model.cuda()

    def Save(self):
        self.optim=None,
        self.lr_scheduler=None,
        pickle.dump(self.__dict__,open(self.modelname+"/modelsetting.pickle",'wb'))
        shutil.make_archive(self.modelname,"zip",self.modelname)
        return
     
    def Load(self,modelname,loadtype='perepoch'):

        with tempfile.TemporaryDirectory() as dirpath:
            with zipfile.ZipFile(modelname + ".zip", "r") as zip_ref:
                zip_ref.extractall(dirpath)
            metadata = pickle.load(open(dirpath + "/modelsetting.pickle", "rb"))   
            self.__dict__.update(metadata)
            self.__build_model()

            if loadtype=='Minloss':
                modelcpkt=torch.load(f"{dirpath}/model/online_model_minloss.cpk")
                self.online_model.load_state_dict(modelcpkt["state_dict"])
                modelcpkt=torch.load(f"{dirpath}/model/ema_model_per_epoch.cpk")
                self.ema_model.load_state_dict(modelcpkt["state_dict"])
                self.epochs=modelcpkt['epochs']
                print ("Load model successfully!")

            if loadtype=='Perbatch':
                modelcpkt=torch.load(f"{dirpath}/model/online_model_per_batch.cpk")
                self.online_model.load_state_dict(modelcpkt["state_dict"])
                modelcpkt=torch.load(f"{dirpath}/model/ema_model_per_batch.cpk")
                self.ema_model.load_state_dict(modelcpkt["state_dict"])

            else:
                modelcpkt=torch.load(f"{dirpath}/model/online_model_perepoch.cpk")
                self.online_model.load_state_dict(parallel_cpkt_to_single(modelcpkt))
                modelcpkt=torch.load(f"{dirpath}/model/ema_model_perepoch.cpk")
                self.ema_model.load_state_dict(parallel_cpkt_to_single(modelcpkt))

            if self.device=='cuda':
                self.online_model.cuda()
                self.ema_model.cuda()
        return 

    def IC_Loss(self,pred,target,zmats,gmasks):
        pred_bonddis,pred_angle,pred_dihedral,j1,j2,j3=xyz2ic(pred,zmats)
        target_bonddis,target_angle,target_dihedral,j1,j2,j3=xyz2ic(target,zmats)
        pred_dismat=torch.cdist(pred,pred,compute_mode='donot_use_mm_for_euclid_dist')
        target_dismat=torch.cdist(target,target,compute_mode='donot_use_mm_for_euclid_dist')
        gmasks_2D=gmasks.unsqueeze(-1)*gmasks.unsqueeze(-1).permute(0,2,1)
        loss_angle=F.mse_loss(pred_angle[gmasks],target_angle[gmasks])
        loss_dismat=F.mse_loss(pred_dismat[gmasks_2D],target_dismat[gmasks_2D])
        loss_bonddis=F.mse_loss(pred_bonddis[gmasks],target_bonddis[gmasks])
        dihedral_diff=torch.abs(pred_dihedral[gmasks]-target_dihedral[gmasks])
        dihedral_diff=torch.where(dihedral_diff>math.pi,math.pi*2-dihedral_diff,dihedral_diff)
        loss_dihedral=torch.mean(torch.square(dihedral_diff))
        return loss_dismat,loss_bonddis,loss_angle,loss_dihedral
    
    def Train_Step(self,Datas,step_id):
        feats=Datas["Feats"]
        adjs=Datas["Adjs"]
        coords=Datas["Coords"]
        zmats=Datas["Zmats"]
        gmasks=Datas["Masks"]
        bzids=torch.arange(zmats.shape[0]).view(-1,1).tile((1,zmats.shape[1])).unsqueeze(-1).long()
        zmats=torch.concat((bzids,zmats),axis=-1)

        if self.device=='cuda':
            feats,adjs,coords,zmats,gmasks=feats.cuda(),adjs.cuda(),coords.cuda(),zmats.cuda(),gmasks.cuda()

        self.optim.zero_grad()
        bond,angle,dihedral,*_=xyz2ic(coords,zmats)
        aligned_coords=ic2xyz(bond,angle,dihedral,zmats) 
        predicted,target=self.consistency_training(self.online_model,self.ema_model,feats,adjs,aligned_coords,gmasks,zmats,step_id,GP.final_timesteps)

        if GP.loss_mod == 'IC': 
            loss_dismat,loss_bonddis,loss_angle,loss_dihedral=self.IC_Loss(predicted,target,zmats,gmasks)
            self.lr=self.optim.state_dict()['param_groups'][0]['lr']
            lstr=f'Dismat: {loss_dismat.item():.3E}, Dis: {loss_bonddis.item():.3E}, Angle: {loss_angle.item():.3E},  dihedral: {loss_dihedral:.3E}'
            loss=loss_dismat+loss_angle+loss_bonddis+loss_dihedral
            
        elif GP.loss_mod == 'rmsd':
            loss = self.RMSD_Loss(predicted,target,zmats,gmasks)
            self.lr=self.optim.state_dict()['param_groups'][0]['lr']
            loss_dismat,loss_bonddis,loss_angle,loss_dihedral = torch.tensor(0),torch.tensor(0),torch.tensor(0),torch.tensor(0)
            lstr = f'RMSD: {loss.item():.3E}'
        
        else:
            raise Exception('need loss_mod')
            
        torch.cuda.empty_cache() 

        loss.backward()

        for group in self.optim.param_groups:
            torch.nn.utils.clip_grad_norm_(group['params'], 1.5, 2)
        self.optim.step()

        num_timesteps=timesteps_schedule(step_id,GP.final_timesteps,initial_timesteps=GP.initial_timesteps,final_timesteps=GP.final_timesteps)
        ema_decay_rate = ema_decay_rate_schedule(
                                num_timesteps,
                                initial_ema_decay_rate=0.95,
                                initial_timesteps=2,
                            )
        update_ema_model(self.ema_model,self.online_model,ema_decay_rate)
        return loss.item(),loss_dismat.item(),loss_bonddis.item(),loss_angle.item(),loss_dihedral.item(),lstr
    
    def Evaluate_Step(self,Datas,step_id):
        feats=Datas["Feats"]
        adjs=Datas["Adjs"]
        coords=Datas["Coords"]
        zmats=Datas["Zmats"]
        gmasks=Datas["Masks"]
        bzids=torch.arange(zmats.shape[0]).view(-1,1).tile((1,zmats.shape[1])).unsqueeze(-1).long()
        zmats=torch.concat((bzids,zmats),axis=-1)

        if self.device=='cuda':
            feats,adjs,coords,zmats,masks=feats.cuda(),adjs.cuda(),coords.cuda(),zmats.cuda(),gmasks.cuda()
        bond,angle,dihedral,*_=xyz2ic(coords,zmats)
        aligned_coords=ic2xyz(bond,angle,dihedral,zmats)
        self.optim.zero_grad()
        
        predicted,target=self.consistency_training(self.online_model,self.ema_model,feats,adjs,aligned_coords,gmasks,zmats,step_id,GP.final_timesteps)

        #loss_dismat,loss_bonddis,loss_angle,loss_dihedral=self.IC_Loss(predicted,target,zmats,gmasks)

        if GP.loss_mod == 'IC': 
            loss_dismat,loss_bonddis,loss_angle,loss_dihedral=self.IC_Loss(predicted,target,zmats,gmasks)
            self.lr=self.optim.state_dict()['param_groups'][0]['lr']
            lstr=f'Dismat: {loss_dismat.item():.3E}, Dis: {loss_bonddis.item():.3E}, Angle: {loss_angle.item():.3E},  dihedral: {loss_dihedral:.3E}'
            loss=loss_dismat+loss_angle+loss_bonddis+loss_dihedral
            #total_loss_per_batch+=loss.item()
            
        elif GP.loss_mod == 'rmsd':
            loss = self.RMSD_Loss(predicted,target,zmats,gmasks)
            self.lr=self.optim.state_dict()['param_groups'][0]['lr']
            loss_dismat,loss_bonddis,loss_angle,loss_dihedral = torch.tensor(0),torch.tensor(0),torch.tensor(0),torch.tensor(0)
            lstr = f'RMSD: {loss.item():.3E}'
            
        else:
            raise Exception('need loss_mod')
        
        torch.cuda.empty_cache() 


        loss=loss_dismat+loss_angle+loss_bonddis+loss_dihedral
        return loss.item(),loss_dismat.item(),loss_bonddis.item(),loss_angle.item(),loss_dihedral.item(),lstr

    def Fit(self,Train_MGFiles,Valid_MGFiles,Epochs=100):
        self.optim=Adam(self.online_model.parameters(), lr = 1e-4, betas=(0.5,0.999))
        self.lr_scheduler= ReduceLROnPlateau(
                self.optim, mode='min',
                factor=0.9, patience=GP.lr_patience,
                verbose=True, threshold=0.0001, threshold_mode='rel',
                cooldown=GP.lr_cooldown,
                min_lr=1e-06, eps=1e-06)
        
        for epoch in range(Epochs):
            for Fname in Train_MGFiles:
                print (Fname)
                self.logger.write(f'{Fname}\n')
                with open(Fname,'rb') as f:
                    Train_MGs=pickle.load(f)
                Train_Dataset=MG_Dataset(Train_MGs,name='trainset')
                trainloader=DataLoader(Train_Dataset,batch_size=self.batchsize,shuffle=False,num_workers=GP.n_workers)
                trainbar=tqdm(enumerate(trainloader))
                self.train_epoch_loss=0
                ntrain_batchs=math.ceil(len(Train_MGs)/self.batchsize)
                for bid,Datas in trainbar:
                    train_batch_loss=0

                    if GP.random_step == 1:
                    #for step in range(GP.final_timesteps): #随机步长
                        step = random.choice(range(GP.final_timesteps))
                        step_loss,step_loss_dismat,step_loss_bonddis,step_loss_angle,step_loss_dihedral,step_lstr=self.Train_Step(Datas,step_id=step)
                        #pprint.pprint(Datas[''])
                        if self.local_rank==0 or self.local_rank is None:
                            lstr=f'Training -- Epochs: {epoch} file: {file_id} bid: {bid} step: {step} lr: {self.lr:.3E} '+step_lstr
                            print (lstr)
                            self.logger.write(lstr+'\n')
                            self.logger.flush()
                            train_batch_loss+=step_loss

                    elif GP.random_step == 2: #固定步长,取第一个
                        step = range(GP.final_timesteps)[0]
                        step_loss,step_loss_dismat,step_loss_bonddis,step_loss_angle,step_loss_dihedral,step_lstr=self.Train_Step(Datas,step_id=step)
                        #pprint.pprint(Datas[''])
                        if self.local_rank==0 or self.local_rank is None:
                            lstr=f'Training -- Epochs: {epoch} file: {file_id} bid: {bid} step: {step} lr: {self.lr:.3E} '+step_lstr
                            print (lstr)
                            self.logger.write(lstr+'\n')
                            self.logger.flush()
                            train_batch_loss+=step_loss
                    

                    elif GP.random_step == 3: #随机取几个步长
                        steps = random.sample(range(GP.final_timesteps), 10)
                        for step in steps:
                            step_loss,step_loss_dismat,step_loss_bonddis,step_loss_angle,step_loss_dihedral,step_lstr=self.Train_Step(Datas,step_id=step)
                            #pprint.pprint(Datas[''])
                            if self.local_rank==0 or self.local_rank is None:
                                lstr=f'Training -- Epochs: {epoch} file: {file_id} bid: {bid} step: {step} lr: {self.lr:.3E} '+step_lstr
                                print (lstr)
                                self.logger.write(lstr+'\n')
                                self.logger.flush()
                                train_batch_loss+=step_loss


                    elif GP.random_step == 4: #按一定间隔取步长
                        steps = np.arange(0, GP.final_timesteps, 10)
                        for step in steps:
                            step_loss,step_loss_dismat,step_loss_bonddis,step_loss_angle,step_loss_dihedral,step_lstr=self.Train_Step(Datas,step_id=step)
                            #pprint.pprint(Datas[''])
                            if self.local_rank==0 or self.local_rank is None:
                                lstr=f'Training -- Epochs: {epoch} file: {file_id} bid: {bid} step: {step} lr: {self.lr:.3E} '+step_lstr
                                print (lstr)
                                self.logger.write(lstr+'\n')
                                self.logger.flush()
                                train_batch_loss+=step_loss

                    
                    elif GP.random_step == 5: #其余步长数量，如50,100
                        for step in range(GP.final_timesteps): #这里在每一个批量下面又设置了默认150个步长的循环，我们可以改成一个试试
                            step_loss,step_loss_dismat,step_loss_bonddis,step_loss_angle,step_loss_dihedral,step_lstr=self.Train_Step(Datas,step_id=step)
                            #pprint.pprint(Datas[''])
                            if self.local_rank==0 or self.local_rank is None:
                                lstr=f'Training -- Epochs: {epoch} file: {file_id} bid: {bid} step: {step} lr: {self.lr:.3E} '+step_lstr
                                print (lstr)
                                self.logger.write(lstr+'\n')
                                self.logger.flush()
                                train_batch_loss+=step_loss

                                

                    elif GP.random_step == 0:
                        for step in range(GP.final_timesteps): #这里在每一个批量下面又设置了默认150个步长的循环，我们可以改成一个试试
                            step_loss,step_loss_dismat,step_loss_bonddis,step_loss_angle,step_loss_dihedral,step_lstr=self.Train_Step(Datas,step_id=step)
                            #pprint.pprint(Datas[''])
                            if self.local_rank==0 or self.local_rank is None:
                                lstr=f'Training -- Epochs: {epoch} file: {file_id} bid: {bid} step: {step} lr: {self.lr:.3E} '+step_lstr
                                print (lstr)
                                self.logger.write(lstr+'\n')
                                self.logger.flush()
                                train_batch_loss+=step_loss

                    else:
                        raise Exception('GP.random_step error')



                    self.train_batch_loss=train_batch_loss
                    self.lr_scheduler.step(metrics=self.train_batch_loss)
                    self.train_epoch_loss+=train_batch_loss

                self.logger.write(f'{Fname} Training -- Epochs: trainloss: {self.train_epoch_loss/ntrain_batchs:.3E}')
                self.valid_epoch_loss=0

                for vFname in Valid_MGFiles[:1]:
                    with open(vFname,'rb') as f:
                        Valid_MGs=pickle.load(f)
                        mgs=random.sample(Valid_MGs,200)
                    Valid_Dataset=MG_Dataset(mgs,name='validset')
                    validloader=DataLoader(Valid_Dataset,batch_size=self.batchsize,shuffle=False,num_workers=GP.n_workers)
                    validbar=tqdm(enumerate(validloader)) 
                    nvalid_batchs=math.ceil(len(mgs)/self.batchsize)
                    for vid,vDatas in validbar:
                        valid_batch_loss=0

                        if GP.random_step == 1:
                            step = random.choice(range(GP.final_timesteps))
                            step_loss,step_loss_dismat,step_loss_bonddis,step_loss_angle,step_loss_dihedral,step_lstr=self.Evaluate_Step(vDatas,step_id=step)
                            lstr=f'{vFname} Valid -- Epochs: {epoch} file: {file_id} bid: {vid} step: {step} lr: {self.lr:.3E} '+step_lstr
                            print (lstr)
                            self.logger.write(lstr+'\n')
                            self.logger.flush()
                            valid_batch_loss+=step_loss
                            
                        elif GP.random_step == 2:
                            step = range(GP.final_timesteps)[0]
                            step_loss,step_loss_dismat,step_loss_bonddis,step_loss_angle,step_loss_dihedral,step_lstr=self.Evaluate_Step(vDatas,step_id=step)
                            lstr=f'{vFname} Valid -- Epochs: {epoch} file: {file_id} bid: {vid} step: {step} lr: {self.lr:.3E} '+step_lstr
                            print (lstr)
                            self.logger.write(lstr+'\n')
                            self.logger.flush()
                            valid_batch_loss+=step_loss
                            
                        elif GP.random_step == 3:
                            step = random.sample(range(GP.final_timesteps), 10)
                            for step in steps:
                                step_loss,step_loss_dismat,step_loss_bonddis,step_loss_angle,step_loss_dihedral,step_lstr=self.Evaluate_Step(vDatas,step_id=step)
                                lstr=f'{vFname} Valid -- Epochs: {epoch} file: {file_id} bid: {vid} step: {step} lr: {self.lr:.3E} '+step_lstr
                                print (lstr)
                                self.logger.write(lstr+'\n')
                                self.logger.flush()
                                valid_batch_loss+=step_loss

                        
                        elif GP.random_step == 4: #按一定间隔取步长
                            steps = np.arange(0, GP.final_timesteps, 10)
                            for step in steps:
                                step_loss,step_loss_dismat,step_loss_bonddis,step_loss_angle,step_loss_dihedral,step_lstr=self.Evaluate_Step(vDatas,step_id=step)
                                lstr=f'{vFname} Valid -- Epochs: {epoch} file: {file_id} bid: {vid} step: {step} lr: {self.lr:.3E} '+step_lstr
                                print (lstr)
                                self.logger.write(lstr+'\n')
                                self.logger.flush()
                                valid_batch_loss+=step_loss

                        
                        elif GP.random_step == 5:
                            for step in range(GP.final_timesteps):
                                step_loss,step_loss_dismat,step_loss_bonddis,step_loss_angle,step_loss_dihedral,step_lstr=self.Evaluate_Step(vDatas,step_id=step)
                                lstr=f'{vFname} Valid -- Epochs: {epoch} file: {file_id} bid: {vid} step: {step} lr: {self.lr:.3E} '+step_lstr
                                print (lstr)
                                self.logger.write(lstr+'\n')
                                self.logger.flush()
                                valid_batch_loss+=step_loss 


                        elif GP.random_step == 0:
                            for step in range(GP.final_timesteps):
                                step_loss,step_loss_dismat,step_loss_bonddis,step_loss_angle,step_loss_dihedral,step_lstr=self.Evaluate_Step(vDatas,step_id=step)
                                lstr=f'{vFname} Valid -- Epochs: {epoch} file: {file_id} bid: {vid} step: {step} lr: {self.lr:.3E} '+step_lstr
                                print (lstr)
                                self.logger.write(lstr+'\n')
                                self.logger.flush()
                                valid_batch_loss+=step_loss 
                                
                        else:
                            raise Exception('GP.random_step error')



                        self.valid_batch_loss=valid_batch_loss
                        self.valid_epoch_loss+=valid_batch_loss
                    
                    self.logger.write(f'{Fname} Valid -- Epochs: validloss: {self.valid_epoch_loss/nvalid_batchs:.3E}') 
                if self.valid_epoch_loss<self.min_valid_loss_epoch:
                    self.min_valid_loss_epoch=self.valid_epoch_loss
                    print (f'Save New check point of online model at Epoch:{epoch} for {Fname}')
                    savepath=f'{self.modelname}/model/online_model_minloss.cpk'
                    savedict={'epochs':self.epochs,'lr':self.lr,'lossmin':self.min_valid_loss_epoch,'state_dict':self.online_model.state_dict()}
                    torch.save(savedict,savepath)
                    print (f'Save New check point of ema model at Epoch:{epoch} for {Fname}')
                    savepath=f'{self.modelname}/model/ema_model_minloss.cpk'
                    savedict={'epochs':self.epochs,'lr':self.lr,'lossmin':self.min_valid_loss_epoch,'state_dict':self.ema_model.state_dict()}
                    torch.save(savedict,savepath)

                savepath=f'{self.modelname}/model/online_model_perepoch.cpk'
                savedict={'epochs':self.epochs,'lr':self.lr,'lossmin':self.min_valid_loss_epoch,'state_dict':self.online_model.state_dict()}
                torch.save(savedict,savepath)    
                savepath=f'{self.modelname}/model/ema_model_perepoch.cpk'
                savedict={'epochs':self.epochs,'lr':self.lr,'lossmin':self.min_valid_loss_epoch,'state_dict':self.ema_model.state_dict()}
                torch.save(savedict,savepath)  
        return 
    def sample_batch(self,Datas):
        feats=Datas["Feats"]
        adjs=Datas["Adjs"]
        coords=Datas["Coords"]
        zmats=Datas["Zmats"]
        gmasks=Datas["Masks"]
        bzids=torch.arange(zmats.shape[0]).view(-1,1).tile((1,zmats.shape[1])).unsqueeze(-1).long()
        zmats=torch.concat((bzids,zmats),axis=-1)

        if self.device=='cuda':
            feats,adjs,coords,zmats,gmasks=feats.cuda(),adjs.cuda(),coords.cuda(),zmats.cuda(),gmasks.cuda()
        with torch.no_grad():
            sigmas = karras_schedule(
                GP.final_timesteps, GP.sigma_min, GP.sigma_max, GP.rho, coords.device
            )

            if GP.final_timesteps == 1:
                sigmas= reversed(sigmas)
                sigmas[-1] = 80
            else:
                #sigmas= reversed(sigmas)[:-1] #第一步不用，这个不能用，直接报错了，索引溢出, 此时需要每一步前加1，保证第一步
                sigmas= reversed(sigmas)
                sigmas[-1] += 1e-8  #修改了这里
            print('sigmas:', sigmas)
            
            samples = self.consistency_sampling_and_editing(
                                    self.online_model,
                                    feats=feats,
                                    adjs=adjs,
                                    y=torch.randn(coords.shape).cuda(), # used to infer the shapes
                                    gmasks=gmasks,
                                    sigmas=sigmas, # sampling starts at the maximum std (T)
                                    clip_denoised=False, # whether to clamp values to [-1, 1] range
                                    verbose=True,
                                )
            if GP.multi_step == 1:
                aligned_samples = {}
                for stp in list(samples.keys()):
                    bond,angle,dihedral,*_=xyz2ic(samples[stp],zmats)
                    aligned_samples[stp] = ic2xyz(bond,angle,dihedral,zmats)
            else:
                bond,angle,dihedral,*_=xyz2ic(samples,zmats)
                aligned_samples=ic2xyz(bond,angle,dihedral,zmats)
            

        return aligned_samples 
    
    def Sample(self,MGs,conf_num_per_mol=10,savepath='./mol.sdf'):
        self.online_model.eval()
        self.ema_model.eval()
        Final_MGs=[]
        
        for MG in MGs: #遍历每一个构象
            Final_MGs+=[MG]*conf_num_per_mol #复制构象2份
        Dataset=MG_Dataset(Final_MGs,name='sample')
        print (self.batchsize)
        Loader=DataLoader(Dataset,batch_size=self.batchsize,shuffle=False,num_workers=GP.n_workers)
        #bar=tqdm(enumerate(Loader)) 
        bar=enumerate(Loader)
        
        if GP.multi_step == 1:
            mols_dict = defaultdict(list)
            total_samples=defaultdict(list)
            for bid,Datas in bar:
                samples=self.sample_batch(Datas) #字典
                #print (samples.unsqueeze(0).shape)
                for stp in list(samples.keys()):
                    total_samples[stp].append(samples[stp])
            for stp in list(total_samples.keys()):   
                total_samples[stp]=torch.concat(total_samples[stp],axis=0) #连接所有采样的构象，其实返回的是构象的原子坐标
                
            #molsupp=Chem.SDWriter(savepath)
            mols_dict = {}
            for stp in list(total_samples.keys()):
                mols=[]
                for i in range(len(Final_MGs)):    #处理每一个构象
                    mg=copy.deepcopy(Final_MGs[i]) #复制真实的构象
                    sampled_coords=total_samples[stp][i].clone().detach().cpu().numpy()[:mg.natoms] #采样出来的构象坐标
                    mg.Update_Coords(sampled_coords) #更新真实构象坐标
                    try:
                        mol=mg.Trans_to_Rdkitmol() #转化成rdkit mol对象
                        mols.append(mol)
                        #molsupp.write(mol)          #这里的写没啥意义，只会保存最后一个分子的mol
                    except:
                        pass
                #molsupp.close()
                mols_dict[stp] = mols
            return mols_dict
    
        else:
            mols=[]
            total_samples=[]
            for bid,Datas in bar:
                samples=self.sample_batch(Datas)
                #print (samples.unsqueeze(0).shape)
                total_samples.append(samples)
            total_samples=torch.concat(total_samples,axis=0) #连接所有采样的构象，其实返回的是构象的原子坐标
            #molsupp=Chem.SDWriter(savepath)
            for i in range(len(Final_MGs)):    #处理每一个构象
                mg=copy.deepcopy(Final_MGs[i]) #复制真实的构象
                sampled_coords=total_samples[i].clone().detach().cpu().numpy()[:mg.natoms] #采样出来的构象坐标
                mg.Update_Coords(sampled_coords) #更新真实构象坐标
                try:
                    mol=mg.Trans_to_Rdkitmol() #转化成rdkit mol对象
                    mols.append(mol)
                    #molsupp.write(mol)          #这里的写没啥意义，只会保存最后一个分子的mol
                except:
                    pass
            #molsupp.close()
            return mols
    
    

            



        







                        
            

        
    

            
            
