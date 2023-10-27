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

from torch import distributed as dist

#半精度、混合精度
from torch.cuda.amp import GradScaler, autocast
scaler = GradScaler()



class Equi_Consistency_Model_Parallel:
    def __init__(self,local_rank=None,**kwargs):

        epochs=kwargs.get('start')
        self.local_rank=local_rank
        
        self.device=GP.device
        
        if "modelname" not in kwargs:

            self.mode="train"
            self.modelname='Equi_Consis_Model'
            self.online_model=None
            self.ema_model=None
            self.optim=None
            self.lr_scheduler=None
            self.min_train_loss_batch=1e20
            self.min_train_loss_epoch=1e20
            self.train_batch_loss=0
            self.train_epoch_loss=0
            self.min_valid_loss_batch=1e20
            self.min_valid_loss_epoch=1e20
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
            #self.logger=open(f'./{self.modelname}/Training.log','a')
            self.logger=open(f'./{self.modelname}/Training.log','w')
            self.logger.write('='*40+datetime.now().strftime("%d/%m/%Y %H:%M:%S")+'='*40+'\n') 
            self.logger.flush()

        else:
            self.loadtype=kwargs.get("loadtype")
            self.Load(self.modelname,self.loadtype)
            #self.logger=open(f'./{self.modelname}/Training.log','a')
            self.logger=open(f'./{self.modelname}/Training.log','w')
            now = datetime.now()
            self.logger.write('='*40+now.strftime("%d/%m/%Y %H:%M:%S")+'='*40+'\n')
            self.logger.flush()
        if epochs:
            self.epochs=epochs 
        else:
            self.epochs=0
        self.batchsize=GP.batchsize
        return
    
    def __build_model(self):

        if self.local_rank is not None:
            dist.init_process_group(backend="nccl")
            torch.cuda.set_device(self.local_rank)
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
            if self.local_rank is not None:
                self.online_model.cuda(self.local_rank)
                self.ema_model.cuda(self.local_rank)
                self.online_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.online_model)
                self.ema_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.ema_model)

                self.online_model = torch.nn.parallel.DistributedDataParallel(self.online_model,
                                                      device_ids=[self.local_rank],
                                                      output_device=self.local_rank,
                                                      find_unused_parameters=True,
                                                      broadcast_buffers=False)

                self.ema_model = torch.nn.parallel.DistributedDataParallel(self.ema_model,
                                                      device_ids=[self.local_rank],
                                                      output_device=self.local_rank,
                                                      find_unused_parameters=True,
                                                      broadcast_buffers=False)
            else:
                self.online_model.cuda()
                self.ema_model.cuda()

    def Save(self):

        self.optim=None,
        self.lr_scheduler=None,
        pickle.dump(self.__dict__,open(self.modelname+"/modelsetting.pickle",'wb'))
        shutil.make_archive(self.modelname,"zip",self.modelname)
        return
     
    def Load(self,modelname,loadtype='Perepoch'):

        with tempfile.TemporaryDirectory() as dirpath:
            with zipfile.ZipFile(modelname + ".zip", "r") as zip_ref:
                zip_ref.extractall(dirpath)
            metadata = pickle.load(open(dirpath + "/modelsetting.pickle", "rb"))
            local_rank = self.local_rank
            self.__dict__.update(metadata)
            
            self.local_rank = local_rank
            
            self.__build_model()

            if loadtype=='Minloss':
                modelcpkt=torch.load(f"{dirpath}/model/online_model_minloss.cpk")
                self.online_model.load_state_dict(modelcpkt["state_dict"])
                modelcpkt=torch.load(f"{dirpath}/model/ema_model_minloss.cpk")
                self.ema_model.load_state_dict(modelcpkt["state_dict"])
                self.epochs=modelcpkt['epochs']
                print ("Load model successfully!")

            if loadtype=='Per_batch':
                modelcpkt=torch.load(f"{dirpath}/model/online_model_per_batch.cpk")
                self.online_model.load_state_dict(modelcpkt["state_dict"])
                modelcpkt=torch.load(f"{dirpath}/model/ema_model_per_batch.cpk")
                self.ema_model.load_state_dict(modelcpkt["state_dict"])

            else:
                modelcpkt=torch.load(f"{dirpath}/model/online_model_perepoch.cpk")
                self.online_model.load_state_dict(modelcpkt["state_dict"])
                modelcpkt=torch.load(f"{dirpath}/model/ema_model_perepoch.cpk")
                self.ema_model.load_state_dict(modelcpkt["state_dict"])

            if self.device=='cuda':
                self.online_model.cuda()
                self.ema_model.cuda()
        return 

    def IC_Loss(self,pred,target,zmats,gmasks):
        pred_bonddis,pred_angle,pred_dihedral,j1,j2,j3=xyz2ic(pred,zmats) #笛卡尔转内坐标
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
    

    def RMSD_Loss(self,pred,target,zmats,gmasks):
        #pred
        bond,angle,dihedral,*_ = xyz2ic(pred,zmats)   
        pred_aligned = ic2xyz(bond,angle,dihedral,zmats) #将笛卡尔坐标与内坐标对齐？
        
        #target
        bond,angle,dihedral,*_ = xyz2ic(target,zmats)
        target_aligned = ic2xyz(bond,angle,dihedral,zmats) #将笛卡尔坐标与内坐标对齐？
        
         
        loss = (pred_aligned - target_aligned)**2
        loss = torch.sum(loss, dim=-1, keepdim=True).mean()
        
        return loss
        
        '''
        pred_bonddis,pred_angle,pred_dihedral,j1,j2,j3=xyz2ic(pred,zmats) #笛卡尔转内坐标
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
        '''
    
    def Train_Step(self,Datas,step_id):
        self.online_model.train()
        self.ema_model.train()
        feats=Datas["Feats"]
        adjs=Datas["Adjs"]
        coords=Datas["Coords"]
        zmats=Datas["Zmats"]
        gmasks=Datas["Masks"]
        bzids=torch.arange(zmats.shape[0]).view(-1,1).tile((1,zmats.shape[1])).unsqueeze(-1).long()
        zmats=torch.concat((bzids,zmats),axis=-1)

        if self.device=='cuda':
            if self.local_rank is not None:
                feats,adjs,coords,zmats,gmasks=feats.cuda(self.local_rank),adjs.cuda(self.local_rank),coords.cuda(self.local_rank),zmats.cuda(self.local_rank),gmasks.cuda(self.local_rank)
            else:
                feats,adjs,coords,zmats,gmasks=feats.cuda(),adjs.cuda(),coords.cuda(),zmats.cuda(),gmasks.cuda()

        self.optim.zero_grad()
        bond,angle,dihedral,*_=xyz2ic(coords,zmats)
        aligned_coords=ic2xyz(bond,angle,dihedral,zmats) #将笛卡尔坐标与内坐标对齐？
        
        #混合精度
        with autocast():
            predicted,target=self.consistency_training(self.online_model,self.ema_model,feats,adjs,aligned_coords,gmasks,zmats,step_id,GP.final_timesteps)
            if GP.loss_mod == 'IC': 
                loss_dismat,loss_bonddis,loss_angle,loss_dihedral=self.IC_Loss(predicted,target,zmats,gmasks)
                self.lr=self.optim.state_dict()['param_groups'][0]['lr']
                lstr=f'Dismat: {loss_dismat.item():.3E}, Dis: {loss_bonddis.item():.3E}, Angle: {loss_angle.item():.3E},  dihedral: {loss_dihedral:.3E}'
                loss=loss_dismat+loss_angle+loss_bonddis+loss_dihedral
                #total_loss_per_batch+=loss.item()
                #scaler.scale(loss).backward()
            else:
                loss = self.RMSD_Loss(predicted,target,zmats,gmasks)
                self.lr=self.optim.state_dict()['param_groups'][0]['lr']
                loss_dismat,loss_bonddis,loss_angle,loss_dihedral = torch.tensor(0),torch.tensor(0),torch.tensor(0),torch.tensor(0)
                lstr = f'RMSD: {loss.item():.3E}'
                #scaler.scale(loss).backward()
                
                

        for group in self.optim.param_groups:
            torch.nn.utils.clip_grad_norm_(group['params'], 1.5, 2)
            
        #self.optim.step() #使用混合精度的时候去掉
        
        #半精度
        scaler.scale(loss).backward()
        scaler.step(self.optim)
        scaler.update()

        num_timesteps=timesteps_schedule(step_id,GP.final_timesteps,initial_timesteps=GP.initial_timesteps,final_timesteps=GP.final_timesteps)
        ema_decay_rate = ema_decay_rate_schedule(
                                num_timesteps,
                                initial_ema_decay_rate=0.95,
                                initial_timesteps=2,
                            )
        update_ema_model(self.ema_model,self.online_model,ema_decay_rate)
        
        if GP.loss_mod == 'IC':
            return loss.item(),loss_dismat.item(),loss_bonddis.item(),loss_angle.item(),loss_dihedral.item(),lstr
        else:
            return loss.item(),loss_dismat.item(),loss_bonddis.item(),loss_angle.item(),loss_dihedral.item(),lstr
    
    def Evaluate_Step(self,Datas,step_id):
        self.online_model.eval()
        self.ema_model.eval()
        
        with torch.no_grad():
            feats=Datas["Feats"]
            adjs=Datas["Adjs"]
            coords=Datas["Coords"]
            zmats=Datas["Zmats"]
            gmasks=Datas["Masks"]
            bzids=torch.arange(zmats.shape[0]).view(-1,1).tile((1,zmats.shape[1])).unsqueeze(-1).long()
            zmats=torch.concat((bzids,zmats),axis=-1)

            if self.device=='cuda':
                if self.local_rank is not None:
                    feats,adjs,coords,zmats,gmasks=feats.cuda(self.local_rank),adjs.cuda(self.local_rank),coords.cuda(self.local_rank),zmats.cuda(self.local_rank),gmasks.cuda(self.local_rank)
                else:
                    feats,adjs,coords,zmats,gmasks=feats.cuda(),adjs.cuda(),coords.cuda(),zmats.cuda(),gmasks.cuda()

            bond,angle,dihedral,*_=xyz2ic(coords,zmats)
            aligned_coords=ic2xyz(bond,angle,dihedral,zmats)
            self.optim.zero_grad()
            
            #混合精度
            with autocast():
                predicted,target=self.consistency_training(self.online_model,self.ema_model,feats,adjs,aligned_coords,gmasks,zmats,step_id,GP.final_timesteps)
                #loss_dismat,loss_bonddis,loss_angle,loss_dihedral=self.IC_Loss(predicted,target,zmats,gmasks)
                if GP.loss_mod == 'IC': 
                    loss_dismat,loss_bonddis,loss_angle,loss_dihedral=self.IC_Loss(predicted,target,zmats,gmasks)
                    self.lr=self.optim.state_dict()['param_groups'][0]['lr']
                    lstr=f'Dismat: {loss_dismat.item():.3E}, Dis: {loss_bonddis.item():.3E}, Angle: {loss_angle.item():.3E},  dihedral: {loss_dihedral:.3E}'
                    loss=loss_dismat+loss_angle+loss_bonddis+loss_dihedral
                    #total_loss_per_batch+=loss.item()
                    
                else:
                    loss = self.RMSD_Loss(predicted,target,zmats,gmasks)
                    self.lr=self.optim.state_dict()['param_groups'][0]['lr']
                    loss_dismat,loss_bonddis,loss_angle,loss_dihedral = torch.tensor(0),torch.tensor(0),torch.tensor(0),torch.tensor(0)
                    lstr = f'RMSD: {loss.item():.3E}'
                
            
            #lstr=f'Dismat: {loss_dismat.item():.3E}, Dis: {loss_bonddis.item():.3E}, Angle: {loss_angle.item():.3E},  dihedral: {loss_dihedral:.3E}'
            loss=loss_dismat+loss_angle+loss_bonddis+loss_dihedral
            
            if GP.loss_mod == 'IC':
                return loss.item(),loss_dismat.item(),loss_bonddis.item(),loss_angle.item(),loss_dihedral.item(),lstr
            else:
                return loss.item(),loss_dismat.item(),loss_bonddis.item(),loss_angle.item(),loss_dihedral.item(),lstr


    def Fit(self,Train_MGFiles,Valid_MGFiles,Epochs=1000000):
        print ('Here')
        self.optim=Adam(self.online_model.parameters(), lr = 1e-3, betas=(0.5,0.999))
        self.lr_scheduler= ReduceLROnPlateau(
                self.optim, mode='min',
                factor=0.9, patience=GP.lr_patience,
                verbose=True, threshold=0.0001, threshold_mode='rel',
                cooldown=GP.lr_cooldown,
                min_lr=1e-06, eps=1e-06)
        
        for epoch in range(Epochs):
            for file_id, Fname in enumerate(Train_MGFiles):
                print (Fname)
                self.logger.write(f'{Fname}\n')
                with open(Fname,'rb') as f:
                    Train_MGs=pickle.load(f)
                Train_Dataset=MG_Dataset(Train_MGs,name='trainset')
                if self.local_rank is not None:
                    Train_Sampler=torch.utils.data.distributed.DistributedSampler(Train_Dataset)
                    Trainloader=DataLoader(Train_Dataset,batch_size=self.batchsize,shuffle=False,num_workers=GP.n_workers,sampler=Train_Sampler)
                    Train_Sampler.set_epoch(epoch)
                    print ('dataset sampler is done')
                else:
                    Trainloader=DataLoader(Train_Dataset,batch_size=self.batchsize,shuffle=True,num_workers=GP.n_workers,)

                trainbar=tqdm(enumerate(Trainloader))
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

            if self.local_rank==0 or self.local_rank is None:
                self.logger.write(f'{Fname} Training -- Epochs: trainloss: {self.train_epoch_loss/ntrain_batchs:.3E}')
                    
                    
            if (self.local_rank is None) or (self.local_rank==0):
                self.valid_epoch_loss=0 
                for file_id, vFname in enumerate(Valid_MGFiles[:1]):
                    with open(vFname,'rb') as f:
                        Valid_MGs=pickle.load(f)
                        mgs=random.sample(Valid_MGs,self.batchsize*10)
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
                self.epochs+=1
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
            if self.local_rank is not None:
                dist.barrier() 
                 
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
            feats,adjs,coords,zmats,masks=feats.cuda(),adjs.cuda(),coords.cuda(),zmats.cuda(),gmasks.cuda()
        with torch.no_grad():
            sigmas = karras_schedule(
                GP.final_timesteps, GP.sigma_min, GP.sigma_max, GP.rho, coords.device
            )
            sigmas= reversed(sigmas)[:-1]
            print (sigmas)
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
            bond,angle,dihedral,*_=xyz2ic(samples,zmats)
            aligned_samples=ic2xyz(bond,angle,dihedral,zmats)

        return aligned_samples 
    
    def Sample(self,MGs,conf_num_per_mol=10,savepath='./mol.sdf'):
        Final_MGs=[]
        for MG in MGs:
            Final_MGs+=[MG]*conf_num_per_mol
        Dataset=MG_Dataset(Final_MGs,name='sample')
        print (self.batchsize)
        Loader=DataLoader(Dataset,batch_size=self.batchsize,shuffle=False,num_workers=GP.n_workers)
        bar=tqdm(enumerate(Loader)) 
        total_samples=[]
        for bid,Datas in bar:
            samples=self.sample_batch(Datas)
            total_samples.append(samples)
        total_samples=torch.concat(total_samples,axis=0)
        molsupp=Chem.SDWriter(savepath)
        for i in range(len(Final_MGs)):
            mg=copy.deepcopy(Final_MGs[i])
            sampled_coords=total_samples[i].clone().detach().cpu().numpy()[:mg.natoms]
            mg.Update_Coords(sampled_coords)
            try:
                mol=mg.Trans_to_Rdkitmol()
                molsupp.write(mol)
            except:
                pass
        molsupp.close()
        return 

            



        







                        
            

        
    

            
            
