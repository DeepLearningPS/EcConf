# EC-Conf: An Ultra-fast Diffusion Model for Molecular conformation Generation with Equivariant Consistency





## Environments

### Install env

```bash
#Create the environment
conda create -n name ecconf python=3.8

#Activate the environment
conda activate ecconf

#Install torch. CUDA 11.3 usually need the gcc/9.3.0
#You'll need to choose between torch and cuda depending on your device. 
Pytorch is avaiable [[here]](https://pytorch.org/get-started/previous-versions/)
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch

#Install pytorch_geometric(PyG). You'll need to choose PyG depending on your Pytroch version. 
PyG is avaiable [[here]](https://data.pyg.org/whl/).  If the online installation fails, you need to manually go to the website to download the files you need.
pip install torch_scatter==2.1.0 torch_sparse==0.6.16 torch_cluster==1.6.1 torch_spline_conv==1.2.2 torch_geometric==1.7.2 -f https://data.pyg.org/whl/torch-1.12.1+cu113.html

#If the "torch_geometric" version is too old when running code, you need to install a newer version. For example:
pip install torch_geometric==2.1.0

#Install rdkit einops filelock
pip install rdkit einops filelock beartype

#Install local package EcConf (setup.py)
cd ./EcConf
pip install -e ./

```


## Dataset

### Offical Dataset
The offical raw GEOM dataset is avaiable [[here]](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/JNGTDF).

### Preprocessed dataset
Our dataset comes directly from GeoDiff in this [[google drive folder]](https://drive.google.com/drive/folders/1b0kNBtck9VNrLRZxg6mckyVUpJA5rBHh?usp=sharing) and is transformed into the data structure required by EcConf with some simple code. 


```bash

#After obtaining the original GeoDiff dataset, you need to do the next steps.
Copy the downloaded GeoDiff dataset into ./scripts/GEMO/Drugs/datasets/Origin_GeoDiff_Drugs/ and ./scripts/GEMO/QM9/datasets/Origin_GeoDiff_QM9/.
cd ./scripts

#For QM9, the data in PyG format is converted to the data required by EcConf.
python DataProcess/qm9/PyG2EcConf_dataStyle/dataset_geodiff.py
cd GEMO/QM9/datasets
python predata.py

#For Drugs, the data in PyG format is converted to the data required by EcConf.
python DataProcess/drugs/PyG2EcConf_dataStyle/dataset_geodiff.py
cd GEMO/Drugs/datasets
python predata.py

```

### Prepare your own GEOM dataset from scratch (optional)

You can also download origianl GEOM full dataset and prepare your own data split. A guide is available at previous work ConfGF's [[github page]](https://github.com/DeepGraphLearning/ConfGF#prepare-your-own-geom-dataset-from-scratch-optional).

This step allows you to build datasets of any size where the built data structure is PyG. Then you just need to run the code we preprocessed the data from the previous step.


## Training
```bash
# For QM9 dataset
cd ./scripts/QM9_25_step/
# Multi-GPU parallel
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --rdzv_backend c10d --rdzv_endpoint localhost:0 --nnodes 1 --nproc_per_node 4 --rdzv_id 4 parallel.py
# Single-GPU
CUDA_VISIBLE_DEVICES=0 torchrun --rdzv_backend c10d --rdzv_endpoint localhost:0 --nnodes 1 --nproc_per_node 1 --rdzv_id 1 parallel.py


# For Drugs dataset
cd ./scripts/Drugs_25_step/
# Multi-GPU parallel
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --rdzv_backend c10d --rdzv_endpoint localhost:0 --nnodes 1 --nproc_per_node 4 --rdzv_id 4 parallel.py
# Single-GPU
CUDA_VISIBLE_DEVICES=0 torchrun --rdzv_backend c10d --rdzv_endpoint localhost:0 --nnodes 1 --nproc_per_node 1 --rdzv_id 1 parallel.py
```


## Generation and Evaluating
```bash
cd ./scripts/QM9_25_step/
##For QM9 dataset
#Package the trained model.
python zip.py
#single-step generation. If the step size is 25, only the diffusion result of step 25 is saved.
CUDA_VISIBLE_DEVICES=0 python singlestep_sample.py
# multi-step generation. If the step size is 25, the diffusion intermediate results of 1 to 25 steps are saved.
CUDA_VISIBLE_DEVICES=0 python multistep_sample.py


cd ./scripts/Drugs_25_step/
##For Drugs dataset
#Package the trained model.
python zip.py
#single-step generation. If the step size is 25, only the diffusion result of step 25 is saved.
CUDA_VISIBLE_DEVICES=0 python singlestep_sample.py
# multi-step generation. If the step size is 25, the diffusion intermediate results of 1 to 25 steps are saved.
CUDA_VISIBLE_DEVICES=0 python multistep_sample.py
```


## Load the pre-trained model
```bash
We also provide trained models in the scripts/Pretrain/QM9_25_step/Equi_Consis_Model.zip, and scripts/Pretrain/Drugs_25_step/Equi_Consis_Model.zip
You need to copy them into scripts/QM9_25_step and scripts/Drugs_25_step respectively.


cd ./scripts/QM9_25_step/
##For QM9 dataset
#single-step generation. If the step size is 25, only the diffusion result of step 25 is saved.
CUDA_VISIBLE_DEVICES=0 python singlestep_sample.py
# multi-step generation. If the step size is 25, the diffusion intermediate results of 1 to 25 steps are saved.
CUDA_VISIBLE_DEVICES=0 python multistep_sample.py


cd ./scripts/Drugs_25_step/
##For Drugs dataset
#single-step generation. If the step size is 25, only the diffusion result of step 25 is saved.
CUDA_VISIBLE_DEVICES=0 python singlestep_sample.py
# multi-step generation. If the step size is 25, the diffusion intermediate results of 1 to 25 steps are saved.
CUDA_VISIBLE_DEVICES=0 python multistep_sample.py
```


## Citation
Please consider citing the our paper if you find it helpful. Thank you!




## Contact

