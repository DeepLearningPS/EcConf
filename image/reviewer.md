# Reviewer 1:

## Q1: 
For a fair comparison between EC-Conf and other methods, I think the underlying score network should remain intact, such that we can pinpoint improved performance to the introduction of consistency training.

## Response:
We have tried to employ consistency training to Geo-Diff, but we didn’t get meaningful results yet due to the training time is not enough. Currently, the result of COV-R and MAT-R are 0 and 5.6536 angstrom on QM9 test set with 25 sampling steps. And the time is also not enough to train a DDPM-based diffusion model with Equiformer for comparison. 


## Q2: 
Is the number of parameters in Equiformer at a similar scale to other baseline models?

## Response:
The parameter number of Equiformer is 3.03M, while it is 0.80 and 0.87 for Geo-Diff and SDE-Gen respectively. 


## Q3: 
In addition, I would like to know the difference between training a consistency model in isolation vs distillation from a pretrained diffusion model. Adding such experiments would nicely complement this work.

## Response:
The review time is not enough for consistency distillation on a given diffusion model such as Geo-Diff too. 


## Q4: 
In the original consistency model paper, the number of training steps (N) has an incremental schedule. Would that also benefit training EC-Conf?

## Response: 
The incremental schedule of training steps is crucial for accommodating a changeable sampling step in EC-Conf, as illustrated in Algorithm 1, denoted as N(∙). Since different maximum sampling steps correspond to distinct time point sequences on the ODE trajectory, the incremental schedule enhances model transferability by learning to map more time points to the ground truth during the training phase. However, this incremental schedule can substantially increase training costs. Therefore, we need to carefully balance the maximum training steps in the incremental schedule with computational costs. Moreover, the maximum training steps in the incremental schedule significantly affect performance. For instance, increasing the maximum training steps to 150 results in a decrease in the mean value of COV-R to 0.5033 and an increase in MAT-R to 0.5138 on the QM9 test set, which is weaker compared to the results obtained with a maximum training steps setting of 25.

## Finally, we provide some links about model parameters, time complexity, and diffusion process (https://github.com/DeepLearningPS/EcConf/tree/main/image).






# Reviewer 2: 

## Response: 
Thanks for your comment. However, we think comparing with Torsional Diffusion model is not fair in this case. Torsional Diffusion is actually a mix of deep learning on torsional angles and force field related knowledge on bond length and bond angle, which, of course, looks better than methods not relying on existing force field knowledge. In our study, we focused on comparing with diffusion methods which learn from molecular conformation data only and doesn’t rely on force field knowledge at all. In this case, our method clearly demonstrates its advantage in sample efficiency. Regarding to other minor things such as computation time, we provide the wall time of generating a conformation of three different methods here for reviewing. The wall time of EC-Conf for single-step diffusion is 0.17s, which is comparable to ETKDG methods with MMFF94 optimizations. Even with 25 steps of diffusion, the wall time is 4.6s, faster than GeoDiff and SDEGen, which are 46.0 and 8.9s, respectively. Despite EC-Conf having nearly four times the number of parameters compared to Geo-Diff and SDE-Gen, leading to longer computation times per step, its acceleration in diffusion steps still renders it more efficient than other methods.

## Finally, we provide some links about model parameters, time complexity, and diffusion process (https://github.com/DeepLearningPS/EcConf/tree/main/image).






# Reviewer 3:

## Q1:
Could the author compare EC-Conf to diffusion-based models using the same number of steps (less than 30), in order to demonstrate the efficiency of EC-Conf over conventional diffusion models fairly.

## Response: 
To our best knowledge, the only existing published molecular conformation diffusion models on cartesian coordinate systems are Geo-Diff and SDE-Gen. Although Torsion-Diff only needs less than 30 diffusion steps for drug-like conformation generation, it mixed the force field related knowledge on bond length and angles. It is not fair for comparison since its freedom is only 1/3 compared to diffusion models on cartesian coordinates. 
Here, we compared the quality metric of intermedia states of EC-Conf, Geo-Diff and SDE-Gen in diffusion process. SDE-Gen and Geo-Diff can’t generate reasonable conformations with fewer than 30 diffusion steps. The intermedia state quality of EC-Conf with 1 step surpass 3000 iterations with Geo-Diff and 5000 iterations with SDE-Gen, clearly demonstrate the efficiency of EC-Conf. 

### We provide the link about diffusion process table (https://github.com/DeepLearningPS/EcConf/blob/main/image/table1.png).



## Q2:
Could the author present the sampling trajectory of EC-Conf and some diffusion-based models to demonstrate the sampling efficiency more intuitively? (Figure 1 provides some trajectories but it seems only to be a schematic illustration)

## Response:
The conformation evolution in diffusion process is as shown in Figure 1. Obviously, the conformations converge faster in EC-Conf than Geo-Diff and SDE-Gen.

### We provide the link about diffusion process figure (https://github.com/DeepLearningPS/EcConf/blob/main/image/fig1.png).



## Q3:
What is the wall time for generating conformations? As the major strength of EC-Conf is its efficiency, it would be better if the author compares the sampling time of EC-Conf with other methods, especially traditional empirical methods such as RDKit (which is lightweight and does not require a lot of computation as deep learning models do).

## Response:
Here, we compared the wall time of generating a conformation of three different methods. The wall time of EC-Conf for single-step diffusion is 0.17s, which is comparable to ETKDG methods with MMFF94 optimizations. Even with 25 steps of diffusion, the wall time is 4.6s, faster than GeoDiff and SDEGen, which are 46.0 and 8.9s, respectively. Despite EC-Conf having nearly four times the number of parameters compared to Geo-Diff and SDE-Gen, leading to longer computation times per step, its acceleration in diffusion steps still renders it more efficient than other methods.

### We provide the link about model parameters, time complexity (https://github.com/DeepLearningPS/EcConf/blob/main/image/table2.png).







