import torch
from libcpab.pytorch import cpab
from libcpab.pytorch.transformer_old import CPAB_transformer
from libcpab.pytorch.torch_funcs import torch_findcellidx_2D, torch_findcellidx_3D
import time

T = cpab(tess_size=[2,2,2], device='gpu')
g = T.uniform_meshgrid([200, 200, 200])
theta = 0.1*T.sample_transformation(1)
gt = T.transform_grid(g, theta)

#T2 = cpab(tess_size=[2,2])
#g2 = T2.uniform_meshgrid([3, 3])
#theta2 = theta.to(T2.device)
#gt2 = T2.transform_grid(g2, theta2)

#print(gt)
#print(gt2)
#print(((gt.cpu() - gt2).abs().sum() / gt2.numel()).numpy())


#T = cpab(tess_size=[2,2,2])
#g = T.uniform_meshgrid([100, 100, 100])
#theta = 0.1*T.sample_transformation(2)
#start = time.time()
#gt = T.transform_grid(g, theta)
#time1 = time.time() - start 
#old = CPAB_transformer(params = T.params, findcellidx_func=torch_findcellidx_3D,
#                       device=T.device)
#start = time.time()
#gt2 = old(g, theta)
#time2 = time.time() - start
#print(gt)
#print(gt2)
#print(((gt-gt2).abs().sum() / gt.numel()).numpy())
#print(time1, time2)
