# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 15:40:59 2018

@author: nsde
"""

if __name__ == "__main__":
    from libcpab import cpab
    import torch
    import matplotlib.pyplot as plt
    import numpy as np
    import pickle
    from BrainFuns import LoadBrain, PlotGuiImg
    # 1D
#    data1 = torch.tensor(np.sin(np.linspace(0,1,100)*5)[None,None,:]).to(torch.float32).cuda()
#    data2 = torch.tensor(np.cos(np.linspace(0,1,100)*5)[None,None,:]).to(torch.float32).cuda()
#    data = torch.cat((data1, data2), dim=1)
#    T1 = cpab([10,], backend="pytorch", device='gpu')
#    B1 = T1.get_basis()
#    P1 = T1.get_params()
#    d1 = T1.get_theta_dim()
#    g1 = T1.uniform_meshgrid([100])
#    t1 = T1.identity()
#    s1 = T1.sample_transformation()
#    s1 = T1.sample_transformation(1, torch.zeros(d1).cuda(), torch.eye(d1).cuda())
#    v1 = T1.calc_vectorfield(g1, s1)
#    n1 = T1.transform_grid(g1, s1)
#    D1 = T1.transform_data(data, s1, outsize=(*data.shape[2:],))
#    vec_plot = T1.visualize_vectorfield(s1, nb_points=20)
#    plt.figure()
#    plt.plot(data[0,0,:].cpu().numpy())
#    plt.figure()
#    plt.plot(D1[0,0,:].cpu().numpy())    
    
#    # 2D
#    data = torch.tensor(plt.imread("cat.jpg")[None]).permute(0,3,1,2).to(torch.float32).cuda() / 255.0
#    T2 = cpab([3,3], backend="pytorch", device='gpu')
#    B2 = T2.get_basis()
#    P2 = T2.get_params()
#    d2 = T2.get_theta_dim()
#    g2 = T2.uniform_meshgrid([50, 50])
#    t2 = T2.identity()
#    s2 = T2.sample_transformation()
#    s2 = T2.sample_transformation(1, torch.zeros(d2).cuda(), torch.eye(d2).cuda())
#    v2 = T2.calc_vectorfield(g2, s2)
#    n2 = T2.transform_grid(g2, s2)
#    D2 = T2.transform_data(data, s2, outsize=(*data.shape[2:],))
#    vec_plot = T2.visualize_vectorfield(s2, nb_points=20)
#    plt.figure()
#    plt.imshow(data[0].permute(1,2,0).cpu().numpy())
#    plt.figure()
#    plt.imshow(D2[0].permute(1,2,0).cpu().numpy())
    # 3D

#    
#    #data = np.transpose(objects[0][:1], (0, 2, 3, 4, 1))

#    
##    data = np.random.rand(1,8,12,5,1).astype(np.float32)
##    T3 = cpab([2,2,2], backend="numpy", device='cpu')
##    s3 = T3.identity()
##    D3 = T3.transform_data(data, s3, outsize=(8,12,5))
#
#    #data = torch.tensor(data).permute(0,4,1,2,3)
#    T3 = cpab([2,2,2], backend="pytorch", device='gpu')
#    s3 = T3.sample_transformation()
#    D3 = T3.transform_data(data, s3, outsize=(190, 128, 172))
#    
#    PlotGuiImg(data.cpu())
#    PlotGuiImg(D3.cpu())
    
    
    objects = []
    with (open("ana_small.pkl", "rb")) as openfile:
        while True:
            try:
                objects.append(pickle.load(openfile))
            except EOFError:
                break
    data1 = torch.tensor(objects[0][:1]).to(torch.float32)[:,:,:50,:50,:50].cuda()
    data2 = torch.tensor(objects[0][1:2]).to(torch.float32)[:,:,:50,:50,:50].cuda()
    
    T = cpab([2,2,2], backend='pytorch', device='gpu')
    from libcpab import DataAligner
    A = DataAligner(T)
    theta, data1_t = A.alignment_by_gradient(data1, data2, maxiter=100, lr=1e-4)
