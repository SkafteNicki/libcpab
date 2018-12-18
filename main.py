# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 15:40:59 2018

@author: nsde
"""

if __name__ == "__main__":
    import torch
    import pickle
    from libcpab import cpab, DataAligner
    from BrainFuns import PlotGuiImg
    objects = pickle.load(open("ana_small.pkl", "rb"))
    data1 = torch.tensor(objects[:1]).to(torch.float32).cuda()
    data2 = torch.tensor(objects[1:2]).to(torch.float32).cuda()
    
    T = cpab([2,2,2], backend='pytorch', device='gpu', override=True)
    T.set_solver_params(numeric_grad=True)

    A = DataAligner(T)
    theta, data1_t = A.alignment_by_gradient(data1, data2, maxiter=100, lr=1e-1)
    
    with open('res.pkl', 'wb') as f:
        pickle.dump([data1.cpu().numpy(), data2.cpu().numpy(), data1_t.cpu().detach().numpy()], f)
    

