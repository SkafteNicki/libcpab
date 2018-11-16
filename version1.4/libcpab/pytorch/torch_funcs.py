# -*- coding: utf-8 -*-
"""
Created on Mon Aug 27 10:55:31 2018

@author: nsde
"""

#%%
import torch

#%%
def memconsumption():
    """ Report the memory consumption """
    import gc
    import numpy as np
    total = 0
    for obj in gc.get_objects():
        try:
             if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                if len(obj.size()) > 0:
#                    print(type(obj), obj.size(), obj.dtype, obj.device)
                    print('{:22s} {:40s} {:15s} {:10s}'.format(str(type(obj)), str(obj.size()), str(obj.dtype), str(obj.device)))
                    if obj.type() == 'torch.cuda.FloatTensor':
                        total += np.prod(obj.size()) * 32
                    elif obj.type() == 'torch.cuda.LongTensor':
                        total += np.prod(obj.size()) * 64
                    elif obj.type() == 'torch.cuda.IntTensor':
                        total += np.prod(obj.size()) * 32
                    #else:
                    # Few non-cuda tensors in my case from dataloader
        except Exception as e:
             pass
    print("{} GB".format(total/((1024**3) * 8)))

#%%
def torch_mymin(a,b):
    return torch.where(a<b, a, torch.round(b))

#%%
def torch_repeat_matrix(A, n):
    return A[None, :, :].repeat(n, 1, 1)

#%%
def torch_expm(A):
    ''' '''
    n_A = A.shape[0]
    A_fro = torch.sqrt(A.abs().pow(2).sum(dim=(1,2), keepdim=True))
    
    # Scaling step
    maxnorm = torch.Tensor([5.371920351148152]).type(A.dtype).to(A.device)
    zero = torch.Tensor([0.0]).type(A.dtype).to(A.device)
    n_squarings = torch.max(zero, torch.ceil(torch_log2(A_fro / maxnorm)))
    Ascaled = A / 2.0**n_squarings    
    n_squarings = n_squarings.flatten().type(torch.int32)
    
    # Pade 13 approximation
    U, V = torch_pade13(Ascaled)
    P = U + V
    Q = -U + V
    R, _ = torch.gesv(P, Q) # solve P = Q*R
    
    # Unsquaring step
    expmA = R
    for i in range(n_A):
        for _ in range(n_squarings[i]):
            expmA[i] = torch.matmul(expmA[i], expmA[i])
    return expmA

#%%
def torch_log2(x):
    return torch.log(x) / torch.log(torch.Tensor([2.0])).type(x.dtype).to(x.device)

#%%    
def torch_pade13(A):
    b = torch.Tensor([64764752532480000., 32382376266240000., 7771770303897600.,
                      1187353796428800., 129060195264000., 10559470521600.,
                      670442572800., 33522128640., 1323241920., 40840800.,
                      960960., 16380., 182., 1.]).type(A.dtype).to(A.device)
        
    ident = torch.eye(A.shape[1], dtype=A.dtype).to(A.device)
    A2 = torch.matmul(A,A)
    A4 = torch.matmul(A2,A2)
    A6 = torch.matmul(A4,A2)
    U = torch.matmul(A, torch.matmul(A6, b[13]*A6 + b[11]*A4 + b[9]*A2) + b[7]*A6 + b[5]*A4 + b[3]*A2 + b[1]*ident)
    V = torch.matmul(A6, b[12]*A6 + b[10]*A4 + b[8]*A2) + b[6]*A6 + b[4]*A4 + b[2]*A2 + b[0]*ident
    return U, V

#%%
def torch_findcellidx_1D(points, ncx):
    p = points[:,0]
                
    # Floor values to find cell
    idx = torch.floor(p * ncx).type(torch.int32)
    idx = torch.clamp(idx, min=0, max=ncx-1).flatten()
    idx = idx.type(torch.int64)
    return idx

#%%
def torch_findcellidx_2D(points, ncx, ncy):
    p = points.squeeze().t()
    inc_x, inc_y = 1.0/ncx, 1.0/ncy 

    # Determine inner coordinates
    zero = torch.Tensor([0.0]).to(points.device)
    p0 = torch.min(torch.Tensor([ncx*inc_x - 1e-8]).to(points.device), torch.max(zero, p[0,:]))
    p1 = torch.min(torch.Tensor([ncy*inc_y - 1e-8]).to(points.device), torch.max(zero, p[1,:]))            
    xmod = torch.fmod(p0, inc_x)
    ymod = torch.fmod(p1, inc_y)            
    x = xmod / inc_x
    y = ymod / inc_y
        
    # Calculate initial cell index    
    cell_idx =  torch_mymin((ncx - 1) * torch.ones_like(p0), (p0 - xmod) / inc_x) + \
                torch_mymin((ncy - 1) * torch.ones_like(p0), (p1 - ymod) / inc_y) * ncx 
    cell_idx *= 4
   
    cell_idx1 = cell_idx+1
    cell_idx2 = cell_idx+2
    cell_idx3 = cell_idx+3

    # Conditions to evaluate        
    cond1 = torch.le(p[0,:], 0)
    cond1_1 = torch.eq(torch.le(p[1,:], 0), p[1,:]/inc_y< p[0,:]/inc_x)
    cond1_2 = torch.eq(torch.ge(p[1,:], ncy*inc_y), p[1,:]/inc_y - ncy> -p[0,:]/inc_x)
    cond2 = torch.ge(p[0,:], ncx*inc_x) 
    cond2_1 = torch.eq(torch.le(p[1,:],0), -p[1,:]/inc_y>p[0,:]/inc_x-ncx)
    cond2_2 = torch.eq(torch.ge(p[1,:],ncy*inc_y), p[1,:]/inc_y - ncy>p[0,:]/inc_x-ncx)
    cond3 = torch.le(p[1,:], 0)
    cond4 = torch.ge(p[1,:], ncy*inc_y)
    cond5 = x<y
    cond5_1 = 1-x<y

    # Take decision based on the conditions
    idx = torch.where(cond1, torch.where(cond1_1, cell_idx, torch.where(cond1_2, cell_idx2, cell_idx3)),
          torch.where(cond2, torch.where(cond2_1, cell_idx, torch.where(cond2_2, cell_idx2, cell_idx1)),
          torch.where(cond3, cell_idx, 
          torch.where(cond4, cell_idx2,
          torch.where(cond5, torch.where(cond5_1, cell_idx2, cell_idx3), 
          torch.where(cond5_1, cell_idx1, cell_idx))))))
    idx = idx.type(torch.int64)
    return idx

#%%
def torch_findcellidx_3D(points, ncx, ncy, ncz):
    p = points.squeeze().t() # [4, n_points]
    
    # Initial row, col placement
    zero = torch.Tensor([0.0]).to(points.device)
    p0 = torch_mymin((ncx-1)*torch.ones_like(p[0,:]).to(points.device), torch.max(zero, p[0,:]*ncx))
    p1 = torch_mymin((ncy-1)*torch.ones_like(p[1,:]).to(points.device), torch.max(zero, p[1,:]*ncy))
    p2 = torch_mymin((ncz-1)*torch.ones_like(p[2,:]).to(points.device), torch.max(zero, p[2,:]*ncz))

    # Initial cell index
    cell_idx = 6*(p0 + p1*ncx + p2*ncx*ncy)
    x = p[0,:]*ncx - p0
    y = p[1,:]*ncy - p1
    z = p[2,:]*ncz - p2
        
    # Find inner thetrahedron
    cell_idx = torch.where((x>y) & (x<=1-y) & (y<z) & (1-y>=z), cell_idx+1, cell_idx)
    cell_idx = torch.where((x>=z) & (x<1-z) & (y>=z) & (y<1-z), cell_idx+2, cell_idx)
    cell_idx = torch.where((x<=z) & (x>1-z) & (y<=z) & (y>1-z), cell_idx+3, cell_idx)
    cell_idx = torch.where((x<y) & (x>=1-y) & (y>z) & (1-y<=z), cell_idx+4, cell_idx)
    cell_idx = torch.where((x>=y) & (1-x<y) & (x>z) & (1-x<=z), cell_idx+5, cell_idx)
    cell_idx = cell_idx.type(torch.int64)
    return cell_idx

#%%
def torch_findcellidx(ndim, points, nc):
    if(ndim==1):
        return torch_findcellidx_1D(points, *nc)
    elif(ndim==2):
        return torch_findcellidx_2D(points, *nc)
    elif(ndim==3):
        return torch_findcellidx_3D(points, *nc)
    else:
        ValueError('What the heck! ndim should only be 1, 2 or 3')

#%%
def torch_interpolate_1D(data, points):
    n_batch = data.shape[0]
    in_size = data.shape[1]
    out_size = points.shape[2]
    max_x = in_size - 1
    
    # Extract points
    x = points[:,0].flatten()
    
    # Scale to domain
    x = ((x+1)*in_size) / 2.0
    
    # Do sampling
    x0 = torch.floor(x).type(torch.int64)
    x1 = x0 + 1
    
    # Clip values
    x0 = torch.clamp(x0, 0, max_x)
    x1 = torch.clamp(x1, 0, max_x)
    
    # Take care of batch effect
    base = (torch.arange(n_batch)*in_size).repeat(out_size,1).t().flatten()
    base = base.to(points.device)
    idx1 = base + x0
    idx2 = base + x1
    
    # Lookup values
    data_flat = data.flatten()
    i1 = torch.gather(data_flat, 0, idx1).type(torch.float32)
    i2 = torch.gather(data_flat, 0, idx2).type(torch.float32)
        
    # Convert to floats
    x0 = x0.type(torch.float32)
    x1 = x1.type(torch.float32)
        
    # Interpolation weights
    w1 = (x - x0)
    w2 = (x1 - x)
        
    # Do interpolation
    new_data = w1*i1 + w2*i2
        
    # Reshape and return
    new_data = torch.reshape(new_data, (n_batch, out_size))
    return new_data
