#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 11:57:29 2018

@author: buejuljorgensen
"""
# function that import brain data
def LoadBrain(filename):
    import csv
    import numpy as np
    
    with open(filename, newline='') as f:
        data=[]
        for row in csv.reader(f,quoting=csv.QUOTE_NONNUMERIC,delimiter=','):
            data.append(row)
        f.close
    data=data[0]

    dim=[0,0,0]
    for i in range(0,3):
        dim[i]=int(data[i])

    data=np.reshape(data[3:],dim)
    return data

# Function that plots a 3d image at corrdinate coor
def PlotBrain(data,coor):
    import matplotlib.pyplot as plt
    import numpy as np
    
    
    dim=data.shape
    
    plt.figure()
    plt.subplot(2,2,1)
    plt.xlim((0,dim[1]))
    plt.ylim((0,dim[0]))
    plt.xlabel('y')
    plt.ylabel('x')
    y=np.linspace(0,dim[1],dim[1])
    x=np.linspace(0,dim[0],dim[0])
    z=np.linspace(0,dim[2],dim[2])
    plt.imshow(data[:,:,coor[2]])
    plt.plot(np.ones(dim[0])*coor[1],x,color='r') #Line with x-value fixed
    plt.plot(y,np.ones(dim[1])*coor[0],color='r') #Line with y-value fixed
    plt.xticks([])
    plt.yticks([])
    plt.title('(y,x)-plane')
    
    plt.subplot(2,2,2)
    plt.xlim((0,dim[0]))
    plt.ylim((0,dim[2]))
    plt.xlabel('z')
    plt.ylabel('x')
    plt.imshow(data[:,coor[1],:])
    plt.plot(np.ones(dim[0])*coor[2],z,color='r') #Line with x-value fixed
    plt.plot(z,np.ones(dim[2])*coor[0],color='r') #Line with z-value fixed
    plt.xticks([])
    plt.yticks([])
    plt.title('(z,x)-plane')
    
    plt.subplot(2,2,3)
    plt.xlim((0,dim[2]))
    plt.ylim((0,dim[1]))
    plt.xlabel('z')
    plt.ylabel('y')
    plt.imshow(data[coor[0],:,:])
    plt.plot(np.ones(dim[1])*coor[2],y,color='r') #Line with y-value fixed
    plt.plot(z,np.ones(dim[2])*coor[1],color='r') #Line with z-value fixed
    plt.xticks([])
    plt.yticks([])
    plt.title('(z,y)-plane')
    
    plt.subplot(2,2,4)
    plt.xlim((0,100))
    plt.ylim((0,100))
    plt.axis('off')
    plt.text(50, 50, '[x,y,x]-coordinate:\n \n' + str(coor), horizontalalignment='center',verticalalignment='center')
    plt.xticks([])
    plt.yticks([])
    
    plt.show()
    
# Function that plots an illustration of a 3d transformation trans from ind to target at coordinate coor
def PlotBrainTrans(ind,target,trans,coor):
    
    import matplotlib.pyplot as plt
    import numpy as np
    
    dim=ind.shape
    
    y=np.linspace(0,dim[1],dim[1])
    x=np.linspace(0,dim[0],dim[0])
    z=np.linspace(0,dim[2],dim[2])
    
    img_type=('Input image','Transformed image','Target image')
    
    data=(ind,trans,target)
    
    fig = plt.figure()
    
    LW = 0.75
    
    for i in range(0,3):
        plt.subplot(3,4,(i*4)+1)
        plt.xlim((0,dim[1]))
        plt.ylim((0,dim[0]))
        plt.xlabel('y')
        plt.ylabel('x')
        plt.imshow(data[i][:,:,coor[2]])
        plt.plot(np.ones(dim[0])*coor[1],x,color='r',lw=LW) #Line with x-value fixed
        plt.plot(y,np.ones(dim[1])*coor[0],color='r',lw=LW) #Line with y-value fixed
        plt.xticks([])
        plt.yticks([])

        
        plt.subplot(3,4,(i*4)+2)
        plt.xlim((0,dim[0]))
        plt.ylim((0,dim[2]))
        plt.xlabel('z')
        plt.ylabel('x')
        plt.imshow(data[i][:,coor[1],:])
        plt.plot(np.ones(dim[0])*coor[2],z,color='r',lw=LW) #Line with x-value fixed
        plt.plot(z,np.ones(dim[2])*coor[0],color='r',lw=LW) #Line with z-value fixed
        plt.xticks([])
        plt.yticks([])
        
        
        plt.subplot(3,4,(i*4)+3)
        plt.xlim((0,dim[2]))
        plt.ylim((0,dim[1]))
        plt.xlabel('z')
        plt.ylabel('y')
        plt.imshow(data[i][coor[0],:,:])
        plt.plot(np.ones(dim[1])*coor[2],y,color='r',lw=LW) #Line with y-value fixed
        plt.plot(z,np.ones(dim[2])*coor[1],color='r',lw=LW) #Line with z-value fixed
        plt.xticks([])
        plt.yticks([])
        
        plt.subplot(3,4,(i*4)+4)
        plt.xlim((0,100))
        plt.ylim((0,100))
        plt.axis('off')
        plt.text(50, 50, img_type[i], horizontalalignment='center',verticalalignment='center')
        plt.xticks([])
        plt.yticks([])
        
    plt.suptitle('[x,y,z]-coordinate:' + str(coor))
    fig.show()

# Function that normalizes the data 
def Normalize(data):
    data = data-data.mean()
    data = data / data.std()
    return data

# Function that extracts a (2d or 3d) patch of the image of sice slice_size
# The patch is extracted from the middle of the image
def SliceBrain(data,slice_size):
    import numpy as np
    
    dim_brain = np.array(np.shape(data))
    dim_slice = np.array(np.shape(slice_size)[0])
        
    if len(dim_brain) != dim_slice:
        print('Data and slize_size must have the same dimensions')
        return
    if (dim_brain%2).sum() != 0:
        print('Brain dimensions must be even')
        return
    
    if (np.array(slice_size)%2).sum() != 0:
        print('slice_size dimensions must be even')
    
    if len(dim_brain) == 2:
        mid = dim_brain/2
        upper = [0,0]
        lower = [0,0]
        for i in range(2):
            upper[i]=int(mid[i]+slice_size[i]/2)
            lower[i]=int(mid[i]-slice_size[i]/2)
        
        return data[lower[0]:upper[0],lower[1]:upper[1]]
    
    
    if len(dim_brain) == 3:
        mid = dim_brain/2
        upper = [0,0,0]
        lower = [0,0,0]
        for i in range(3):
            upper[i]=int(mid[i]+slice_size[i]/2)
            lower[i]=int(mid[i]-slice_size[i]/2)
        
        return data[lower[0]:upper[0],lower[1]:upper[1],lower[2]:upper[2]]

# Function that extrapolates a 2d image to size dim_ex
# Extrapolate: mirror the image in the img-boundery to make the img larger
def ExtrapolateBrain2d(img,dim_ex):
    import numpy as np
    
    dim_img = np.array(img.shape)
    
    if len(dim_img)==2:
        diff = (dim_ex-dim_img) // 2

        Data = np.zeros(dim_ex)
        
        Data[diff[0]:diff[0]+dim_img[0],diff[1]:diff[1]+dim_img[1]] = img
        
        Data[0:diff[0],0:diff[1]]=np.flip(np.flip(img[0:diff[0],0:diff[1]],axis=0),axis=1)
        
        Data[0:diff[0],(dim_ex[1]-diff[1]):dim_ex[1]]=np.flip(np.flip(img[0:diff[0],(dim_img[1]-diff[1]):dim_img[1]],axis=0),axis=1)
        
        Data[(dim_ex[0]-diff[0]):dim_ex[0],0:diff[1]]=np.flip(np.flip(img[(dim_img[0]-diff[0]):dim_img[1],0:diff[1]],axis=0),axis=1)
        
        Data[(dim_ex[0]-diff[0]):dim_ex[0],(dim_ex[1]-diff[1]):dim_ex[1]]=np.flip(np.flip(img[(dim_img[0]-diff[0]):dim_img[0],(dim_img[1]-diff[1]):dim_img[1]],axis=0),axis=1)
        
        
        
        Data[diff[0]:diff[0]+dim_img[0],0:diff[1]]=np.flip(img[0:dim_img[0],0:diff[1]],axis=1)
        
        Data[diff[0]:diff[0]+dim_img[0],dim_ex[1]-diff[1]:dim_ex[1]]=np.flip(img[0:dim_img[0],dim_img[1]-diff[1]:dim_img[1]],axis=1)
        
        Data[0:diff[0],diff[1]:diff[1]+dim_img[1]]=np.flip(img[0:diff[0],0:dim_img[1]],axis=0)
        
        Data[dim_ex[0]-diff[0]:dim_ex[0],diff[1]:diff[1]+dim_img[1]]=np.flip(img[dim_img[0]-diff[0]:dim_img[0],0:dim_img[1]],axis=0)
    
    
        return Data
    
    return []





# Function that calculates the waights in the loss function (eq 5 in U-Net paper)
def BrainWaight2d(seg,sigma=5,w0=1,plot_waights=False,return_all=False):
    import numpy as np
    import matplotlib.pyplot as plt
    
    im_size=np.array(seg.shape)
    
    # Initializing matrix to contain w_c (first term in eq (2))
    w_c = np.zeros(im_size)
    
    # Initializing matriz to contain waights
    W = np.zeros(im_size)
    
    # Calculating w_c
    for j in np.unique(seg):
        w = (seg==j).mean()
        w_c += (seg==j)*np.exp(-w)   #(1-w)


    # Initializing matrix to contain edges
    edg = np.zeros(im_size)
    
    # Finding edges between segments but not between segment and background
    for i in range(1,(im_size[0]-1)):
        for j in range(1,(im_size[1]-1)):
            p = seg[i-1:i+2,j-1:j+2]
            if (p==0).sum()==0:
                if (p!=seg[i,j]).sum()>0:
                    edg[i,j]=1
    
    # Initializing matrix to contain mask of forground / background
    mask = np.zeros(im_size)
    
    # Setting all segments to 1 in mask
    mask[seg!=0] = 1
    
    # Setting egdes (and background) to 0
    mask[edg==1]=0
    
                           
    # Finding indices where there is edge
    edg_idx=np.zeros((2,(edg>0).sum()))
    
    
    edg_idx[0]=np.where(edg>0)[0]
    
    edg_idx[1]=np.where(edg>0)[1]
    
    edg_idx = np.transpose(edg_idx)
    
    # Finding the segment of each edge pixel
    edg_seg = seg[edg_idx[:,0].astype("int"),edg_idx[:,1].astype("int")]
    
    # Initializing matrices to contain distances to nearest and second to nearest segment
    # Note these distances are only found for background
    dist1 = np.zeros(im_size)
    dist2 = np.zeros(im_size)
    
    for i in range(im_size[0]):
        for j in range(im_size[1]):
            if mask[i,j]==0:
                d1 =(((edg_idx-[i,j])**2).sum(axis=1)**0.5)
                dist1[i,j]=d1.min()
                min_idx = np.where(d1==d1.min())[0][0]
                min_seg = edg_seg[min_idx]
                idx_min_seg = np.where(edg_seg==min_seg)[0]
                d2 = np.delete(d1,idx_min_seg)
                dist2[i,j]=d2.min()

    
    # Calculating the second term in eq (2) in U-Net paper
    d_egd = np.exp((-(dist1+dist2)**2)/(2*sigma**2))
    
    # Setting foreground to 0
    d_egd[mask==1]=0
    
    
    w_c = w_c
    
    W = w_c+w0*d_egd
    
    if plot_waights:
        fig = plt.figure()
        ax = fig.add_subplot(131)
        ax.set_title('W')
        ax = plt.imshow(W,cmap='jet')
        ax = plt.colorbar()
        
        
        ax1 = fig.add_subplot(132)
        ax1.set_title('w_c')
        ax1 = plt.imshow(w_c,cmap='jet')
        ax2 = plt.colorbar()
        
        ax2 = fig.add_subplot(133)
        ax2.set_title('d_egd')
        ax2 = plt.imshow(d_egd,cmap='jet')
        ax2 = plt.colorbar()
    
    if return_all:
        return W, w_c, d_egd
    else:
        return W

def Com(seg):       
    for i in range(39):
        test = (seg==i).float()
        
        if test.sum()!=0:
        
            test /= test.float().sum()
            
            dz = test.sum(dim=0).sum(dim=0)
            
            dy = test.sum(dim=0).sum(dim=1)
            
            dx = test.sum(dim=1).sum(dim=1)
            
            com = torch.zeros((39,3))
            
            com[i,2] = (torch.arange(172).float()*dz).sum()
            
            com[i,1] = (torch.arange(128).float()*dy).sum()
            
            com[i,0] = (torch.arange(190).float()*dx).sum()
            
            COM = com.long()
        
        
            fig = plt.figure()
            ax1 = fig.add_subplot(131)
            ax1 = plt.imshow(test[COM[i,0],:,:])
            ax1 = plt.scatter(com[i,2],com[i,1])
            
            ax2 = fig.add_subplot(132)
            ax2 = plt.imshow(test[:,COM[i,1],:])
            ax2 = plt.scatter(com[i,2],com[i,0])
            
            ax3 = fig.add_subplot(133)
            ax3 = plt.imshow(test[:,:,COM[i,2]])
            ax3 = plt.scatter(com[i,1],com[i,0])

def Interpolate(data,ti,device = 'cuda'):

    
    #Finding the actually used data/segment
    if len(data.shape)>3:
        Data = data[0,0,:,...,:]
    else:
        Data = data
    
    Outsize = torch.Tensor([Data.shape[i] for i in range(len(Data.shape))])
    
    if device == 'cuda':
        Outsize = Outsize.cuda()
    
    
    ti_floor = ti.floor()
    
    ti_ceil = ti_floor+1
    
    # Calculating the distances from the not rounded indices to the floored indices
    d1 = 1-(ti-ti_floor)
    
    # Calculating the distances from the not rounded indices to the ceiled indices
    d2 = 1-(ti_ceil-ti)
    
    
    # Finding the lowest transformed index smaller than zero in each dimansion    
    idx_min=ti.floor().min(dim=1)[0].long()
    idx_min[idx_min>0]=0

    # Finding the highest transformed index larger than the largest image dimension in each dimension
    idx_max = (ti.ceil().max(dim=1)[0]-Outsize).long()
    idx_max[idx_max<0]=0
    
    
    # Padding 2D image/segment apropriately
    if len(data.shape)==4 or len(data.shape)==2:
        # Padding the data such that any transformed index outside original image corrosponds to a zero
        data_pad = torch.zeros((idx_max[0]-idx_min[0],idx_max[1]-idx_min[1]))
        data_pad[-idx_min[0]:-idx_min[0]+Data.shape[0],-idx_min[1]:-idx_min[1]+Data.shape[1]]=Data
        
        
        dist = torch.zeros(2,2,ti.shape[1])
        dist[0,0]=d1[0]*d1[1]
        dist[0,1]=d1[0]*d2[1]
        dist[1,0]=d2[0]*d1[1]
        dist[1,1]=d2[0]*d2[1]
        Dist = dist.reshape(4,-1)
                
        # Finding the values in the four proxcimity pixels 
        prox = torch.Tensor(2,2,ti.shape[1])
        prox[0,0] = data_pad[ti_floor[0].long()-idx_min[0]-1,ti_floor[1].long()-idx_min[1]-1]
        prox[0,1] = data_pad[ti_floor[0].long()-idx_min[0]-1,ti_ceil[1].long()-idx_min[1]-1]
        prox[1,0] = data_pad[ti_ceil[0].long()-idx_min[0]-1,ti_floor[1].long()-idx_min[1]-1]
        prox[1,1] = data_pad[ti_ceil[0].long()-idx_min[0]-1,ti_ceil[1].long()-idx_min[1]-1]
        
        Prox = prox.reshape(4,-1)
        
        # Interpolating the image at the transformed indices
        if len(data.shape)>3:
            inter = (Prox*Dist).sum(0)
            return inter
        # Interpolating the segment at the transformed indices
        else:
            #(dummy, ind) = Dist.max(dim=0)
        
            #inter_seg = Prox[ind,torch.arange(ti.shape[1])]
            
            Ti = ti.round().long()
            
            inter_seg = data_pad[Ti[0]-idx_min[0]-1,Ti[1]-idx_min[1]-1]
            return inter_seg
        
        
        
    # Padding 3D image / segment   
    if len(data.shape)==5 or len(data.shape)==3:
        
        padding = torch.nn.ReplicationPad3d((-idx_min[2],idx_max[2],-idx_min[1],idx_max[1],-idx_min[0],idx_max[0]))
        
        data_pad = padding(Data.unsqueeze(0).unsqueeze(0))
        data_pad = data_pad[0,0]
        
        
        dist = torch.zeros(2,2,2,ti.shape[1])
        dist[0,0,0]=d1[0]*d1[1]*d1[2]
        dist[0,0,1]=d1[0]*d1[1]*d2[2]
        dist[0,1,0]=d1[0]*d2[1]*d1[2]
        dist[0,1,1]=d1[0]*d2[1]*d2[2]
        dist[1,0,0]=d2[0]*d1[1]*d1[2]
        dist[1,0,1]=d2[0]*d1[1]*d2[2]
        dist[1,1,0]=d2[0]*d2[1]*d1[2]
        dist[1,1,1]=d2[0]*d2[1]*d2[2]
        
        Dist = dist.reshape(8,-1)
        
        # Finding the values of the image in the six corners of the square that the transformed index is in
        
        prox = torch.Tensor(2,2,2,ti.shape[1])
        prox[0,0,0]=data_pad[ti_floor[0].long()-idx_min[0]-1,ti_floor[1].long()-idx_min[1]-1,ti_floor[2].long()-idx_min[2]-1]
        prox[0,0,1]=data_pad[ti_floor[0].long()-idx_min[0]-1,ti_floor[1].long()-idx_min[1]-1,ti_ceil[2].long()-idx_min[2]-1]
        prox[0,1,0]=data_pad[ti_floor[0].long()-idx_min[0]-1,ti_ceil[1].long()-idx_min[1]-1,ti_floor[2].long()-idx_min[2]-1]
        prox[0,1,1]=data_pad[ti_floor[0].long()-idx_min[0]-1,ti_ceil[1].long()-idx_min[1]-1,ti_ceil[2].long()-idx_min[2]-1]
        prox[1,0,0]=data_pad[ti_ceil[0].long()-idx_min[0]-1,ti_floor[1].long()-idx_min[1]-1,ti_floor[2].long()-idx_min[2]-1]
        prox[1,0,1]=data_pad[ti_ceil[0].long()-idx_min[0]-1,ti_floor[1].long()-idx_min[1]-1,ti_ceil[2].long()-idx_min[2]-1]
        prox[1,1,0]=data_pad[ti_ceil[0].long()-idx_min[0]-1,ti_ceil[1].long()-idx_min[1]-1,ti_floor[2].long()-idx_min[2]-1]
        prox[1,1,1]=data_pad[ti_ceil[0].long()-idx_min[0]-1,ti_ceil[1].long()-idx_min[1]-1,ti_ceil[2].long()-idx_min[2]-1]
        
        Prox = prox.reshape(8,-1)
        
        # Interpolating the image at the transformed indices
        if len(data.shape)>3:
            inter = (Prox*Dist).sum(0)
            return inter
        # Interpolating the segment at the transformed indices
        else:
            (dummy, ind) = Dist.max(dim=0)
        
            inter_seg = Prox[ind,torch.arange(ti.shape[1])]
            return inter_seg

    

def SparseTransformation(T, data , theta, idx, plot_trans = False,trans_seg = False, seg = [],device ='cpu'):
    import matplotlib as plt
    
    if len(data.shape) not in [4,5]:
        raise ValueError("Invalid dimension of data. Must be 4 for 2d or 5 for 3d")
        
    #Finding the image size
    outsize = [data.shape[i] for i in range(2,len(data.shape))]
    
    # Making a grid of points corrosponding to the image size
    points = T.uniform_meshgrid(outsize)
    
    # Picking out some of the grid points to transform - If there are more indices than points, all points are used
    used_points = points[:,idx]
    
    
    # Converting the image size to torch variable with appropriate dimensions
    outsize = torch.Tensor(np.expand_dims(outsize,1))
    if device == 'cuda':
        outsize = outsize.cuda()

    # Transforming the picked-out gridpoints and converting them to NOT rounded indices 
    ti = T.transform_grid(used_points, theta)[0]*(outsize-1)

    
    # Converting the used points to indexies
    used_idx = (used_points*(outsize-1)).round().long()
    
    inter = Interpolate(data,ti)
    
    if trans_seg == True:
        inter_seg = Interpolate(seg,ti)
        return inter, used_idx, inter_seg, ti
    else:
        return inter, used_idx, ti
    

#%% Simularity measures
        
def dist_loss(ti,Dist,target_seg,used_idx,seg_normalization=False,device='gpu',seg_intersect = [0]):
    
    if len(seg_intersect)==1:
        raise ValueError('seg_intersect must be defined')
    
    d = []
    
    outsize = [Dist[0].shape[i] for i in range(0,len(Dist[0].shape))]    
    outsize = torch.Tensor(np.expand_dims(outsize,1))
    
    
    # Finding the target segments of the transformed points
    t_seg = target_seg[used_idx[0],used_idx[1],used_idx[2]].long()
    
    if device == 'cuda':
        outsize = outsize.cuda()
        t_seg = t_seg.cuda()

    # looping over all segments in both input and target
    for i in seg_intersect:
        
        # Picking distance "mask" according to segment
        dist = Dist[i].unsqueeze(0).unsqueeze(0)
        

        # Finding indicies where the target segment is equal to the looped segment         
        ind = (t_seg==i).nonzero().flatten()
        
        if len(ind)!=0:
            
            #Interpolating the distance mask at the transformed indices
            if seg_normalization:
                # Taking the mean over each segment to put equal emphaphis on each segment regardless of size
                d.append(Interpolate(dist,ti[:,ind]).mean())
            else:
                d.append(Interpolate(dist,ti[:,ind]))
    
    if  seg_normalization:
        d = torch.stack(d).mean()
    else: 
        d=torch.cat(d).mean()
    
    return d

# Function that calculates Cohen's Kappa
def Kappa(true_l,pred):
    
    from sklearn.metrics import confusion_matrix
    
    if true_l.shape != pred.shape:
        raise ValueError("Arrays must have the same size")
    
    if len(true_l.shape)==3:
        
        res = 0
        
        for j in range(true_l.shape[0]):
            conf_mat= confusion_matrix(true_l[j].flatten(),pred[j].flatten())
    
            n = conf_mat.sum()
            
            P_o = conf_mat.diagonal().sum()/n
            
            P_e = 0
            
            for i in range(conf_mat.shape[0]):
                P = conf_mat[:,i].sum()/n*conf_mat[i,:].sum()/n
                
                P_e += P
                
            res += (P_o-P_e)/(1-P_e)
            
        return res/true_l.shape[0]
            
    else:
        conf_mat= confusion_matrix(true_l.flatten(),pred.flatten())
        
        n = conf_mat.sum()
        
        P_o = conf_mat.diagonal().sum()/n
        
        P_e = 0
        
        for i in range(conf_mat.shape[0]):
            P = conf_mat[:,i].sum()/n*conf_mat[i,:].sum()/n
            
            P_e += P
            
        return (P_o-P_e)/(1-P_e)
    
# Function that calculates the Sørensen - Dice coefficent
def Dice(s1,s2):
    import numpy as np
    import torch
    
    if s1.shape != s2.shape:
        raise ValueError("Arrays must have the same size")
    
    if type(s1)==torch.Tensor:
        s1 = s1.numpy()
    if type(s2)==torch.Tensor:    
        s2 = s2.numpy()
        
    if (len(s1.shape)==3) & (len(s2.shape)==3):
        res = 0
        for j in range(s1.shape[0]):
            numerator = 0
            denom = 0
            
            for i in np.unique([s1,s2]):
                numerator += np.min([(s1[j]==i).sum(),(s2[j]==i).sum()])
                denom += (s1[j]==i).sum()+(s2[j]==i).sum()
                
            numerator *= 2
            
            res += numerator/ denom
        
        return res / s1.shape[0]
        
            
    else:        
        numerator = 0
        denom = 0
        
        for i in np.unique([s1,s2]):
            numerator += np.min([(s1==i).sum(),(s2==i).sum()])
            denom += (s1==i).sum()+(s2==i).sum()
            
        numerator *= 2
     
    return numerator/denom

# Function that calculates the recall for each class and takes the average of these
def AvgRecall(y_pred,y_true):
    avg_recall = 0
    
    for h in np.unique(y_true):
        avg_recall += ((y_pred==int(h)) & (y_true==int(h))).sum().float()/((y_true==int(h)).sum()).float()
    
    avg_recall /= len(np.unique(y_true))
    
    return avg_recall

# Function that calculates the precission for each class and takes the average of these
def AvgPrecission(y_pred,y_true):
    avg_pre = 0
    
    for h in np.unique(y_true):
        if h in np.unique(y_pred):
            avg_pre += ((y_pred==int(h)) & (y_true==int(h))).sum().float()/((y_pred==int(h)).sum()).float()
    
    avg_pre /= len(np.unique(y_true))
    
    return avg_pre

#%% U-Nets
    
from torch.nn import Linear, Conv2d, BatchNorm2d, MaxPool2d, Dropout2d, ConvTranspose2d, Softmax
from torch.nn.functional import relu, elu, relu6, sigmoid, tanh, softmax

import torch.nn as nn
import torch
import numpy as np
'''
# Function for computing dimension after a conv. layer is applied
def compute_conv_dim(dim_size,kernel_size,stride,padding):
    return (np.array(dim_size) - np.array(kernel_size) + 2 * np.array(padding)) // np.array(stride) + 1
'''

# define network
class U_Net2d(nn.Module):

    def __init__(self,num_classes):
        super(U_Net2d, self).__init__()
        
        # hyperameters of the model
        channels = 1
        
        num_filters = 64
        kernel_size_down = [3,3] # [height, width]
        kernel_size_up = [2,2]
        stride = [1,1] # [stride_height, stride_width]
        padding = [0,0]
        self.activation = relu
        
        
        
        self.conv_11 = Conv2d(in_channels=channels,
                            out_channels=num_filters,
                            kernel_size=kernel_size_down,
                            stride=stride,
                            padding=padding)
        
        
        self.conv_12 = Conv2d(in_channels=num_filters,
                            out_channels=num_filters,
                            kernel_size=kernel_size_down,
                            stride=stride,
                            padding=padding)
        
        
        self.MaxPool = MaxPool2d(2)
        
        
        self.conv_21 = Conv2d(in_channels=num_filters,
                            out_channels=num_filters*2,
                            kernel_size=kernel_size_down,
                            stride=stride,
                            padding=padding)
        
        
        self.conv22 = Conv2d(in_channels=num_filters*2,
                            out_channels=num_filters*2,
                            kernel_size=kernel_size_down,
                            stride=stride,
                            padding=padding)
        
        
        self.conv31 = Conv2d(in_channels=num_filters*2,
                            out_channels=num_filters*2*2,
                            kernel_size=kernel_size_down,
                            stride=stride,
                            padding=padding)
        
        
        self.conv32 = Conv2d(in_channels=num_filters*2*2,
                            out_channels=num_filters*2*2,
                            kernel_size=kernel_size_down,
                            stride=stride,
                            padding=padding)
        
        
        self.conv41 = Conv2d(in_channels=num_filters*2*2,
                            out_channels=num_filters*2*2*2,
                            kernel_size=kernel_size_down,
                            stride=stride,
                            padding=padding)
        
        
        self.conv42 = Conv2d(in_channels=num_filters*2*2*2,
                            out_channels=num_filters*2*2*2,
                            kernel_size=kernel_size_down,
                            stride=stride,
                            padding=padding)
        
        
        self.conv51 = Conv2d(in_channels=num_filters*2*2*2,
                            out_channels=num_filters*2*2*2*2,
                            kernel_size=kernel_size_down,
                            stride=stride,
                            padding=padding)
        

        
        self.conv52 =  Conv2d(in_channels=num_filters*2*2*2*2,
                            out_channels=num_filters*2*2*2*2,
                            kernel_size=kernel_size_down,
                            stride=stride,
                            padding=padding)
        
        self.conv_trans1 = ConvTranspose2d(in_channels = num_filters*2*2*2*2,
                                           out_channels = num_filters*2*2*2,
                                           kernel_size = kernel_size_up,
                                           stride = [2,2],
                                           padding = padding)
        
        self.conv61 = Conv2d(in_channels=num_filters*2*2*2*2,
                            out_channels=num_filters*2*2*2,
                            kernel_size=kernel_size_down,
                            stride=stride,
                            padding=padding)
        
        self.conv62 = Conv2d(in_channels=num_filters*2*2*2,
                            out_channels=num_filters*2*2*2,
                            kernel_size=kernel_size_down,
                            stride=stride,
                            padding=padding)
        
        self.conv_trans2 = ConvTranspose2d(in_channels = num_filters*2*2*2,
                                           out_channels = num_filters*2*2,
                                           kernel_size = kernel_size_up,
                                           stride = [2,2],
                                           padding = padding)
        
        
        self.conv71 = Conv2d(in_channels=num_filters*2*2*2,
                            out_channels=num_filters*2*2,
                            kernel_size=kernel_size_down,
                            stride=stride,
                            padding=padding)
        
        self.conv72 = Conv2d(in_channels=num_filters*2*2,
                            out_channels=num_filters*2*2,
                            kernel_size=kernel_size_down,
                            stride=stride,
                            padding=padding)
        
        self.conv_trans3 = ConvTranspose2d(in_channels = num_filters*2*2,
                                           out_channels = num_filters*2,
                                           kernel_size = kernel_size_up,
                                           stride = [2,2],
                                           padding = padding)
        
        self.conv81 = Conv2d(in_channels=num_filters*2*2,
                            out_channels=num_filters*2,
                            kernel_size=kernel_size_down,
                            stride=stride,
                            padding=padding)
        
        self.conv82 = Conv2d(in_channels=num_filters*2,
                            out_channels=num_filters*2,
                            kernel_size=kernel_size_down,
                            stride=stride,
                            padding=padding)
        
        self.conv_trans4 = ConvTranspose2d(in_channels = num_filters*2,
                                           out_channels = num_filters,
                                           kernel_size = kernel_size_up,
                                           stride = [2,2],
                                           padding = padding)
        

        self.conv91 = Conv2d(in_channels=num_filters*2,
                            out_channels=num_filters,
                            kernel_size=kernel_size_down,
                            stride=stride,
                            padding=padding)
        
        self.conv92 = Conv2d(in_channels=num_filters,
                            out_channels=num_filters,
                            kernel_size=kernel_size_down,
                            stride=stride,
                            padding=padding)
        
        self.conv_out = Conv2d(in_channels=num_filters,
                            out_channels=num_classes,
                            kernel_size=[5,5],
                            stride=stride,
                            padding=padding)
        
        self.Softmax = Softmax(dim=1)
        
        
        # add dropout to network
        #self.dropout = Dropout2d(p=0.5)
        
        # Feedforward layers
        #self.l_1 = Linear(in_features=num_filters_conv1*np.prod(dim_MP), 
        #                  out_features=num_l1,
        #                  bias=True)
        #self.l_out = Linear(in_features=num_l1, 
        #                    out_features=np.prod(dim_in),
        #                   bias=False)
    
    def forward(self, x):
        
        # Down convoluting
        x1 = self.activation(self.conv_11(x))
        x1 = self.activation(self.conv_12(x1))
        x = self.MaxPool(x1)
        
        x2 = self.activation(self.conv_21(x))
        x2 = self.activation(self.conv22(x2))
        x = self.MaxPool(x2)
        
        x3 = self.activation(self.conv31(x))
        x3 = self.activation(self.conv32(x3))
        x = self.MaxPool(x3)
        
        x4 = self.activation(self.conv41(x))
        x4 = self.activation(self.conv42(x4))
        x = self.MaxPool(x4)
        
        x = self.activation(self.conv51(x))
        x = self.activation(self.conv52(x))
        
        # Up convoluting
        x = self.conv_trans1(x)
        x = self.combine(x4,x)
        x = self.activation(self.conv61(x))
        x = self.activation(self.conv62(x))
        
        x = self.conv_trans2(x)
        x = self.combine(x3,x)
        x = self.activation(self.conv71(x))
        x = self.activation(self.conv72(x))
        
        x = self.conv_trans3(x)
        x = self.combine(x2,x)
        x = self.activation(self.conv81(x))
        x = self.activation(self.conv82(x))

        x = self.conv_trans4(x)
        x = self.combine(x1,x)
        x = self.activation(self.conv91(x))
        x = self.activation(self.conv92(x))
        
        #Output
        x = self.Softmax(self.conv_out(x))
        
        return x
    
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
    
    def combine(self,old,new):
        old_size = np.array(old.shape[2:4])
        
        new_size = np.array(new.shape[2:4])
        
        diff = (old_size-new_size) // 2
        
        return torch.cat((old[:,:,diff[0]:diff[0]+new_size[0],diff[1]:diff[1]+new_size[1]],new),dim=1)
 
#%% Gui's
        
def PlotGuiSeg(seg,Ind=[0],inp = [0],Ind_r=[0],Ind_w = [0]):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider
    
    
    outsize = [seg.shape[i] for i in range(len(seg.shape))]
    cmap = plt.get_cmap('jet', 39)
    
    coor0 = int(outsize[0]/2)
    coor1 = int(outsize[1]/2)
    coor2 = int(outsize[2]/2)
    
    delta_coor = 1
    
    if len(inp)==1:
        fig, ax = plt.subplots(nrows=1,ncols=3,)
        plt.subplots_adjust(left=0.1, bottom=0.25)
        ax[0].set_xlabel('z')
        ax[0].set_ylabel('y')
        ax[1].set_xlabel('z')
        ax[1].set_ylabel('x')
        ax[2].set_xlabel('y')
        ax[2].set_ylabel('x')
        
        # Removing axis
        for i in range(3):
            ax[i].xaxis.set_major_locator(plt.NullLocator())
            ax[i].yaxis.set_major_locator(plt.NullLocator())
        
           
        ax[0].imshow(seg[coor0,:,:],cmap=cmap,vmin=0,vmax = 39)
        ax[1].imshow(seg[:,coor1,:],cmap=cmap,vmin=0,vmax = 39)
        ax[2].imshow(seg[:,:,coor2],cmap=cmap,vmin=0,vmax = 39)
        
        if len(Ind)!=1:
            ind0 = Ind[Ind[:,0]==coor0]
            ind1 = Ind[Ind[:,1]==coor1]
            ind2 = Ind[Ind[:,2]==coor2]
            ax[0].scatter(ind0[:,2],ind0[:,1],s=1,marker='s')
            ax[1].scatter(ind1[:,2],ind1[:,0],s=1,marker='s') 
            ax[2].scatter(ind2[:,1],ind2[:,0],s=1,marker='s')
        
        if len(Ind_r)!=1:
            ind0 = Ind_r[Ind_r[:,0]==coor0]
            ind1 = Ind_r[Ind_r[:,1]==coor1]
            ind2 = Ind_r[Ind_r[:,2]==coor2]
            ax[0].scatter(ind0[:,2],ind0[:,1],s=1,marker='s',c='g')
            ax[1].scatter(ind1[:,2],ind1[:,0],s=1,marker='s',c='g') 
            ax[2].scatter(ind2[:,1],ind2[:,0],s=1,marker='s',c='g')
            
        if len(Ind_w)!=1:
            ind0 = Ind_w[Ind_w[:,0]==coor0]
            ind1 = Ind_w[Ind_w[:,1]==coor1]
            ind2 = Ind_w[Ind_w[:,2]==coor2]
            ax[0].scatter(ind0[:,2],ind0[:,1],s=1,marker='s',c='r')
            ax[1].scatter(ind1[:,2],ind1[:,0],s=1,marker='s',c='r') 
            ax[2].scatter(ind2[:,1],ind2[:,0],s=1,marker='s',c='r')
            
        
        
        axcolor = 'lightgoldenrodyellow'
        axcoor2 = plt.axes([0.15, 0.05, 0.65, 0.03], facecolor=axcolor)
        axcoor1 = plt.axes([0.15, 0.10, 0.65, 0.03], facecolor=axcolor)
        axcoor0 = plt.axes([0.15, 0.15, 0.65, 0.03], facecolor=axcolor)
        
        scoor0 = Slider(axcoor0, 'x coor', 0, outsize[0]-1, valinit=coor0, valstep=delta_coor)
        scoor1 = Slider(axcoor1, 'y coor ', 0, outsize[1]-1, valinit=coor1,valstep=delta_coor)
        scoor2 = Slider(axcoor2, 'z coor ', 0, outsize[2]-1, valinit=coor2,valstep=delta_coor)
        
        
    
        def update0(val):
            ax[0].cla()
            ax[0].xaxis.set_major_locator(plt.NullLocator())
            ax[0].yaxis.set_major_locator(plt.NullLocator())
            ax[0].set_xlabel('z')
            ax[0].set_ylabel('y')
            coor0 = scoor0.val.astype(int)
            ax[0].imshow(seg[coor0,:,:],cmap = cmap,vmin=0,vmax = 39)
            if len(Ind)!=1:
                ind0 = Ind[Ind[:,0]==coor0]
                ax[0].scatter(ind0[:,2],ind0[:,1],s=1,marker='s')
            if len(Ind_r)!=1:
                ind0 = Ind_r[Ind_r[:,0]==coor0]
                ax[0].scatter(ind0[:,2],ind0[:,1],s=1,marker='s',c='g')
            if len(Ind_w)!=1:
                ind0 = Ind_w[Ind_w[:,0]==coor0]
                ax[0].scatter(ind0[:,2],ind0[:,1],s=1,marker='s',c='r')
            
        def update1(val):
            ax[1].cla()
            ax[1].xaxis.set_major_locator(plt.NullLocator())
            ax[1].yaxis.set_major_locator(plt.NullLocator())
            ax[1].set_xlabel('z')
            ax[1].set_ylabel('x')
            coor1 = scoor1.val.astype(int)
            ax[1].imshow(seg[:,coor1,:],cmap = cmap,vmin=0,vmax = 39)
            if len(Ind)!=1:
                ind1 = Ind[Ind[:,1]==coor1]
                ax[1].scatter(ind1[:,2],ind1[:,0],s=1,marker='s')
            if len(Ind_r)!=1:
                ind1 = Ind_r[Ind_r[:,1]==coor1]
                ax[1].scatter(ind1[:,2],ind1[:,0],s=1,marker='s',c='g')
            if len(Ind_w)!=1:
                ind1 = Ind_w[Ind_w[:,1]==coor1]
                ax[1].scatter(ind1[:,2],ind1[:,0],s=1,marker='s',c='r')
            
        def update2(val):
            ax[2].cla()
            ax[2].xaxis.set_major_locator(plt.NullLocator())
            ax[2].yaxis.set_major_locator(plt.NullLocator())
            ax[2].set_xlabel('y')
            ax[2].set_ylabel('x')
            coor2 = scoor2.val.astype(int)
            ax[2].imshow(seg[:,:,coor2],cmap = cmap,vmin=0,vmax = 39)
            if len(Ind)!=1:
                ind2 = Ind[Ind[:,2]==coor2]
                ax[2].scatter(ind2[:,1],ind2[:,0],s=1,marker='s')
            if len(Ind_r)!=1:
                ind2 = Ind_r[Ind_r[:,2]==coor2]
                ax[2].scatter(ind2[:,1],ind2[:,0],s=1,marker='s',c='g')
            if len(Ind_w)!=1:
                ind2 = Ind_w[Ind_w[:,2]==coor2]
                ax[2].scatter(ind2[:,1],ind2[:,0],s=1,marker='s',c='r')
            
            
        scoor0.on_changed(update0)
        scoor1.on_changed(update1)
        scoor2.on_changed(update2)
        
        
        
        plt.show()
        
    else:
        fig, ax = plt.subplots(nrows=2,ncols=3,)
        plt.subplots_adjust(left=0.1, bottom=0.25,wspace=0.01,hspace=0.1)
        ax[0,0].set_xlabel('z')
        ax[0,0].set_ylabel('y')
        ax[0,1].set_xlabel('z')
        ax[0,1].set_ylabel('x')
        ax[0,2].set_xlabel('y')
        ax[0,2].set_ylabel('x')
        
        ax[1,0].set_xlabel('z')
        ax[1,0].set_ylabel('y')
        ax[1,1].set_xlabel('z')
        ax[1,1].set_ylabel('x')
        ax[1,2].set_xlabel('y')
        ax[1,2].set_ylabel('x')
        
        # Removing axis
        for i in range(3):
            for j in range(2):
                ax[j,i].xaxis.set_major_locator(plt.NullLocator())
                ax[j,i].yaxis.set_major_locator(plt.NullLocator())
        
           
        ax[1,0].imshow(seg[coor0,:,:],cmap=cmap,vmin=0,vmax = 39)
        ax[1,1].imshow(seg[:,coor1,:],cmap=cmap,vmin=0,vmax = 39)
        ax[1,2].imshow(seg[:,:,coor2],cmap=cmap,vmin=0,vmax = 39)
        
        ax[0,0].imshow(inp[coor0,:,:],cmap=cmap,vmin=0,vmax = 39)
        ax[0,1].imshow(inp[:,coor1,:],cmap=cmap,vmin=0,vmax = 39)
        ax[0,2].imshow(inp[:,:,coor2],cmap=cmap,vmin=0,vmax = 39)
        
        
        if len(Ind)!=1:
            ind0 = Ind[Ind[:,0]==coor0]
            ind1 = Ind[Ind[:,1]==coor1]
            ind2 = Ind[Ind[:,2]==coor2]
            ax[0,0].scatter(ind0[:,2],ind0[:,1],s=1,marker='s')
            ax[0,1].scatter(ind1[:,2],ind1[:,0],s=1,marker='s') 
            ax[0,2].scatter(ind2[:,1],ind2[:,0],s=1,marker='s')  
            
            ax[1,0].scatter(ind0[:,2],ind0[:,1],s=1,marker='s')
            ax[1,1].scatter(ind1[:,2],ind1[:,0],s=1,marker='s') 
            ax[1,2].scatter(ind2[:,1],ind2[:,0],s=1,marker='s')
        
        if len(Ind_r)!=1:
            ind0 = Ind_r[Ind_r[:,0]==coor0]
            ind1 = Ind_r[Ind_r[:,1]==coor1]
            ind2 = Ind_r[Ind_r[:,2]==coor2]
            ax[0,0].scatter(ind0[:,2],ind0[:,1],s=1,marker='s',c='r')
            ax[0,1].scatter(ind1[:,2],ind1[:,0],s=1,marker='s',c='r') 
            ax[0,2].scatter(ind2[:,1],ind2[:,0],s=1,marker='s',c='r')  
            
            ax[1,0].scatter(ind0[:,2],ind0[:,1],s=1,marker='s',c='g')
            ax[1,1].scatter(ind1[:,2],ind1[:,0],s=1,marker='s',c='g') 
            ax[1,2].scatter(ind2[:,1],ind2[:,0],s=1,marker='s',c='g')
        
        if len(Ind_w)!=1:
            ind0 = Ind_w[Ind_w[:,0]==coor0]
            ind1 = Ind_w[Ind_w[:,1]==coor1]
            ind2 = Ind_w[Ind_w[:,2]==coor2]
            ax[0,0].scatter(ind0[:,2],ind0[:,1],s=1,marker='s',c='g')
            ax[0,1].scatter(ind1[:,2],ind1[:,0],s=1,marker='s',c='g') 
            ax[0,2].scatter(ind2[:,1],ind2[:,0],s=1,marker='s',c='g')  
            
            ax[1,0].scatter(ind0[:,2],ind0[:,1],s=1,marker='s',c='r')
            ax[1,1].scatter(ind1[:,2],ind1[:,0],s=1,marker='s',c='r') 
            ax[1,2].scatter(ind2[:,1],ind2[:,0],s=1,marker='s',c='r')
        
        axcolor = 'lightgoldenrodyellow'
        axcoor2 = plt.axes([0.15, 0.05, 0.65, 0.03], facecolor=axcolor)
        axcoor1 = plt.axes([0.15, 0.10, 0.65, 0.03], facecolor=axcolor)
        axcoor0 = plt.axes([0.15, 0.15, 0.65, 0.03], facecolor=axcolor)
        
        scoor0 = Slider(axcoor0, 'x coor', 0, outsize[0]-1, valinit=coor0, valstep=delta_coor)
        scoor1 = Slider(axcoor1, 'y coor ', 0, outsize[1]-1, valinit=coor1,valstep=delta_coor)
        scoor2 = Slider(axcoor2, 'z coor ', 0, outsize[2]-1, valinit=coor2,valstep=delta_coor)
        
        
    
        def update0(val):
            ax[0,0].cla()
            ax[0,0].xaxis.set_major_locator(plt.NullLocator())
            ax[0,0].yaxis.set_major_locator(plt.NullLocator())
            ax[0,0].set_xlabel('z')
            ax[0,0].set_ylabel('y')
            ax[1,0].cla()
            ax[1,0].xaxis.set_major_locator(plt.NullLocator())
            ax[1,0].yaxis.set_major_locator(plt.NullLocator())
            ax[1,0].set_xlabel('z')
            ax[1,0].set_ylabel('y')
            
            coor0 = scoor0.val.astype(int)
            
            ax[1,0].imshow(seg[coor0,:,:],cmap = cmap,vmin=0,vmax = 39)
            ax[0,0].imshow(inp[coor0,:,:],cmap = cmap,vmin=0,vmax = 39)
            
            if len(Ind)!=1:
                ind0 = Ind[Ind[:,0]==coor0]
                
                ax[0,0].scatter(ind0[:,2],ind0[:,1],s=1,marker='s')
                ax[1,0].scatter(ind0[:,2],ind0[:,1],s=1,marker='s')
                
            if len(Ind_r)!=1:
                ind0 = Ind_r[Ind_r[:,0]==coor0]
                
                ax[0,0].scatter(ind0[:,2],ind0[:,1],s=1,marker='s',c='r')
                ax[1,0].scatter(ind0[:,2],ind0[:,1],s=1,marker='s',c='g')
                
            if len(Ind_w)!=1:
                ind0 = Ind_w[Ind_w[:,0]==coor0]
                
                ax[0,0].scatter(ind0[:,2],ind0[:,1],s=1,marker='s',c='g')
                ax[1,0].scatter(ind0[:,2],ind0[:,1],s=1,marker='s',c='r')
            
        def update1(val):
            ax[0,1].cla()
            ax[0,1].xaxis.set_major_locator(plt.NullLocator())
            ax[0,1].yaxis.set_major_locator(plt.NullLocator())
            ax[0,1].set_xlabel('z')
            ax[0,1].set_ylabel('x')
            
            ax[1,1].cla()
            ax[1,1].xaxis.set_major_locator(plt.NullLocator())
            ax[1,1].yaxis.set_major_locator(plt.NullLocator())
            ax[1,1].set_xlabel('z')
            ax[1,1].set_ylabel('x')
            
            coor1 = scoor1.val.astype(int)
            
            ax[1,1].imshow(seg[:,coor1,:],cmap = cmap,vmin=0,vmax = 39)
            ax[0,1].imshow(inp[:,coor1,:],cmap = cmap,vmin=0,vmax = 39)
            
            if len(Ind)!=1:
                ind1 = Ind[Ind[:,1]==coor1]
                
                ax[0,1].scatter(ind1[:,2],ind1[:,0],s=1,marker='s')
                ax[1,1].scatter(ind1[:,2],ind1[:,0],s=1,marker='s')
                
            if len(Ind_r)!=1:
                ind1 = Ind_r[Ind_r[:,1]==coor1]
                
                ax[0,1].scatter(ind1[:,2],ind1[:,0],s=1,marker='s',c='r')
                ax[1,1].scatter(ind1[:,2],ind1[:,0],s=1,marker='s',c='g')
                
            if len(Ind_w)!=1:
                ind1 = Ind_w[Ind_w[:,1]==coor1]
                
                ax[0,1].scatter(ind1[:,2],ind1[:,0],s=1,marker='s',c='g')
                ax[1,1].scatter(ind1[:,2],ind1[:,0],s=1,marker='s',c='r')
            
        def update2(val):
            ax[0,2].cla()
            ax[0,2].xaxis.set_major_locator(plt.NullLocator())
            ax[0,2].yaxis.set_major_locator(plt.NullLocator())
            ax[0,2].set_xlabel('y')
            ax[0,2].set_ylabel('x')
            
            ax[1,2].cla()
            ax[1,2].xaxis.set_major_locator(plt.NullLocator())
            ax[1,2].yaxis.set_major_locator(plt.NullLocator())
            ax[1,2].set_xlabel('y')
            ax[1,2].set_ylabel('x')
            
            coor2 = scoor2.val.astype(int)
            
            ax[1,2].imshow(seg[:,:,coor2],cmap = cmap,vmin=0,vmax = 39)
            ax[0,2].imshow(inp[:,:,coor2],cmap = cmap,vmin=0,vmax = 39)
            
            if len(Ind)!=1:
                ind2 = Ind[Ind[:,2]==coor2]
                
                ax[0,2].scatter(ind2[:,1],ind2[:,0],s=1,marker='s')
                ax[1,2].scatter(ind2[:,1],ind2[:,0],s=1,marker='s')
                
            if len(Ind_r)!=1:
                ind2 = Ind_r[Ind_r[:,2]==coor2]
                
                ax[0,2].scatter(ind2[:,1],ind2[:,0],s=1,marker='s',c='r')
                ax[1,2].scatter(ind2[:,1],ind2[:,0],s=1,marker='s',c='g')
                
            if len(Ind_w)!=1:
                ind2 = Ind_w[Ind_w[:,2]==coor2]
                
                ax[0,2].scatter(ind2[:,1],ind2[:,0],s=1,marker='s',c='g')
                ax[1,2].scatter(ind2[:,1],ind2[:,0],s=1,marker='s',c='r')
            
            
        scoor0.on_changed(update0)
        scoor1.on_changed(update1)
        scoor2.on_changed(update2)
        
        
        
        plt.show()
    


def PlotGuiImg(data):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider, Button
    
    if len(data.shape)!=5:
        raise ValueError('Dimension of data must be 5')
        
    
    outsize = [data.shape[i] for i in range(2,len(data.shape))]
    cmap='jet'
    data = data*50/data.max()
    
    fig, ax = plt.subplots(nrows=1,ncols=3,)
    plt.subplots_adjust(left=0.1, bottom=0.25)
    
    coor0 = int(outsize[0]/2)
    coor1 = int(outsize[1]/2)
    coor2 = int(outsize[2]/2)
    
    delta_coor = 1
    
    
    ax[0].set_xlabel('z')
    ax[0].set_ylabel('y')
    ax[1].set_xlabel('z')
    ax[1].set_ylabel('x')
    ax[2].set_xlabel('y')
    ax[2].set_ylabel('x')
    
    # Removing axis
    for i in range(3):
        ax[i].xaxis.set_major_locator(plt.NullLocator())
        ax[i].yaxis.set_major_locator(plt.NullLocator())
    
    
    ax[0].imshow(data[0,0,coor0,:,:],cmap=cmap,vmin = data.min(),vmax=data.max())
    ax[1].imshow(data[0,0,:,coor1,:],cmap=cmap,vmin = data.min(),vmax=data.max())
    ax[2].imshow(data[0,0,:,:,coor2],cmap=cmap,vmin = data.min(),vmax=data.max())

    
    axcolor = 'lightgoldenrodyellow'
    axcoor2 = plt.axes([0.15, 0.05, 0.65, 0.03], facecolor=axcolor)
    axcoor1 = plt.axes([0.15, 0.10, 0.65, 0.03], facecolor=axcolor)
    axcoor0 = plt.axes([0.15, 0.15, 0.65, 0.03], facecolor=axcolor)
    
    scoor0 = Slider(axcoor0, 'x coor', 0, outsize[0]-1, valinit=coor0, valstep=delta_coor)
    scoor1 = Slider(axcoor1, 'y coor ', 0, outsize[1]-1, valinit=coor1,valstep=delta_coor)
    scoor2 = Slider(axcoor2, 'z coor ', 0, outsize[2]-1, valinit=coor2,valstep=delta_coor)
    
    
    def update0(val):
        coor0 = scoor0.val.astype(int)
        ax[0].imshow(data[0,0,coor0,:,:],cmap='jet',vmin = data.min(),vmax=data.max())
        
        
    def update1(val):
        coor1 = scoor1.val.astype(int)
        ax[1].imshow(data[0,0,:,coor1,:],cmap='jet',vmin = data.min(),vmax=data.max())
        
    def update2(val):
        coor2 = scoor2.val.astype(int)
        ax[2].imshow(data[0,0,:,:,coor2],cmap='jet',vmin = data.min(),vmax=data.max())
    
        
    scoor0.on_changed(update0)
    scoor1.on_changed(update1)
    scoor2.on_changed(update2)
    
    
    
    plt.show()    



