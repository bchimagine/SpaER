import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, Dataset, DataLoader
from torch.autograd import grad
from torch.autograd import Variable
import torch.nn.functional as F
import SimpleITK as sitk
import os, glob
import sys
import random
from torch.optim.lr_scheduler import CosineAnnealingLR
from Diffeo_losses import NCC, MSE, Grad
from Diffeo_networks import DiffeoDense  
from SitkDataSet import SitkDataset as SData
from tools import ReadFiles as rd
from functools import partial
import utils
import losses
import custom_image3d as ci3d
import rxfm_net
import SimpleITK as sitk 
import time
from pytorch3d import transforms as pt3d_xfms
from RMI import RMILoss



'''Read parameters by yaml'''
para = rd.read_yaml('./parameters.yml')

''' Load data by json'''
json_file = './data_norm_squ_train.json'
json_file_val = './data_norm_squ_val.json'
batch_size = para.solver.batch_size
dataset = SData(json_file, "train")

dataset_val = SData(json_file_val, "val")

'''Set device (GPU or CPU)'''
dev = "cuda" if torch.cuda.is_available() else "cpu"
print("Loading data on:", dev)

'''Create a DataLoader'''
trainloader = DataLoader(dataset, batch_size= batch_size, shuffle=False)
valloader = DataLoader(dataset_val, batch_size= 1, shuffle=False)



batch_size = para.solver.batch_size
IMG_SIZE = [96,96,96]
loss_func_name = "xfm_6D"

n_conv_chan = 1
n_chan = 64
overfit = True 
running_loss = 0 
def_weight = para.solver.def_weight
LR = 0.000025
LR_def = 0.001
low = 1
high = 20
seq_len = 5
d_model = 32*3
attention_weight = 0.006

net_obj = rxfm_net.RXFM_Net_Wrapper(IMG_SIZE[0:3], n_chan, masks_as_input=False)
# print (net_obj)
rmi_loss = RMILoss(with_logits=True, radius=11, bce_weight=0.5, downsampling_method='max', stride=2, use_log_trace=True, use_double_precision=True)


if loss_func_name == "xfm_MSE":
    loss_func = partial( losses.xfm_loss_MSE, weight_R=1.0, weight_T=5.0)
elif loss_func_name == "xfm_6D":
    loss_func = partial( losses.xfm_loss_6D, weight_R=1.0, weight_T=5.0)
else:
    print("Loss function not recognized")
    exit(1)
dice_func = partial(losses.dice_loss, hard=False, ign_first_ch=False)
shape_func = partial(losses.curva_loss)
net_obj = net_obj.to(dev)
# Set different learning rates for each network

optimizer = torch.optim.Adam(net_obj.parameters(), LR)
scheduler = CosineAnnealingLR(optimizer, T_max=len(trainloader), eta_min=0.00001)
if (para.model.deformable == True):
    Diffeo_net = DiffeoDense(inshape=(IMG_SIZE[0], IMG_SIZE[1], IMG_SIZE[2]),
                      nb_unet_features=[[16, 32,], [32, 32, 16, 16]],
                      nb_unet_conv_per_level=1,
                      int_steps=7,
                      int_downsize=2,
                      src_feats=1,
                      trg_feats=1,
                      unet_half_res=True,
                      velocity_tag = 0)
    diff_net = Diffeo_net.to(dev)
    optimizer = torch.optim.Adam([
        {'params': net_obj.parameters(), 'lr': LR},
        {'params': diff_net.parameters(), 'lr': LR_def}
    ], lr=0.01) 
    criterion = nn.MSELoss()


# Define the optimizer and specify different learning rates for each network

# def calculate_errors(mat, xfm_1to2):
#     # Extract rotation and translation components
#     rot_real = mat[:, 0:3, 0:3].detach().cpu().numpy()
#     trans_real = mat[:, 0:3, 3:].detach().cpu().numpy()

#     rot_approx = xfm_1to2[:, 0:3, 0:3].detach().cpu().numpy()
#     trans_approx = xfm_1to2[:, 0:3, 3:].detach().cpu().numpy()

#     # Convert rotation matrices to Euler angles
#     angles_real = pt3d_xfms.matrix_to_euler_angles(torch.tensor(rot_real[0, :, :]), convention="XYZ")
#     angles_approx = pt3d_xfms.matrix_to_euler_angles(torch.tensor(rot_approx[0, :, :]), convention="XYZ")
#     # Calculate angular error
#     angular_error = np.rad2deg(np.mean(np.abs(np.array(angles_real) - np.array(angles_approx))))
#     # Calculate translation error
#     translation_error = np.linalg.norm(trans_real - trans_approx)

#     return angular_error, translation_error



for epoch in range(para.solver.epochs):
    total= 0; 
    total_val =0; 
    print('epoch:', epoch)
    for idx, image_data in enumerate(trainloader):
   
        source, temp=image_data
        b = source.shape[0]    
        source = source.to(dev).float() 
        temp =temp.to(dev).float() 
        idendity = ci3d.create_transform(
            rx=0, ry=0, rz=0,
            tx=0, ty=0, tz=0
        )

        idendity = idendity[np.newaxis,:,:]
        idendity = idendity[:,0:3,:]
        idendity = torch.tensor(idendity).float()
 
        # print (source.shape)
        optimizer.zero_grad()   
        '''@@@@@@@@ Generating semi-simulation motions on data from different time points @@@@@@@@@@'''
        rx_train = random.randint(low, high)
        ry_train = random.randint(low, high)
        rz_train = random.randint(low, high)

        tx_train = random.randint(low, high)
        ty_train = random.randint(low, high)
        tz_train = random.randint(low, high)
        sequence = [source]
        mat_seq = [idendity]
        for time_step in range (0, seq_len):

            new_rx = (rx_train*(time_step+1))/seq_len; new_ry = (ry_train*(time_step+1))/seq_len; new_rz = (rz_train*(time_step+1))/seq_len; 
            new_tx= (tx_train*(time_step+1))/seq_len; new_ty= (ty_train*(time_step+1))/seq_len; new_tz= (tz_train*(time_step+1))/seq_len; 

            mat = ci3d.create_transform(
                rx=new_rx, ry=new_ry, rz=new_rz,
                tx=2.0*new_tx/IMG_SIZE[0], ty=2.0*new_ty/IMG_SIZE[1], tz=2.0*new_tz/IMG_SIZE[2]
            )

            mat = mat[np.newaxis,:,:]
            mat = mat[:,0:3,:]
            mat = torch.tensor(mat).float()
            grids = torch.nn.functional.affine_grid(mat, [1,1] + IMG_SIZE).to(dev)
            target = torch.nn.functional.grid_sample(source, grids, mode="bilinear",padding_mode='border',align_corners=True)
            sequence.append(target)
            mat_seq.append(mat)
        for k in range (0, len(sequence)-1):

            k= torch.tensor(k).to(dev)
            xfm_1to2 = net_obj.forward((sequence[k],sequence[k+1]), k, k+1, attention_weight)
            predicted_grids = torch.nn.functional.affine_grid(xfm_1to2, [1,1] + IMG_SIZE)
            x_aligned = F.grid_sample(sequence[k],
                                  grid=predicted_grids,
                                  mode='bilinear',
                                  padding_mode='border',
                                  align_corners=True)
            #loss_val = loss_func(mat_seq[k+1].to(dev), xfm_1to2 )
            loss_val = NCC().loss(sequence[k+1], x_aligned)
            # loss_val = loss_image
            if (abs(loss_val) > 10e-5):
                loss_val.backward(retain_graph=True)


    for idx, image_data in enumerate(valloader):
   
        source, temp=image_data
        b = source.shape[0]    
        source = source.to(dev).float() 
        temp =temp.to(dev).float() 
        idendity = ci3d.create_transform(
            rx=0, ry=0, rz=0,
            tx=0, ty=0, tz=0
        )

        idendity = idendity[np.newaxis,:,:]
        idendity = idendity[:,0:3,:]
        idendity = torch.tensor(idendity).float()
 
        # print (source.shape)
        optimizer.zero_grad()   
        '''@@@@@@@@ Generating semi-simulation motions on data from different time points @@@@@@@@@@'''
        rx_train = random.randint(low, high)
        ry_train = random.randint(low, high)
        rz_train = random.randint(low, high)

        tx_train = random.randint(low, high)
        ty_train = random.randint(low, high)
        tz_train = random.randint(low, high)

        '''Create sequence'''
        sequence = [source]
        mat_seq = [idendity]
        for time_step in range (0, seq_len):

            new_rx = (rx_train*(time_step+1))/seq_len; new_ry = (ry_train*(time_step+1))/seq_len; new_rz = (rz_train*(time_step+1))/seq_len; 
            new_tx= (tx_train*(time_step+1))/seq_len; new_ty= (ty_train*(time_step+1))/seq_len; new_tz= (tz_train*(time_step+1))/seq_len; 

            mat = ci3d.create_transform(
                rx=new_rx, ry=new_ry, rz=new_rz,
                tx=2.0*new_tx/IMG_SIZE[0], ty=2.0*new_ty/IMG_SIZE[1], tz=2.0*new_tz/IMG_SIZE[2]
            )

            mat = mat[np.newaxis,:,:]
            mat = mat[:,0:3,:]
            mat = torch.tensor(mat).float()
            grids = torch.nn.functional.affine_grid(mat, [1,1] + IMG_SIZE).to(dev)
            target = torch.nn.functional.grid_sample(source, grids, mode="bilinear",padding_mode='border',align_corners=True)
            sequence.append(target)
            mat_seq.append(mat)
        for k in range (0, len(sequence)-1):

            k= torch.tensor(k).to(dev)
            # # Start the timer
            start_time = time.time()
            # Stop the timer
            xfm_1to2 = net_obj.forward((sequence[k],sequence[k+1]), k, k+1, attention_weight)
            end_time = time.time()
            elapsed_time = end_time - start_time

            print ("Motion Estimation Time:", elapsed_time)
            predicted_grids = torch.nn.functional.affine_grid(xfm_1to2, [1,1] + IMG_SIZE)

            x_aligned = F.grid_sample(sequence[k],
                                  grid=predicted_grids,
                                  mode='bilinear',
                                  padding_mode='border',
                                  align_corners=True)

            if (idx%10 ==0):
                saved= sitk.GetImageFromArray(np.array(sequence[k][0,0,:,:,:].detach().cpu()))
                save_name = './check_result/source_' + str(epoch) + '_'+ str(idx) + '_' + str(k.item()) + '.nii.gz'
                sitk.WriteImage(saved, save_name)
                
                saved= sitk.GetImageFromArray(np.array(sequence[k+1][0,0,:,:,:].detach().cpu()))
                save_name = './check_result/target_' + str(epoch) + '_'+ str(idx) + '_' + str(k.item()) + '.nii.gz'
                sitk.WriteImage(saved, save_name)
                
                saved= sitk.GetImageFromArray(np.array(x_aligned[0,0,:,:,:].detach().cpu()))
                save_name = './check_result/aligned_' + str(epoch) + '_'+ str(idx) + '_' + str(k.item()) + '.nii.gz'
                sitk.WriteImage(saved, save_name)

