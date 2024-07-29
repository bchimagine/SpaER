import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import random 
import numpy as np

import custom_image3d as ci3d
from SitkDataSet import SitkDataset as SData
import SimpleITK as sitk
import rxfm_net
from pytorch3d import transforms as pt3d_xfms
import time

# Load the saved model checkpoint
checkpoint = torch.load('./best_model_29.pth')
json_file_val = './data_squ_test.json'
dataset_val = SData(json_file_val, "test")

'''Set device (GPU or CPU)'''
dev = "cuda" if torch.cuda.is_available() else "cpu"
print("Loading data on:", dev)
'''Create a DataLoader'''
valloader = DataLoader(dataset_val, batch_size= 1, shuffle=False)

IMG_SIZE = [96,96,96]
n_conv_chan = 1
n_chan = 64
overfit = True 
low_val = 1
high_val = 5
trans_arr = np.zeros(len(valloader))
angular_arr =  np.zeros(len(valloader))

net_obj = rxfm_net.RXFM_Net_Wrapper(IMG_SIZE[0:3], n_chan, masks_as_input=False)
net_obj.load_state_dict(checkpoint['eq_tracking_state_dict'])
net_obj = net_obj.to(dev)
net_obj.eval()

'''Testing'''        

  

for idx, image_data in enumerate(valloader):
    source, temp=image_data
    b = source.shape[0]
    source = source.to(dev).float() 
    #target = temp.to(dev).float() 
    rx_val = random.randint(low_val, high_val)
    ry_val = random.randint(low_val, high_val)
    rz_val = random.randint(low_val, high_val)

    tx_val = random.randint(low_val, high_val)
    ty_val = random.randint(low_val, high_val)
    tz_val = random.randint(low_val, high_val)
    print ("rotation x:", rx_val, "rotation y:", ry_val, "rotation z:", rz_val, "translation x:", tx_val, "translation y:", ty_val, "translation z:", tz_val)
    mat = ci3d.create_transform(
        rx=rx_val, ry=ry_val, rz=rz_val,
        tx=2.0*tx_val/IMG_SIZE[0], ty=2.0*ty_val/IMG_SIZE[1], tz=2.0*tz_val/IMG_SIZE[2]
    )

    mat = mat[np.newaxis,:,:]
    mat = mat[:,0:3,:]
    mat = torch.tensor(mat).float()


    grids = torch.nn.functional.affine_grid(mat, [1,1] + IMG_SIZE).to(dev)
    target = torch.nn.functional.grid_sample(source, grids, mode="bilinear",padding_mode='border',align_corners=False)
    # # Start the timer
    start_time = time.time()
    xfm_1to2 = net_obj.forward((source,target))
    # Stop the timer
    end_time = time.time()
    # Calculate the elapsed time
    elapsed_time = end_time - start_time



    predicted_grids = torch.nn.functional.affine_grid(xfm_1to2, [1,1] + IMG_SIZE)
    x_aligned = F.grid_sample(source,
                              grid=predicted_grids,
                              mode='bilinear',
                              padding_mode='border',
                              align_corners=False)

    '''Print angular and tanslational error with or without deformable model '''
    '''To be implemented'''
    print ("epoch: ", epoch, " batch loss: ", running_loss)
    running_loss = 0.0

    rot_real = mat[:,0:3,0:3].detach().cpu().numpy()
    trans_real = mat[:,0:3,3:].detach().cpu().numpy()
    
    rot_approx = xfm_1to2[:,0:3,0:3].detach().cpu().numpy()
    trans_approx = xfm_1to2[:,0:3,3:].detach().cpu().numpy()
    
    angles_real = pt3d_xfms.matrix_to_euler_angles(torch.tensor(rot_real[0,:,:]), convention="XYZ")
    angles_approx = pt3d_xfms.matrix_to_euler_angles(torch.tensor(rot_approx[0,:,:]), convention="XYZ")
    
    angular_error = np.rad2deg(np.mean(np.abs(np.array(angles_real) - np.array(angles_approx))))
    translation_error = np.linalg.norm(trans_real - trans_approx)
    trans_arr[idx] = translation_error
    angular_arr [idx] = angular_error
    # print("angular abs. error (mean degrees)", np.rad2deg(np.mean(np.abs(np.array(angles_real) - np.array(angles_approx)))))
    # print("trans error", np.linalg.norm(trans_real - trans_approx))
    
    if (idx % 1 == 0 ):
        saved= sitk.GetImageFromArray(np.array(source[0,0,:,:,:].detach().cpu()))
        save_name = './check_deform_anything/source_' +  str(idx) + '.nii.gz'
        sitk.WriteImage(saved, save_name)
        
        saved= sitk.GetImageFromArray(np.array(target[0,0,:,:,:].detach().cpu()))
        save_name = './check_deform_anything/target_' +  str(idx) + '.nii.gz'
        sitk.WriteImage(saved, save_name)

        saved= sitk.GetImageFromArray(np.array(x_aligned[0,0,:,:,:].detach().cpu()))
        save_name = './check_deform_anything/rigid_' +  str(idx) + '.nii.gz'
        sitk.WriteImage(saved, save_name)


if (para.model.deformable == True):
    save_name_trans = "trans_error_withdef_" + str(epoch) + ".npy"
    save_name_angular = "angular_error_withdef_" + str(epoch) + ".npy"
else:
    save_name_trans = "trans_error_withoutdef_" + str(epoch) + ".npy"
    save_name_angular = "angular_error_withoutdef_" + str(epoch) + ".npy"

np.save(save_name_trans, trans_arr)
np.save(save_name_angular, angular_arr)
print ("validation loss:", total_val)  