import numpy as np
import cv2
import copy
import torch

from utils.h3m_utils import angles_to3dspace, expmap2rotmat_torch, rotmat2euler_torch


### norm by right thigh  ###### -4.42894413e+02
def pck(predictions, targets, thresh = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3]):
    """
    Percentage of correct keypoints.
    Args:
        predictions: np array of predicted 3D joint positions in format (..., n_joints, 3)
        targets: np array of same shape as `predictions`
        thresh: radius within which a predicted joint has to lie.

    Returns:
        Percentage of correct keypoints at the given threshold level, stored in a np array of shape (..., len(threshs))
    
    code taken from https://github.com/eth-ait/motion-transformer/blob/master/metrics/motion_metrics.py and adapted

    """
    # value of the right thigh is used to normalize the lenghts of the bones
    norm_factor = 4.42894413e+02

    predictions = predictions / norm_factor
    targets = targets / norm_factor

    dist = np.sqrt(np.sum((predictions - targets) ** 2, axis=-1))
    bool_array = np.zeros((dist.shape[0],17,len(thresh)), dtype = np.float32)

    for i in range(len(thresh)):
      for j in range(dist.shape[0]):
        for k in range(17):
          if dist[j,k] <= thresh[i]:
            bool_array[j,k,i] = 1

    pck_ = np.mean(bool_array, axis=(0,1))
    return pck_

"""
function to calculate the auc for an function where y_val is a 1 dim array containing the
x values and y_val is an array of identical size containing the y values of the function

parameter norm decides if the range of the x values is scaled to length 1 or stays the input range.
Scaling the lenght to 1 is done to facilitate comparing auc values for differeny x value ranges.
by default x values get scaled. This is also the case when calculating the pck_auc value
"""
def auc_trapez(y_val, x_val = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3], norm = True): 
  auc_sum = 0
  norm_factor = x_val[-1] - x_val[0]
  for i in range(len(x_val)-1):
    width = x_val[i+1] - x_val[i]
    min_val = min(y_val[i+1], y_val[i])
    max_val = max(y_val[i+1], y_val[i])

    auc_part = min_val * width + ((max_val - min_val)*width / 2)
    auc_sum = auc_sum +  auc_part

  if norm == True:
    return auc_sum / norm_factor

  return auc_sum



def positional(predictions, targets):
    """
    Computes the Euclidean distance between joints in 3D space.
    Args:
        predictions: np array of predicted 3D joint positions in format (..., n_joints, 3)
        targets: np array of same shape as `predictions`

    Returns:
        The Euclidean distance for each joint as an np array of shape (..., n_joints)

     code taken from https://github.com/eth-ait/motion-transformer/blob/master/metrics/motion_metrics.py
    """
    return np.sqrt(np.sum((predictions - targets) ** 2, axis=-1))


import numpy as np
import cv2
import copy


def angle_diff(predictions, targets):
    """
    Computes the angular distance between the target and predicted rotations. We define this as the angle that is
    required to rotate one rotation into the other. This essentially computes || log(R_diff) || where R_diff is the
    difference rotation between prediction and target.

    Args:
        predictions: np array of predicted joint angles represented as rotation matrices, i.e. in shape
          (..., n_joints, 3, 3)
        targets: np array of same shape as `predictions`

    Returns:
        The geodesic distance for each joint as an np array of shape (..., n_joints)

    code taken from 
    """
    assert predictions.shape[-1] == predictions.shape[-2] == 3
    assert targets.shape[-1] == targets.shape[-2] == 3

    ori_shape = predictions.shape[:-2]
    preds = np.reshape(predictions, [-1, 3, 3])
    targs = np.reshape(targets, [-1, 3, 3])

    # compute R1 * R2.T, if prediction and target match, this will be the identity matrix
    r = np.matmul(preds, np.transpose(targs, [0, 2, 1]))

    # convert `r` to angle-axis representation and extract the angle, which is our measure of difference between
    # the predicted and target orientations
    angles = []
    for i in range(r.shape[0]):
        aa, _ = cv2.Rodrigues(r[i].numpy())
        angles.append(np.linalg.norm(aa))
    angles = np.array(angles)

    return np.reshape(angles, ori_shape)


def euler_error(ang_pred, ang_gt):
    """
    function to calculate the euler error 
    code taken from https://github.com/eth-ait/motion-transformer/blob/master/metrics/motion_metrics.py and adapted
    """
    dim_full_len=ang_gt.shape[2]

    pred_expmap = ang_pred.contiguous().view(-1,dim_full_len).view(-1, 3)
    targ_expmap = ang_gt.contiguous().view(-1,dim_full_len).view(-1, 3)

    pred_eul = rotmat2euler_torch(expmap2rotmat_torch(pred_expmap))
    pred_eul = pred_eul.view(-1, dim_full_len)

    targ_eul = rotmat2euler_torch(expmap2rotmat_torch(targ_expmap))
    targ_eul = targ_eul.view(-1, dim_full_len)
    mean_errors = torch.mean(torch.norm(pred_eul - targ_eul, 2, 1))

    return mean_errors




def eval_model(model, data_loader, criterion, device, input = 'angle', loss_only = False, output_len = 10):
  """
  function to evaluate an input model with a given dataloader.
  If the input data is in exponential map format 'input' must be set to 'angle', if it is 3D euclidean data it must be 'coord_3d'
  For fast evaluation where only the loss is calculated 'loss_only' must be set to True
  """
  with torch.no_grad():
      model.eval()
      
      loss_list = []

      geodesic_list = []
      pck_list = []
      eulerangle_list = []
      mpjpe_list = []
      
      for seed_frames, target_frames in data_loader:
          seed_frames = seed_frames.to(device)
          target_frames = target_frames.to(device)
          
          if output_len == 10:
            outputs = model(seed_frames)
          else:
             rep = int(np.ceil(output_len/10))
             outputs = seed_frames
             out_list = []
             for i in range(rep):
                outputs = model(outputs)
                out_list.append(outputs)
             outputs = torch.cat(out_list, dim = 1)[:,:output_len,:]
          
                  
          loss = criterion(outputs, target_frames)
          loss_list.append(loss.item())
          #print('after_loss')
          if loss_only == True:
             geodesic_list.append(0)
             pck_list.append(0)
             eulerangle_list.append(0)
             mpjpe_list.append(0)
             continue

          target_frames = target_frames.cpu()
          outputs = outputs.cpu()

          if input == 'angle':
              # geodesic
              for i in range(target_frames.shape[0]):
                rotmatTarget = expmap2rotmat_torch(target_frames[i,:,3:].contiguous().view(-1, 3)).view(-1, int(target_frames[i,:,3:].shape[-1]/3), 3, 3).cpu()
                rotmatOut = expmap2rotmat_torch(outputs[i,:,3:].contiguous().view(-1, 3)).view(-1, outputs.shape[1], 3, 3).cpu()
                err = angle_diff(rotmatOut, rotmatTarget)
                geodesic_list.append(err)
                # eulerangle
              err = euler_error(target_frames[:,:,3:], outputs[:,:,3:])
              eulerangle_list.append(err)

              for i in range(target_frames.shape[0]):
                xyz_Target = np.asarray(angles_to3dspace(target_frames[i,:,:], 'cpu'))
                xyz_Out = np.asarray(angles_to3dspace(outputs[i,:,:], 'cpu'))

                pck_list.append(pck(xyz_Out, xyz_Target))
                mpjpe_list.append(positional(xyz_Out, xyz_Target).mean(axis=1))

          elif input == 'coord_3d':
             b_size, n_frames, n_features = outputs.shape
             outputs = np.asarray(outputs.view(b_size, n_frames, -1, 3))
             target_frames = np.asarray(target_frames.view(b_size, n_frames, -1, 3))

             for i in range(target_frames.shape[0]):             
                pck_list.append(pck(outputs[i,:,:], target_frames[i,:,:]))
                mpjpe_list.append(positional(outputs[i,:,:], target_frames[i,:,:]).mean(axis=1))

                

              
      loss = np.mean(loss_list)

      geodesic = 0
      eulerangle = 0
      pck_auc = 0
      mpjpe = 0

      if loss_only != True:
        if input == 'angle':
          full_geodesic = np.stack(geodesic_list)
          geodesic = full_geodesic.mean()
          full_eulerangle = np.stack(eulerangle_list)
          eulerangle = full_eulerangle.mean()

          pck_out = np.asarray(pck_list).mean(axis = 0)
          pck_auc = auc_trapez(pck_out)
          mpjpe = np.asarray(mpjpe_list).mean()
        elif input == 'coord_3d':
          pck_out = np.asarray(pck_list).mean(axis = 0)
          pck_auc = auc_trapez(pck_out)
          mpjpe = np.asarray(mpjpe_list).mean()

      return loss, geodesic, eulerangle, pck_auc, mpjpe
        
