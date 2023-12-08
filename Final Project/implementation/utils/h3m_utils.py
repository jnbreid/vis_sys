import torch
import numpy as np


def expmap2rotmat_torch(r, device = 'cpu'):
    """
    Converts expmap matrix to rotation
    batch pytorch version ported from the corresponding method above
    :param r: N*3
    :return: N*3*3
    """
    theta = torch.norm(r, 2, 1)
    r0 = torch.div(r, theta.unsqueeze(1).repeat(1, 3) + 0.0000001)
    r1 = torch.zeros_like(r0).repeat(1, 3)
    r1[:, 1] = -r0[:, 2]
    r1[:, 2] = r0[:, 1]
    r1[:, 5] = -r0[:, 0]
    r1 = r1.view(-1, 3, 3)
    r1 = r1 - r1.transpose(1, 2)
    n = r1.data.shape[0]
    R = torch.eye(3, 3).repeat(n, 1, 1).float().to(device) + torch.mul(
        torch.sin(theta).unsqueeze(1).repeat(1, 9).view(-1, 3, 3), r1) + torch.mul(
        (1 - torch.cos(theta).unsqueeze(1).repeat(1, 9).view(-1, 3, 3)), torch.matmul(r1, r1))
    return R


def rotmat2euler_torch(R, device = 'cpu'):
    """
    Converts a rotation matrix to euler angles
    batch pytorch version ported from the corresponding numpy method above
    :param R:N*3*3
    :return: N*3
    """
    n = R.data.shape[0]
    eul = torch.zeros(n, 3).float().to(device)
    idx_spec1 = (R[:, 0, 2] == 1).nonzero().cpu().data.numpy().reshape(-1).tolist()
    idx_spec2 = (R[:, 0, 2] == -1).nonzero().cpu().data.numpy().reshape(-1).tolist()
    if len(idx_spec1) > 0:
        R_spec1 = R[idx_spec1, :, :]
        eul_spec1 = torch.zeros(len(idx_spec1), 3).float().cuda()
        eul_spec1[:, 2] = 0
        eul_spec1[:, 1] = -np.pi / 2
        delta = torch.atan2(R_spec1[:, 0, 1], R_spec1[:, 0, 2])
        eul_spec1[:, 0] = delta
        eul[idx_spec1, :] = eul_spec1

    if len(idx_spec2) > 0:
        R_spec2 = R[idx_spec2, :, :]
        eul_spec2 = torch.zeros(len(idx_spec2), 3).float().to(device)
        eul_spec2[:, 2] = 0
        eul_spec2[:, 1] = np.pi / 2
        delta = torch.atan2(R_spec2[:, 0, 1], R_spec2[:, 0, 2])
        eul_spec2[:, 0] = delta
        eul[idx_spec2] = eul_spec2

    idx_remain = np.arange(0, n)
    idx_remain = np.setdiff1d(np.setdiff1d(idx_remain, idx_spec1), idx_spec2).tolist()
    if len(idx_remain) > 0:
        R_remain = R[idx_remain, :, :]
        eul_remain = torch.zeros(len(idx_remain), 3).float().cpu()#.to(device)
        eul_remain[:, 1] = -torch.asin(R_remain[:, 0, 2])
        eul_remain[:, 0] = torch.atan2(R_remain[:, 1, 2] / torch.cos(eul_remain[:, 1]),
                                       R_remain[:, 2, 2] / torch.cos(eul_remain[:, 1]))
        eul_remain[:, 2] = torch.atan2(R_remain[:, 0, 1] / torch.cos(eul_remain[:, 1]),
                                       R_remain[:, 0, 0] / torch.cos(eul_remain[:, 1]))
        eul[idx_remain, :] = eul_remain

    return eul

"""
function to convert angle coordinates to 3d euklidean space. 
the code is adapted from https://github.com/MotionMLP/MotionMixer/blob/main/utils/forward_kinematics.py with slight changes
"""
def fkl_torch(angles, parent, offset, device = 'cpu'):#, rotInd, expmapInd):
    """
    pytorch version of fkl.

    convert joint angles to joint locations
    batch pytorch version of the fkl() method above
    :param angles: N*99
    :param parent:
    :param offset:
    :param rotInd:
    :param expmapInd:
    :return: N*joint_n*3
    """

    n = angles.data.shape[0]
    j_n = offset.shape[0]
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    p3d = torch.from_numpy(offset).float().to(device).unsqueeze(0).repeat(n, 1, 1)
    angles = angles[:, 3:].contiguous().view(-1, 3)

    R = expmap2rotmat_torch(angles, device).view(n, j_n, 3, 3)
    for i in np.arange(1, j_n):
        if parent[i] > 0:
            R[:, i, :, :] = torch.matmul(R[:, i, :, :], R[:, parent[i], :, :]).clone()
            p3d[:, i, :] = torch.matmul(p3d[0, i, :], R[:, parent[i], :, :]) + p3d[:, parent[i], :]

    return p3d




"""
function to calculate the new offset array. only the offset distances of bones between joints that are used are important.
The original offset array from https://github.com/MotionMLP/MotionMixer/blob/main/utils/forward_kinematics.py is used below

in the actual implementation where euklidean 3d coordinates are calculated the output offset array from this function is used.
to decrease runtime the offset array is hardcoded directly, so that the array operations below do not need to be executed for each pose individually
"""
def calc_offset():
  # use the 'old' offset array from the original implementation of MotionMixer
  offset = np.array(
        [0.000000, 0.000000, 0.000000, -132.948591, 0.000000, 0.000000, 0.000000, -442.894612, 0.000000, 0.000000,
         -454.206447, 0.000000, 0.000000, 0.000000, 162.767078, 0.000000, 0.000000, 74.999437, 132.948826, 0.000000,
         0.000000, 0.000000, -442.894413, 0.000000, 0.000000, -454.206590, 0.000000, 0.000000, 0.000000, 162.767426,
         0.000000, 0.000000, 74.999948, 0.000000, 0.100000, 0.000000, 0.000000, 233.383263, 0.000000, 0.000000,
         257.077681, 0.000000, 0.000000, 121.134938, 0.000000, 0.000000, 115.002227, 0.000000, 0.000000, 257.077681,
         0.000000, 0.000000, 151.034226, 0.000000, 0.000000, 278.882773, 0.000000, 0.000000, 251.733451, 0.000000,
         0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 99.999627, 0.000000, 100.000188, 0.000000, 0.000000,
         0.000000, 0.000000, 0.000000, 257.077681, 0.000000, 0.000000, 151.031437, 0.000000, 0.000000, 278.892924,
         0.000000, 0.000000, 251.728680, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 99.999888,
         0.000000, 137.499922, 0.000000, 0.000000, 0.000000, 0.000000])
  offset = offset.reshape(-1, 3)
  # but reduce it to the offset actually needed for 17 joints according to the list of indices below
  offset_keep = np.array([0,1,2,3,6,7,8,11,12,13,14,15,16,17,18,19,24,25,26,27])
  new_offset = offset[offset_keep]
  return new_offset


"""
function to transform angle representation of vector to representation in 3d euklid space

input must have dimensions (n_frames, 51)

for batched data each element must be calculated seperately
"""
def angles_to3dspace(angles, device):
  n_frames, n_features = angles.shape
  cal_angles = angles.view(n_frames, -1, 3)
  n_joints = cal_angles.shape[1]

  if n_joints == 16:  # exception if root joint is not part of coordinates
    attach = torch.zeros(n_frames,1,3).to(device)
    cal_angles = torch.cat([attach,cal_angles], dim = 1)

  full_angles = torch.zeros(n_frames,21,3).to(device)

  full_angles[:,:7,:] = cal_angles[:,:7,:]
  full_angles[:,8:12,:] = cal_angles[:,7:11,:]
  full_angles[:,13:16,:] = cal_angles[:,11:14,:]
  full_angles[:,17:20,:] = cal_angles[:,14:17,:]

  new_angles = full_angles.view(n_frames, -1)

  new_parents = np.array([-1,0,1,2,0,4,5,0,7,8,9,10,8,12,13,14,8,16,17,18]) # new parent array suited for 17 joint vectors

  # new offset is calculated by function 'calc_offset' by only picking the offset which will be used for the 17 joints
  # here we use the hardcoded version to make it a little faster
  new_offset = np.array([[ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
                              [-1.32948591e+02,  0.00000000e+00,  0.00000000e+00],
                              [ 0.00000000e+00, -4.42894612e+02,  0.00000000e+00],
                              [ 0.00000000e+00, -4.54206447e+02,  0.00000000e+00],
                              [ 1.32948826e+02,  0.00000000e+00,  0.00000000e+00],
                              [ 0.00000000e+00, -4.42894413e+02,  0.00000000e+00],
                              [ 0.00000000e+00, -4.54206590e+02,  0.00000000e+00],
                              [ 0.00000000e+00,  1.00000000e-01,  0.00000000e+00],
                              [ 0.00000000e+00,  2.33383263e+02,  0.00000000e+00],
                              [ 0.00000000e+00,  2.57077681e+02,  0.00000000e+00],
                              [ 0.00000000e+00,  1.21134938e+02,  0.00000000e+00],
                              [ 0.00000000e+00,  1.15002227e+02,  0.00000000e+00],
                              [ 0.00000000e+00,  2.57077681e+02,  0.00000000e+00],
                              [ 0.00000000e+00,  1.51034226e+02,  0.00000000e+00],
                              [ 0.00000000e+00,  2.78882773e+02,  0.00000000e+00],
                              [ 0.00000000e+00,  2.51733451e+02,  0.00000000e+00],
                              [ 0.00000000e+00,  2.57077681e+02,  0.00000000e+00],
                              [ 0.00000000e+00,  1.51031437e+02,  0.00000000e+00],
                              [ 0.00000000e+00,  2.78892924e+02,  0.00000000e+00],
                              [ 0.00000000e+00,  2.51728680e+02,  0.00000000e+00]])

  xyz = fkl_torch(new_angles, new_parents, new_offset)
  to_keep = [0,1,2,3,4,5,6,8,9,10,11,13,14,15,17,18,19]
  xyz = xyz[:,to_keep,:]

  return xyz