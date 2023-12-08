import torch
import numpy as np
import os
from torch.utils.data import Dataset

from utils.h3m_utils import fkl_torch, angles_to3dspace #fkl_torch_fullrange


"""
function which defines the indices of the 17 joints that are used.
input:  tensor from the original dataset with 33 joints of size (n_frames, 99)
output: tensor of size (n_frames, 51)
"""
def keep_joints(angles):
  ind_keep = [0,1,2,3,6,7,8,12,13,14,15,17,18,19,25,26,27]
  n_frames, vect_len = angles.shape
  res_angles = angles.view(n_frames, -1, 3)

  reduced_angles = res_angles[:,ind_keep,:]
  reduced_angles = reduced_angles.view(n_frames, -1)

  return reduced_angles



class Human36(Dataset):
  """
  Dataset class for the modified human3.6m dataset 
  Data can either be loaded in exponential map format (angles between joints)
  or in coord_3d format, so where each joint is represented by a 3d point in space.
  """
  def __init__(self, data_path, d_mode = 'train', data_format = 'exp_map', seed_length = 10, prediction_length = 10): # predict 10 frames from the first 10 frames (so 20 frames overall)
    self.data_path = data_path # must be the path to h3.6m directory
    self.data_format = data_format # can either be 'exp_map' or 'coord_3d'
    self.d_mode = d_mode  # train eval or test

    self.seed_length = seed_length # per task defined as 10
    self.prediction_lenght = prediction_length # also 10 (or 25)
    self.full_lenght = seed_length + prediction_length

    self.data_list = []
    # load the test train or validation set of the H3.6m dataset
    if d_mode == 'train':
      self.data_subset = ['S1', 'S6', 'S7', 'S8', 'S9']
    elif d_mode == 'test':
      self.data_subset = ['S11']
    else:
      self.data_subset = ['S5']

    """
    function to load one full instance from the dataset given the path to the .txt file
    """
    def load_full_instance(path_to_file):

      with open(path_to_file, "r") as f:
          lines = [line.rstrip() for line in f]

      lines = [line.split(',') for line in lines]
      for i in range(len(lines)):
        lines[i] = [float(line) for line in lines[i]]

      lines = np.asarray(lines)
      # downsample from 50 to 25Hz (according to slides)
      out = lines[::2]

      # to keep the distance moved of the root joint consistent the distance moved from two frames needs to be added
      if len(lines) % 2 == 1:
        out[1::,:3] = out[1::,:3] + lines[1::2,:3]
      else:
        out[1::,:3] = out[1::,:3] + lines[1:-1:2,:3]

      return out

    for direc in self.data_subset:
      individual_data_path = self.data_path + '/' + 'dataset' + '/' + direc
      scenarios = os.listdir(individual_data_path)

      for sc in scenarios:
        full_data_path = individual_data_path + '/' + sc
        instance = load_full_instance(full_data_path)
        instance = torch.tensor(instance)
        instance = keep_joints(instance)

        for k in range(int(instance.shape[0]/self.full_lenght)):
          inst = instance[k*self.full_lenght:(k+1)*self.full_lenght].clone()
          inst[0,:3] = torch.zeros(3)
          self.data_list.append(inst.float())

    if self.data_format == 'coord_3d':
      for i in range(len(self.data_list)):
        xyz = angles_to3dspace(self.data_list[i], 'cpu').view(self.data_list[i].shape[0], -1)
        self.data_list[i] = torch.tensor(xyz).float()

    return

  def __len__(self):
    return len(self.data_list)

  def __getitem__(self, idx):
    instance = self.data_list[idx]
    return instance[:self.seed_length].clone(), instance[self.seed_length:].clone()





import json

class AISData(Dataset):
  """
  Dataset class for the AIS data. Data is loaded in in coord_3d format, so where each joint is represented by a 3d point in space.
  """
  def __init__(self, data_path, seed_length = 10, prediction_length = 10):
    self.data_path = data_path
    self.seed_length = seed_length
    self.prediction_length = prediction_length
    self.full_length = prediction_length + seed_length

    self.data_list = []

    def load_full_instance(path_to_file):

      with open(path_to_file, "r") as f:
        pose_data = json.load(f)

      full_sequence = [pose['person']['keypoints'] for pose in pose_data]
      scaling_factor = 4.42894413e+02/0.41

      for j in range(len(full_sequence)):
        frame = full_sequence[j]
        full_frame = np.asarray([fr['pos'] for fr in frame if fr['score'] != 0])
        rearanged_list = []
        append_list_start = [8,12,13,14,9,10,11]
        append_list_middle = [1,0]
        append_list_end = [2,3,4,5,6,7]

        for i in append_list_start:
          rearanged_list.append(full_frame[i])

        middle_spine = (full_frame[1] + full_frame[8])/2
        rearanged_list.append(middle_spine)

        for i in append_list_middle:
          rearanged_list.append(full_frame[i])

        top_head = (full_frame[15] + full_frame[16])/2# + full_frame[17] + full_frame[18])/4
        rearanged_list.append(top_head)

        for i in append_list_end:
          rearanged_list.append(full_frame[i])

        frame = np.asarray(rearanged_list)*scaling_factor
        root = frame[0].reshape((1,3)).repeat(17, axis = 0)
        frame = frame - root
        full_sequence[j] = frame

      full_instance = np.asarray(full_sequence)
      return full_instance

    # list containing hardcoded file names for the dataset as there are also other files in the directory
    file_list = ['2021-08-04-singlePerson_000.json',
                  '2021-08-04-singlePerson_001.json',
                  '2021-08-04-singlePerson_002.json',
                  '2021-08-04-singlePerson_003.json',
                  #'2022-05-26_2persons_000.json',  # these three sequences have some problem with scores = 0  where they should not be
                  #'2022-05-26_2persons_001.json',
                  #'2022-05-26_2persons_002.json',
                  '2022-05-26_2persons_003.json']

    for file_name in file_list:
      full_data_path = self.data_path + '/' + file_name
      instance = load_full_instance(full_data_path)
      instance[:,:,1:] = instance[:,:,:0:-1]
      instance = torch.tensor(instance)

      for k in range(int(instance.shape[0]/self.full_length)):
        inst = instance[k*self.full_length:(k+1)*self.full_length].clone()
        self.data_list.append(inst.float())

    return

  def __len__(self):
    return len(self.data_list)

  def __getitem__(self, idx):
    instance = self.data_list[idx]
    return instance[:self.seed_length].clone().reshape(self.seed_length,-1), instance[self.seed_length:].clone().reshape(self.prediction_length,-1)