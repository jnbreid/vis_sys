import cmath
import math
import numpy as np
import torch
from matplotlib import pyplot as plt

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib.animation as animation

import matplotlib.pyplot as plt
import numpy as np
import h5py
import os


from utils.h3m_utils import angles_to3dspace


class HumanPoseViz():
  """
  Class acting as container for the creation of a fuction animation 
  """
  def __init__(self, ax, seed, gt, pred = None):

    self.ax = ax

 
    r = 750;
    xroot, yroot, zroot = 0,0,0
    self.ax.set_xlim3d([-r+xroot, r+xroot])
    self.ax.set_zlim3d([-r+zroot, r+zroot])
    self.ax.set_ylim3d([-r+yroot, r+yroot])

    self.ax.set_aspect('equal')


    self.seed = seed
    self.gt = gt
    
    self.pred = pred

    self.seed_len = seed.shape[0]
    self.pred_len = gt.shape[0]
    
    self.I = [0,1,2,0,4,5,0,7,8, 9, 8,11,12, 8,14,15]
    self.J = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]


    self.LR =  np.array([1,1,1,0,0,0,0, 0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=bool)

    self.i = 0

    pose = np.zeros((17,3))

    self.gt_plots = []
    
    for i in np.arange( len(self.I) ):
        x = np.array( [pose[self.I[i], 0], pose[self.J[i], 0]] )
        y = np.array( [pose[self.I[i], 1], pose[self.J[i], 1]] )
        z = np.array( [pose[self.I[i], 2], pose[self.J[i], 2]] )

        self.gt_plots.append(ax.plot(x, y, z, lw=2, c='b' if self.LR[i] else "r", alpha=0.7))

    self.pred_plots = []
    pred_pose = np.zeros((17,3))

    for i in np.arange( len(self.I) ):
        x = np.array( [pred_pose[self.I[i], 0], pred_pose[self.J[i], 0]] )
        y = np.array( [pred_pose[self.I[i], 1], pred_pose[self.J[i], 1]] )
        z = np.array( [pred_pose[self.I[i], 2], pred_pose[self.J[i], 2]] )

        self.pred_plots.append(ax.plot(x, y, z, lw=2, c='c' if self.LR[i] else "m", alpha=0.7))



  def update(self,j):
    if j >= self.seed_len + self.pred_len:
      return self.ax
  
    if j < self.seed_len:
      gt_pose = self.seed[j,:,:]
      

    else:
      gt_pose = self.gt[j-self.seed_len,:,:]
    
    for i in np.arange( len(self.I) ):
      x = np.array( [gt_pose[self.I[i], 0], gt_pose[self.J[i], 0]] )
      y = np.array( [gt_pose[self.I[i], 1], gt_pose[self.J[i], 1]] )
      z = np.array( [gt_pose[self.I[i], 2], gt_pose[self.J[i], 2]] )

      self.gt_plots[i][0].set_xdata(x)
      self.gt_plots[i][0].set_ydata(y)
      self.gt_plots[i][0].set_3d_properties(z)

    if self.pred is not None:
      if j >= self.seed_len:
        pred_pose = self.pred[j-self.seed_len,:,:]
        for i in np.arange(len(self.I)):
          x = np.array( [pred_pose[self.I[i], 0], pred_pose[self.J[i], 0]] )
          y = np.array( [pred_pose[self.I[i], 1], pred_pose[self.J[i], 1]] )
          z = np.array( [pred_pose[self.I[i], 2], pred_pose[self.J[i], 2]] )
          self.pred_plots[i][0].set_xdata(x)
          self.pred_plots[i][0].set_ydata(y)
          self.pred_plots[i][0].set_3d_properties(z)

    return self.ax
    


def viz_figure(seed_frames, gt_frames, pred_frames = None, save_path = None, angles = True):
  """
  Function to create a mp4 video of a single sequence of poses consisting of seed frames, and gt frames.
  Optionally predicted frames can be added and visualized.
  The mp4 file is saved in the save path
  The animation object is returned.
  """
  if angles is True:
    seed_frames = angles_to3dspace(seed_frames.cpu(), 'cpu')
    gt_frames = angles_to3dspace(gt_frames.cpu(), 'cpu')
    
    if pred_frames is not None:
      pred_frames = angles_to3dspace(pred_frames.cpu(), 'cpu')


  seed_frames = seed_frames.reshape(seed_frames.shape[0],-1,3).cpu().numpy()
  seed_frames[:,:,1:] = seed_frames[:,:,:0:-1]

  gt_frames = gt_frames.reshape(gt_frames.shape[0],-1,3).cpu().numpy()
  gt_frames[:,:,1:] = gt_frames[:,:,:0:-1]
  if pred_frames is not None:
    pred_frames = pred_frames.reshape(pred_frames.shape[0],-1,3).cpu().numpy()
    pred_frames[:,:,1:] = pred_frames[:,:,:0:-1]


  fig = plt.figure(figsize=(7,7))
  ax = fig.add_subplot(projection='3d')
  ax.set_xlabel("x")
  ax.set_ylabel("y")
  ax.set_zlabel("z")


  vis_wizzard = HumanPoseViz(ax, seed_frames, gt_frames, pred = pred_frames)

  def animate(i):
    ret_ax = vis_wizzard.update(i)
    return ret_ax

  anim = animation.FuncAnimation(fig, animate, frames = seed_frames.shape[0]+gt_frames.shape[0])

  if save_path != None:
    fps = 15
    anim.save(save_path, fps=fps, extra_args=['-vcodec', 'libx264'])

  return anim


def vis_one_prediction(model, dataset, output_len, index = 250, save_path = 'viz.mp4', angles = True):
  """
  function to visualize predictions given a specific dataset. The output lenght of the dataset
  needs to be the same as 'pred_len'
  """
  model.cpu().eval()

  test_item = dataset.__getitem__(index)


  if output_len == 10:
    output = model(test_item[0].cpu().unsqueeze(0).repeat(2,1,1))
    output = output.detach()[0,:,:]
  else:
    rep = int(np.ceil(output_len/10))
    outputs = test_item[0].cpu().unsqueeze(0).repeat(2,1,1)
    out_list = []
    for i in range(rep):
        outputs = model(outputs)
        out_list.append(outputs)

    output = torch.cat(out_list, dim = 1)[:,:output_len,:].detach()[0,:,:]

  anim = viz_figure(test_item[0], test_item[1], pred_frames = output, save_path= save_path, angles = angles)

  return anim