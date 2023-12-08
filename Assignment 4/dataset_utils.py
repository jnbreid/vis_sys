import os
import sys
import subprocess
import shutil


def extract_frames(data_path = '/home/profloo/Documents/KTH'):

    # sequences_list = ''
    video_files=os.listdir(data_path + '/data/videos/')
    video_files.sort()

    # extract frames from video clips
    args=['ffmpeg', '-i']
    for video in video_files:
        video_name = video[:-11]	# remove '_uncomp.avi' from name
        # print 'video name is: ', video_name
        frame_name = 'frame_%d.jpg'	# count starts from 1 by default
        #print(data_path + '/data/frames/'+video_name)
        os.makedirs(data_path + '/data/frames/'+video_name)
        args.append(data_path + '/data/videos/'+video)
        args.append(data_path + '/data/frames/'+video_name+'/'+frame_name)
        ffmpeg_call = ' '.join(args)
        # print ffmpeg_call
        # print args
        subprocess.call(ffmpeg_call, shell=True)		# execute the system call
        args=['ffmpeg', '-i']
        if (video_files.index(video) + 1) % 50 == 0:
            print('Completed till video : ', (video_files.index(video) + 1))
                

    print('[MESSAGE]	Frames extracted from all videos')

    os.makedirs(data_path + '/data/' + 'TRAIN')
    os.makedirs(data_path + '/data/' + 'VALIDATION')
    os.makedirs(data_path + '/data/' + 'TEST')

    train = [11, 12, 13, 14, 15, 16, 17, 18]
    validation =[19, 20, 21, 23, 24, 25, 1, 4]
    test = [22, 2, 3, 5, 6, 7, 8, 9, 10]

    lines = [line.rstrip('\n').rstrip('\r') for line in open('sequences_list.txt')]
    # remove blank entries i.e. empty lines

    lines = [item for item in lines if len(item) > 6] # at least 'person' is written
    lines = [item for item in lines if item[0:6] == 'person']
    lines = [line.split('\t') for line in lines]

    lines.sort()

    success_count=0
    error_count=0
    for line in lines:
        vid = line[0].strip(' ')
        subsequences = line[-1].split(',')
        person = int(vid[6:8])
        if person in train:
            move_to = 'TRAIN'
        elif person in validation:
            move_to = 'VALIDATION'
        else:
            move_to = 'TEST'
        for seq in subsequences:
            try:
                limits=seq.strip(' ').split('-')
                seq_path=data_path + '/data/' + move_to + '/' + vid + '_frame_' + limits[0] + '_' + limits[1]
                os.makedirs(seq_path)
            except:
                print('-----------------------------------------------------------')
                print('[ERROR MESSAGE]: ')
                print('limits : ', limits)
                print('seq_path : ', seq_path)
                print('-----------------------------------------------------------')
                continue
            error_flag=False
            for i in range(int(limits[0]), int(limits[1])+1):
                src = data_path + '/data' + '/frames/' + vid + '/frame_' + str(i) + '.jpg'
                # print i, src, limits
                dst = seq_path
                try:
                    shutil.copy(src, dst)
                except:
                    error_flag = True
            if error_flag:
                print("[ERROR]: ", seq_path)
                error_count+=1

        if (lines.index(line) + 1) % 50 == 0:
            print('Completed till video : ', (lines.index(line) + 1))
        success_count+=1

    print('[ALERT]		Total error count is : ', error_count)
    print('[MESSAGE]	Data split into train, validation and test')


# dataset class for KTH dataset

import os
import random
from PIL import Image
import torch
import torchvision.transforms as transforms

class KTH_dataset(torch.utils.data.Dataset):

  def __init__(self, DATA_PATH,  transforms = None, mode = 'TRAIN', instance_len = 20):
    self.instances = []
    self.inst_class = []
    self.path = DATA_PATH
    self.mode = mode
    self.transforms = transforms
    self.instance_len = instance_len

    self.mapping = {'boxing':0, 'handclapping':1, 'handwaving':2, 'jogging':3, 'running':4, 'walking':5}

    all_vids = os.listdir(self.path + '/' + self.mode)


    for vid in all_vids:
      frames = os.listdir(self.path + '/' + self.mode + '/' + vid)
      frames.sort()
      current_instance = []
 
      for i in range(int(len(frames)/self.instance_len)):
        current_instance = frames[i*self.instance_len: i*self.instance_len + self.instance_len]

        current_instance = [(self.path + '/' + self.mode + '/' + vid + '/' + item) for item in current_instance]

        self.instances.append(current_instance)

        self.inst_class.append(int(self.mapping[vid.split('_')[1]]))


  def __getitem__(self, x):
    instance_img = self.instances[x]
    label = torch.tensor(self.inst_class[x])

    tens_list = []

    for inst in instance_img:
      image = Image.open(inst).convert("RGB")
      inst_tensor = transforms.ToTensor()(image)
      tens_list.append(inst_tensor)
      
    inst_tensor = torch.stack(tens_list, dim=0)

    if self.transforms != None:
      inst_tensor = self.transforms(inst_tensor)

    return inst_tensor, label

  def __len__(self):
    return len(self.instances)
