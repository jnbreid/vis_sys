import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import cv2
import numpy as np
from PIL import Image
from google.colab.patches import cv2_imshow

"""
parts of the code for Market1501 are used from
https://github.com/CoinCheung/triplet-reid-pytorch/blob/master/datasets/Market1501.py
and changed to fit our needs
"""
# dataset class for market1501 dataset
class Market1501(Dataset):
    def __init__(self, data_path, transforms, *args, **kwargs):
        super(Market1501, self).__init__(*args, **kwargs)
        self.data_path = data_path
        self.imgs = os.listdir(data_path)
        self.imgs = [el for el in self.imgs if os.path.splitext(el)[1] == '.jpg']
        self.lb_ids = [int(el.split('_')[0]) for el in self.imgs]
        self.lb_cams = [int(el.split('_')[1][1]) for el in self.imgs]
        self.imgs = [os.path.join(data_path, el) for el in self.imgs]
        self.transforms = transforms

        # useful for sampler
        self.lb_img_dict = dict()
        self.lb_ids_uniq = set(self.lb_ids)
        lb_array = np.array(self.lb_ids)
        for lb in self.lb_ids_uniq:
            idx = np.where(lb_array == lb)[0]
            self.lb_img_dict.update({lb: idx})

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = Image.open(self.imgs[idx])
        img = self.transforms(img)
        return img, torch.tensor(self.lb_ids[idx])

# dataset wrapper for market 1501 dataset to always get triplet (anchor, positive anchor, negative anchor)
class TripletDataset:
    """
    Dataset class from which we sample random triplets
    """
    def __init__(self, dataset):
        """ Dataset initializer"""
        self.dataset = dataset
        self.labels = np.array(([l for _,l in dataset]))
        self.unique_labels = np.unique(np.asarray(self.labels))
        self.label_dict = {}    # dict to save indices of instances for each class

        for i in range(self.unique_labels.shape[0]):
          w_ids = np.where(self.labels == self.unique_labels[i])[0]
          self.label_dict[self.unique_labels[i]] = w_ids

        return

    def __len__(self):
        """ Returning number of anchors """
        return len(self.dataset)

    def __getitem__(self, i):
        """
        Sampling a triplet for the dataset. Index i corresponds to anchor
        """
        # sampling anchor
        anchor_img, anchor_lbl = self.dataset[i]

        # lists for positives and negatives
        # get another positive instance
        pos_id = np.random.choice(self.label_dict[anchor_lbl.item()])
        while pos_id == i and len(self.label_dict[anchor_lbl.item()]) > 1:
          pos_id = np.random.choice(self.label_dict[anchor_lbl.item()])
        # get any other negarive class (other class)
        neg_class = np.random.choice(self.unique_labels)
        while neg_class == anchor_lbl.item() and len(self.unique_labels) > 1:
          neg_class = np.random.choice(self.unique_labels)
        # and a random instance from this class
        neg_id = np.random.choice(self.label_dict[neg_class])

        pos_img, pos_lbl = self.dataset.__getitem__(pos_id)
        neg_img, neg_lbl = self.dataset.__getitem__(neg_id)

        return (anchor_img, pos_img, neg_img), (anchor_lbl, pos_lbl, neg_lbl)

# class to implement the online mining strategie
# in the paper they use batches directly, here they are used indirectly in the dataset class
# this is quite inefficient, but i dont know how to get batches with the same individual as anchor to compare min and max
# for each instance a certain number of positive and negative instances are sampled and the max/min are retuned as sample
class TripletDatasetWrapper:

    def __init__(self, dataset, model, device, mini_batch_size = 32):
        """ Dataset initializer"""
        self.dataset = dataset
        self.model = model
        self.mini_batch_size = mini_batch_size
        
        return

    def __len__(self):
        """ Returning number of anchors """
        return len(self.dataset)

    def __getitem__(self, i):
        #(anchor_img, pos_img, neg_img), (anchor_lbl, pos_lbl, neg_lbl) = self.dataset[i]

        anch_imgs = []
        anch_lb = []
        pos_imgs = []
        pos_lb = []
        neg_imgs = []
        neg_lb = []

        # get random instances
        for k in range(self.mini_batch_size):
          (anchor_img, pos_img, neg_img), (anchor_lbl, pos_lbl, neg_lbl) = self.dataset[i]
          anch_imgs.append(anchor_img)
          anch_lb.append(anchor_lbl)
          pos_imgs.append(pos_img)
          pos_lb.append(pos_lbl)
          neg_imgs.append(neg_img)
          neg_lb.append(neg_lbl)

        anchor_imgs_t = torch.stack(anch_imgs, dim = 0)
        pos_imgs_t = torch.stack(pos_imgs, dim = 0)
        neg_imgs_t = torch.stack(neg_imgs, dim = 0)
        # pass them through the network
        embs = model(anchor_imgs_t.to(device), pos_imgs_t.to(device), neg_imgs_t.to(device))
        # calculate distances
        d_p = torch.sqrt((embs[0] - embs[1]).pow(2).sum(dim=-1)).detach()
        d_n = torch.sqrt((embs[0] - embs[2]).pow(2).sum(dim=-1)).detach()
        # and find the min negative and max positive instance
        max_ind = torch.argmax(d_p)
        min_ind = torch.argmin(d_n)

        max_pos = pos_imgs[max_ind.item()]
        max_label = pos_lb[max_ind.item()]
        min_neg = neg_imgs[min_ind.item()]
        min_label = neg_lb[min_ind.item()]
        # and return this instance 
        return (anch_imgs[0], max_pos, min_neg), (anch_lb[0], max_label, min_label)