import torch, os
import numpy as np
import os.path as osp
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from utils import *
import frame_utils


class CustomImageDataset(Dataset):
  """
  Image under just one folder
  """
  def __init__(self, data_dir, img_size=256, transform=None, shuffle=True):
    self.transform = transform
    self.data_dir = data_dir
    self.img_size = img_size
    self.filelist = os.listdir(data_dir)
    self.filelist.sort()
    self.len = len(self.filelist)
    self.shuffle = shuffle
    self.idxs = np.array(range(self.len))
    if shuffle:
      self.rng = np.random.RandomState(1234)
      self.reset()

  def reset(self):
    self.rng.shuffle(self.idxs)

  def __getitem__(self, idx):
    fpath = os.path.join(self.data_dir, self.filelist[self.idxs[idx]])

    with open(fpath, "rb") as f:
      img = Image.open(f).convert("RGB")

    if self.transform is not None:
      img = self.transform(img)

    if self.img_size is not None and (img.shape[1] != self.img_size or img.shape[2] != self.img_size):
      print("!> Shape error: " + str(img.shape))

    # ignore label information
    return img, idx

  def __len__(self):
    return self.len


class FlowDataset(Dataset):
  def __init__(self, no_flow=False):
    self.augmentor = None
    self.no_flow = no_flow

    self.flow_list = []
    self.image_list = []
    self.mask_list = []
    self.extra_info = []

  def __getitem__(self, index):
    index = index % len(self.image_list)
    imgs = []
    for i in range(len(self.image_list[index])):
      img = frame_utils.read_gen(self.image_list[index][i])
      img = np.array(img).astype(np.uint8)[..., :3]
      img = torch.from_numpy(img).permute(2, 0, 1).float()
      imgs.append(img)
    imgs = torch.stack(imgs)

    flows = []
    masks = []
    if not self.no_flow:
      for i in range(len(self.flow_list[index])):
        flow = frame_utils.read_gen(self.flow_list[index][i])
        flow = np.array(flow).astype(np.float32)
        flow = torch.from_numpy(flow).permute(2, 0, 1).float()
        flows.append(flow)
        mask = frame_utils.read_gen(self.mask_list[index][i])
        mask = np.array(mask).astype(np.uint8)
        mask = torch.from_numpy(mask).float()
        masks.append(mask / mask.max())
      flows = torch.stack(flows)
      masks = torch.stack(masks)
    return imgs, flows, masks, self.extra_info[index]

  def __rmul__(self, v):
    self.flow_list = v * self.flow_list
    self.image_list = v * self.image_list
    return self
    
  def __len__(self):
    return len(self.image_list)
    

class DAVISDataset(FlowDataset):
  def __init__(self,
    data_dir="data/DAVIS", split="train", no_flow=True,
    seq_size=2, interval=1):
    """
    Args:
      split: train, val, test.
      seq_size: The sequence length of batch
      interval: The frame interval
    """
    super().__init__()
    self.data_dir = data_dir
    self.split = split
    self.is_test = no_flow
    vnames = open(osp.join(self.data_dir, "ImageSets", "2017",
      f"{split}.txt")).read().split("\n")[:-1]
    image_dir = osp.join(self.data_dir, "JPEGImages", "480p")
    for vname in vnames:
      images = os.listdir(osp.join(image_dir, vname))
      images.sort()
      fpaths = [osp.join(image_dir, vname, i) for i in images]
      for i in range(len(fpaths)):
        if i + (seq_size - 1) * interval >= len(fpaths):
          break # out of frame list
        frame_groups, flow_groups, mask_groups = [], [], []
        for j in range(seq_size):
          frame_groups.append(fpaths[i + j * interval])
          flow_groups.append(DAVISDataset.flow_path(frame_groups[-1]))
          mask_groups.append(DAVISDataset.conf_path(frame_groups[-1]))
        self.image_list += [frame_groups]
        self.flow_list += [flow_groups[:-1]]
        self.mask_list += [mask_groups[:-1]]
        self.extra_info += [frame_groups]

  @staticmethod
  def flow_path(fp):
    ind = fp.rfind("/") + 1
    fp = fp[:ind] + "forward_" + fp[ind:]
    fp = fp.replace("JPEGImages", "Flows")
    if ".png" in fp:
      return fp.replace(".png", ".flo")
    elif ".jpg" in fp:
      return fp.replace(".jpg", ".flo")

  @staticmethod
  def conf_path(fp):
    ind = fp.rfind("/") + 1
    fp = fp[:ind] + "reliable_" + fp[ind:]
    fp = fp.replace("JPEGImages", "Occlusions")
    if ".png" in fp:
      return fp.replace(".png", ".pgm")
    elif ".jpg" in fp:
      return fp.replace(".jpg", ".pgm")

  def __len__(self):
    """The number of frame parts"""
    return len(self.image_list)
