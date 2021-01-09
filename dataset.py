import torch, os
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from utils import *

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


class DAVISDataset(Dataset):
    """
    data_dir: dir/subitem/*.jpg, dir/subitem/flow_/forward_%d_%d.flo
    batch size must be 1
    """

    def __init__(self, data_dir,
        seq_size=4, interval=1, img_size=(400, 400), use_flow=False):
        """
        Args:
            seq_size: The sequence length of batch
            interval: The frame interval
        """
        self.builtin_transform = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Lambda(lambda x: x.mul(255))])
        self.use_flow = use_flow
        self.data_dir = data_dir
        self.img_size = img_size
        self.seq_size = seq_size
        self.interval = interval
        self.seq_length = 1 + interval * (seq_size - 1)

        # video folder: image list
        self.forward_flow_list = {}
        self.backward_flow_list = {}
        self.forward_conf_list = {}
        self.backward_conf_list = {}
        self.image_list = {}

        for basepath, subdir, files in os.walk(data_dir):
            if "flow_" in basepath and self.use_flow: # entered flow dir
                video_name = basepath.split("/")[-2]
                forward_flow_files = [f for f in files if "forward" in f]
                backward_flow_files = [f for f in files if "backward" in f]
                forward_flow_files = ["forward_%d_%d.flo" % (i, i+1) for i in range(1, len(forward_flow_files)+1)]
                backward_flow_files = ["backward_%d_%d.flo" % (i+1, i) for i in range(1, len(forward_flow_files)+1)]
                forward_conf_files = ["reliable_%d_%d.pgm" % (i, i+1) for i in range(1, len(forward_flow_files)+1)]
                backward_conf_files = ["reliable_%d_%d.pgm" % (i+1, i) for i in range(1, len(forward_flow_files)+1)]
                self.forward_flow_list[video_name] = forward_flow_files
                self.backward_flow_list[video_name] = backward_flow_files
                self.forward_conf_list[video_name] = forward_conf_files
                self.backward_conf_list[video_name] = backward_conf_files
            else:
                imagefiles = [f for f in files if ".jpg" in f or ".png" in f]
                if len(imagefiles) < 2: continue
                video_name = basepath.split("/")[-1]
                imagefiles.sort()
                self.image_list[video_name] = imagefiles
        
        self.vlen = len(self.image_list.keys()) # video number
        self.len = 0 # image number
        for k, v in self.image_list.items():
            self.len += len(v)
        
        self.rng = np.random.RandomState(1234)
        self.idxs = np.array(range(self.len))

    def reset(self): #shuffle
        self.rng.shuffle(self.idxs)
    
    def transform(self, imgs, flow, conf):
        N, H, W, C = imgs.shape
        H_, W_ = H - self.img_size[0], W - self.img_size[1]
        H_rand, W_rand = self.rng.randint(H_), self.rng.randint(W_)
        imgs = imgs[:, H_rand:H_rand+self.img_size[0], W_rand:W_rand+self.img_size[1], :]

        if self.use_flow:
            flow = flow[:, H_rand:H_rand+self.img_size[0], W_rand:W_rand+self.img_size[1], :]
            conf = conf[:, H_rand:H_rand+self.img_size[0], W_rand:W_rand+self.img_size[1]]
            flow = torch.from_numpy(flow.transpose(0, 3, 1, 2)).float()
            conf = torch.from_numpy(conf).float() / 255. # confidence map to 0/1
        
        imgs = torch.from_numpy(imgs.transpose(0, 3, 1, 2)).float()

        # imgs_t is in (0, 255) scale
        return imgs, flow, conf

    def __getitem__(self, idx):
        vid_idx = self.rng.randint(self.vlen)
        vid_name = list(self.image_list.keys())[vid_idx]
        frame_number = len(self.image_list[vid_name])

        frame_idx = self.rng.randint(frame_number - self.seq_length)
        workdir = os.path.join(self.data_dir, vid_name)

        imgs = []
        #print("b")
        for i in range(0, self.seq_length, self.interval):
            fname = self.image_list[vid_name][frame_idx + i]
            fpath = os.path.join(workdir, fname)
            imgs.append(read_image_file(fpath))
            #print(fpath)

        flow = []
        conf = []
        if self.use_flow:
            for i in range(self.seq_size - 1):
                fpath = os.path.join(workdir, "flow_", self.forward_flow_list[vid_name][frame_idx])
                flow.append(read_flow_file(fpath))
                fpath = os.path.join(workdir, "flow_", self.forward_conf_list[vid_name][frame_idx])
                conf.append(read_image_file(fpath))
            return self.transform(np.array(imgs), np.stack(flow, 0), np.stack(conf, 0))
        
        return self.transform(np.array(imgs), 0, 0)

    def __len__(self):
        return self.len
