"""
Train the SFN baseline model (without temporal loss)
"""
import numpy as np
import argparse, sys, os, time, glob
import torch
from torch.functional import F
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import dataset
from torchvision import transforms
from torch.functional import F
import utils
import transformer_net
from vgg16 import Vgg16
from flow_vis import flow_to_color
import time
from tqdm import tqdm


def center_crop(x, h, w):
  # Assume x is (N, C, H, W)
  H, W = x.size()[2:]
  if h == H and w == W: return x
  assert(h <= H and w <= W)
  dh, dw = H - h, W - w
  ddh, ddw = dh // 2, dw // 2
  return x[:, :, ddh:-(dh-ddh), ddw:-(dw-ddw)]

def weighted_mse(x, y, conf):
  diff = (x - y) * conf
  diff = diff * diff
  return diff.sum() / conf.sum()

def warp(x, flo):
  """
  warp an image/tensor (im2) back to im1, according to the optical flow
  x: [B, C, H, W] (im2)
  flo: [B, 2, H, W] flow
  """
  B, C, H, W = x.size()
  # mesh grid 
  xx = torch.arange(0, W).view(1,-1).repeat(H,1)
  yy = torch.arange(0, H).view(-1,1).repeat(1,W)
  xx = xx.view(1,1,H,W).repeat(B,1,1,1)
  yy = yy.view(1,1,H,W).repeat(B,1,1,1)
  grid = torch.cat((xx,yy),1).float()

  if x.is_cuda: grid = grid.cuda()
  vgrid = grid + flo

  # scale grid to [-1,1] 
  vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:].clone() / max(W-1,1)-1.0
  vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:].clone() / max(H-1,1)-1.0

  vgrid = vgrid.permute(0,2,3,1)    
  output = F.grid_sample(x, vgrid)
  mask = torch.ones(x.size()).cuda()
  mask = F.grid_sample(mask, vgrid)

  # if W==128:
    # np.save('mask.npy', mask.cpu().data.numpy())
    # np.save('warp.npy', output.cpu().data.numpy())
  
  mask[mask<0.9999] = 0
  mask[mask>0] = 1
  
  return output, mask


def train_fdb(args):
  transformer = transformer_net.TransformerNet(args.pad_type)
  train_dataset = dataset.DAVISDataset(args.dataset,
    seq_size=2, use_flow=args.flow)
  train_loader = DataLoader(train_dataset, batch_size=1)

  transformer.train()
  optimizer = torch.optim.Adam(transformer.parameters(), args.lr)
  mse_loss = torch.nn.MSELoss()
  l1_loss = torch.nn.SmoothL1Loss()

  vgg = Vgg16()
  utils.init_vgg16(args.vgg_model_dir)
  vgg.load_state_dict(torch.load(os.path.join(args.vgg_model_dir, "vgg16.weight")))
  vgg.eval()

  if args.cuda:
    transformer.cuda()
    vgg.cuda()
    mse_loss.cuda()
    l1_loss.cuda()

  style = utils.tensor_load_resize(args.style_image, args.style_size)
  style = style.unsqueeze(0)
  print("=> Style image size: " + str(style.size()))

  style = utils.preprocess_batch(style)
  if args.cuda: style = style.cuda()
  style = utils.subtract_imagenet_mean_batch(style)
  features_style = vgg(style)
  gram_style = [utils.gram_matrix(y).detach() for y in features_style]

  train_loader.dataset.reset()
  agg_content_loss = agg_style_loss = agg_pixelfdb_loss = agg_featurefdb_loss = 0.
  iters = 0
  elapsed_time = 0
  for batch_id, (x, flow, conf) in enumerate(tqdm(train_loader)):
    x=x[0]
    iters += 1

    optimizer.zero_grad()
    x = utils.preprocess_batch(x) # (N, 3, 256, 256)
    if args.cuda: x = x.cuda()
    y = transformer(x) # (N, 3, 256, 256)

    xc = center_crop(x.detach(), y.shape[2], y.shape[3])

    y = utils.subtract_imagenet_mean_batch(y)
    xc = utils.subtract_imagenet_mean_batch(xc)

    features_y = vgg(y)
    features_xc = vgg(xc)

    # FDB
    begin_time = time.time()
    pixel_fdb_loss = mse_loss(y[1:] - y[:-1], xc[1:] - xc[:-1])
    # temporal content: 16th
    feature_fdb_loss = l1_loss(
      features_y[2][1:] - features_y[2][:-1],
      features_xc[2][1:] - features_xc[2][:-1])
    pixel_fdb_loss.backward()
    elapsed_time += time.time() - begin_time

    if batch_id > 1000: break
  print(elapsed_time / float(batch_id + 1))

def train_ofb(args):
  train_dataset = dataset.DAVISDataset(args.dataset, use_flow=True)
  train_loader = DataLoader(train_dataset, batch_size=1)

  transformer = transformer_net.TransformerNet(args.pad_type)
  transformer.train()
  optimizer = torch.optim.Adam(transformer.parameters(), args.lr)
  mse_loss = torch.nn.MSELoss()
  l1_loss = torch.nn.SmoothL1Loss()

  vgg = Vgg16()
  utils.init_vgg16(args.vgg_model_dir)
  vgg.load_state_dict(torch.load(os.path.join(args.vgg_model_dir, "vgg16.weight")))
  vgg.eval()

  if args.cuda:
    transformer.cuda()
  vgg.cuda()
  mse_loss.cuda()
  l1_loss.cuda()

  style = utils.tensor_load_resize(args.style_image, args.style_size)
  style = style.unsqueeze(0)
  print("=> Style image size: " + str(style.size()))
  print("=> Pixel OFB loss weight: %f" % args.time_strength)

  style = utils.preprocess_batch(style)
  if args.cuda: style = style.cuda()
  style = utils.subtract_imagenet_mean_batch(style)
  features_style = vgg(style)
  gram_style = [utils.gram_matrix(y).detach() for y in features_style]

  train_loader.dataset.reset()
  transformer.train()
  transformer.cuda()
  agg_content_loss = agg_style_loss = agg_pixelofb_loss = 0.
  iters = 0
  anormaly = False
  elapsed_time = 0
  for batch_id, (x, flow, conf) in enumerate(tqdm(train_loader)):
    x,flow,conf=x[0],flow[0],conf[0]
    iters += 1

    optimizer.zero_grad()
    x = utils.preprocess_batch(x) # (N, 3, 256, 256)
    if args.cuda:
      x = x.cuda()
      flow = flow.cuda()
      conf = conf.cuda()
    y = transformer(x) # (N, 3, 256, 256)

    begin_time = time.time()
    warped_y, warped_y_mask = warp(y[1:], flow)
    warped_y = warped_y.detach()
    warped_y_mask *= conf
    pixel_ofb_loss = args.time_strength * weighted_mse(
    y[:-1], warped_y, warped_y_mask)
    pixel_ofb_loss.backward()
    elapsed_time += time.time() - begin_time
    if batch_id > 1000: break
  print(elapsed_time / float(batch_id + 1))


def main():
  main_arg_parser = argparse.ArgumentParser(description="parser for fast-neural-style (FDB)")
  subparsers = main_arg_parser.add_subparsers(title="subcommands", dest="subcommand")

  train_arg_parser = subparsers.add_parser("train",
                       help="parser for training arguments")

  train_arg_parser.add_argument("--flow", type=bool, default=True,
                  help="If to train use OFB loss")
  train_arg_parser.add_argument("--time-strength", type=float, default=1000.0,
                  help="pixel OFB weight")
  train_arg_parser.add_argument("--init-model-dir", type=str, default="../exprs/rnn/NetStyle1/",
                  help="model dir")
  train_arg_parser.add_argument("--model-type", type=str, default="rnn", # rnn | sfn
                  help="model dir")
  train_arg_parser.add_argument("--pad-type", type=str, default="reflect-start", # rnn | sfn
                  help="reflect-start|none")
  train_arg_parser.add_argument("--epochs", type=int, default=1,
                  help="number of training epochs, default is 1")
  train_arg_parser.add_argument("--batch-size", type=int, default=2,
                  help="batch size for training, default is 4: BPTT for 4 steps")
  train_arg_parser.add_argument("--dataset", type=str,
                  default="../data/DAVIS_VIDEO/train/JPEGImages/480p/",
                  help="path to training dataset, the path should point to a folder "
                     "containing another folder with all the training images")
  train_arg_parser.add_argument("--style-image", type=str, default="../data/styles/starry_night.jpg",
                  help="path to style-image")
  train_arg_parser.add_argument("--vgg-model-dir", type=str, default="../pretrained/",
                  help="directory for vgg, if model is not present in the directory it is downloaded")
  train_arg_parser.add_argument("--save-model-dir", type=str, default="../exprs/rnn/",
                  help="path to folder where trained model will be saved.")
  train_arg_parser.add_argument("--image-size", type=int, default=400,
                  help="size of training images, default is 256 X 256")
  train_arg_parser.add_argument("--style-size", type=int, default=400,
                  help="size of style-image, default is the original size of style image")
  train_arg_parser.add_argument("--cuda", type=int, default=1, help="set it to 1 for running on GPU, 0 for CPU")
  train_arg_parser.add_argument("--seed", type=int, default=1234, help="random seed for training")
  train_arg_parser.add_argument("--content-weight", type=float, default=1.0,
                  help="weight for content-loss, default is 1.0")
  train_arg_parser.add_argument("--style-weight", type=float, default=10.0,
                  help="weight for style-loss, default is 10.0")
  train_arg_parser.add_argument("--lr", type=float, default=1e-4,
                  help="learning rate, default is 0.001")
  train_arg_parser.add_argument("--log-interval", type=int, default=100,
                  help="number of images after which the training loss is logged, default is 100")

  args = main_arg_parser.parse_args()

  train_ofb(args)
  train_fdb(args)

if __name__ == "__main__":
  main()
