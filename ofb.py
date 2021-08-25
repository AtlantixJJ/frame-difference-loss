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

  if x.is_cuda:
    grid = grid.cuda()
  vgrid = grid + flo

  # scale grid to [-1,1] 
  vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:].clone() / max(W-1,1)-1.0
  vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:].clone() / max(H-1,1)-1.0

  vgrid = vgrid.permute(0,2,3,1)    
  output = F.grid_sample(x, vgrid)
  mask = F.grid_sample(torch.ones_like(x), vgrid)

  mask[mask < 0.9999] = 0
  mask[mask > 0] = 1

  return output, mask


def train(args):
  np.random.seed(args.seed)
  torch.manual_seed(args.seed)

  if args.cuda:
    torch.cuda.manual_seed(args.seed)
    kwargs = {'num_workers': 0, 'pin_memory': False}
  else:
    kwargs = {}

  if args.model_type == "rnn":
    transformer = transformer_net.TransformerRNN(args.pad_type)
    seq_size = 4
  else:
    transformer = transformer_net.TransformerNet(args.pad_type)
    seq_size = 2

  train_dataset = dataset.DAVISDataset(args.dataset, "train",
    seq_size=seq_size, no_flow=False)
  train_loader = DataLoader(train_dataset,
    shuffle=True, batch_size=1, **kwargs)

  if args.model_type == "rnn":
    transformer = transformer_net.TransformerRNN(args.pad_type)
  else:
    transformer = transformer_net.TransformerNet(args.pad_type)
  model_path = args.init_model
  print("=> Load from model file %s" % model_path)
  transformer.load_state_dict(torch.load(model_path))
  transformer.train()
  if args.model_type == "rnn":
    transformer.conv1 = transformer_net.ConvLayer(6, 32, kernel_size=9, stride=1, pad_type=args.pad_type)
  optimizer = torch.optim.Adam(transformer.parameters(), args.lr)
  mse_loss = torch.nn.MSELoss()

  vgg = Vgg16()
  vgg.load_state_dict(torch.load(os.path.join(
    args.vgg_model_dir, "vgg16.weight")))
  vgg.eval()

  if args.cuda:
    transformer.cuda()
    vgg.cuda()
    mse_loss.cuda()

  style = utils.tensor_load_resize(args.style_image, args.style_size)
  style = style.unsqueeze(0)
  print("=> Style image size: " + str(style.size()))
  print("=> Pixel OFB loss weight: %f" % args.time_strength)

  style = utils.preprocess_batch(style)
  if args.cuda: style = style.cuda()
  utils.tensor_save_bgrimage(style[0].detach(), os.path.join(args.save_model_dir, 'train_style.jpg'), args.cuda)
  style = utils.subtract_imagenet_mean_batch(style)
  features_style = vgg(style)
  gram_style = [utils.gram_matrix(y).detach()
    for y in features_style]
  iters = 0
  for e in range(args.epochs):
    transformer.train()
    transformer.cuda()
    for batch_id, (x, flow, occ, _) in enumerate(train_loader):
      x, flow, occ = x[0], flow[0], occ[0]
      iters += 1

      x = utils.preprocess_batch(x) # (N, 3, 256, 256)
      flow = utils.factor4_crop(flow)
      occ = utils.factor4_crop(occ)
      if args.cuda:
        x, flow, occ = x.cuda(), flow.cuda(), occ.cuda()
      y = transformer(x) # (N, 3, 256, 256)

      vgg_y = utils.subtract_imagenet_mean_batch(y)
      vgg_x = utils.subtract_imagenet_mean_batch(x)
      
      f_y = vgg(vgg_y)
      f_c = f_y[2]
      with torch.no_grad():
        f_x = vgg(vgg_x)
      f_xc_c = f_x[2]

      content_loss = args.content_weight * mse_loss(f_c, f_xc_c)

      style_loss = 0.
      for m in range(len(f_y)):
        gram_s = gram_style[m]
        gram_y = utils.gram_matrix(f_y[m])
        batch_style_loss = 0
        for n in range(gram_y.shape[0]):
          batch_style_loss += args.style_weight * mse_loss(
            gram_y[n], gram_s[0])
        style_loss += batch_style_loss / gram_y.shape[0]

      warped_y, mask = warp(y[1:], flow)
      warped_y = warped_y.detach()
      mask *= occ
      pixel_ofb_loss = args.time_strength * weighted_mse(
        y[:-1], warped_y, mask)

      total_loss = content_loss + style_loss + pixel_ofb_loss

      total_loss.backward()
      optimizer.step()
      optimizer.zero_grad()

      if (iters + 1) % 100 == 0:
        prefix = args.save_model_dir
        idx = (iters + 1) // 100
        for i in range(x.shape[0]):
          utils.tensor_save_bgrimage(
            y.data[i], f"{prefix}/out_{idx}-{i}.png", args.cuda)
          utils.tensor_save_bgrimage(
            x.data[i], f"{prefix}/in_{idx}-{i}.png", args.cuda)
          if i < flow.shape[0]:
            flow_image = flow_to_color(
              flow.data[i].cpu().numpy().transpose(1,2,0))
            utils.save_image(f"{prefix}/forward_flow_{idx}-{i}.png", flow_image)
            warped_x, _ = warp(x[i+1:i+2], flow[i:i+1])
            utils.tensor_save_bgrimage(
              warped_y.data[0], f"{prefix}/wout_{idx}-{i}.png", args.cuda)
            utils.tensor_save_bgrimage(
              warped_x.data[0], f"{prefix}/win_{idx}-{i}.png", args.cuda)
            utils.tensor_save_image(f"{prefix}/conf_{idx}-{i}.png", mask.data[0])

      mesg = "{}\tEpoch {}:\t[{}/{}]\tcontent: {:.6f}\tstyle: {:.6f}\tpixel ofb: {:.6f}\ttotal: {:.6f}".format(
        time.ctime(), e + 1, batch_id + 1, len(train_loader),
        content_loss, style_loss, pixel_ofb_loss, total_loss)
      print(mesg)

    # save model
    transformer.eval()
    transformer.cpu()
    save_model_filename = "epoch_" + str(e) + "_" + str(args.content_weight) + "_" + str(args.style_weight) + ".model"
    save_model_path = os.path.join(args.save_model_dir, save_model_filename)
    torch.save(transformer.state_dict(), save_model_path)

  print("\nDone, trained model saved at", save_model_path)


def check_paths(args):
    try:
        if not os.path.exists(args.save_model_dir):
            os.makedirs(args.save_model_dir)
    except OSError as e:
        print(e)
        sys.exit(1)
        

def main():
    main_arg_parser = argparse.ArgumentParser(description="parser for fast-neural-style (FDB)")
    subparsers = main_arg_parser.add_subparsers(title="subcommands", dest="subcommand")

    train_arg_parser = subparsers.add_parser("train",
                                             help="parser for training arguments")
    train_arg_parser.add_argument("--time-strength", type=float, default=400.0,
                                  help="OFB loss weight.")
    train_arg_parser.add_argument("--init-model", type=str, default="",
                                  help="model dir")
    train_arg_parser.add_argument("--model-type", type=str, default="rnn", # rnn | sfn
                                  help="model dir")
    train_arg_parser.add_argument("--pad-type", type=str, default="reflect-start", # rnn | sfn
                                  help="reflect-start|none")
    train_arg_parser.add_argument("--epochs", type=int, default=1,
                                  help="number of training epochs, default is 1")
    train_arg_parser.add_argument("--dataset", type=str,
                                  default="data/DAVIS",
                                  help="path to training dataset, the path should point to a folder "
                                       "containing another folder with all the training images")
    train_arg_parser.add_argument("--style-image", type=str, default="data/styles/starry_night.jpg",
                                  help="path to style-image")
    train_arg_parser.add_argument("--vgg-model-dir", type=str, default="pretrained",
                                  help="directory for vgg, if model is not present in the directory it is downloaded")
    train_arg_parser.add_argument("--save-model-dir", type=str, default="exprs",
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

    if args.subcommand is None:
        print("ERROR: specify either train or eval")
        sys.exit(1)

    if args.subcommand == "train":
        check_paths(args)
        train(args)


if __name__ == "__main__":
    main()
