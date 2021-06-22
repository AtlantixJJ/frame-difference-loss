"""
Train the SFN baseline model (without temporal loss)
"""
import numpy as np
import argparse, sys, os, time, glob
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.functional import F
import dataset, utils
from transformer_net import TransformerNet
from vgg16 import Vgg16

def center_crop(x, h, w):
  # Assume x is (N, C, H, W)
  H, W = x.size()[2:]
  if h == H and w == W: return x
  assert(h <= H and w <= W)
  dh, dw = H - h, W - w
  ddh, ddw = dh // 2, dw // 2
  return x[:, :, ddh:-(dh-ddh), ddw:-(dw-ddw)]

def train(args):
  np.random.seed(args.seed)
  torch.manual_seed(args.seed)
  torch.cuda.manual_seed(args.seed)
  kwargs = {'num_workers': 0, 'pin_memory': False}

  transform = transforms.Compose([
    transforms.Resize((
      args.image_size, args.image_size)),
      transforms.ToTensor(),
      transforms.Lambda(lambda x: x.mul(255))])
  train_dataset = dataset.CustomImageDataset(
    args.dataset,
    transform=transform,
    img_size=args.image_size)
  train_loader = DataLoader(train_dataset, batch_size=args.batch_size, **kwargs)

  transformer = TransformerNet(args.pad_type)
  transformer = transformer.train()
  optimizer = torch.optim.Adam(transformer.parameters(), args.lr)
  mse_loss = torch.nn.MSELoss()
  #print(transformer)
  vgg = Vgg16()
  vgg.load_state_dict(torch.load(os.path.join(args.vgg_model_dir, "vgg16.weight")))
  vgg.eval()

  transformer = transformer.cuda()
  vgg = vgg.cuda()

  style = utils.tensor_load_resize(args.style_image, args.style_size)
  style = style.unsqueeze(0)
  print("=> Style image size: " + str(style.size()))

  #(1, H, W, C)
  style = utils.preprocess_batch(style).cuda()
  utils.tensor_save_bgrimage(style[0].detach(), os.path.join(args.save_model_dir, 'train_style.jpg'), True)
  style = utils.subtract_imagenet_mean_batch(style)
  features_style = vgg(style)
  gram_style = [utils.gram_matrix(y).detach() for y in features_style]

  for e in range(args.epochs):
    train_loader.dataset.reset()
    agg_content_loss = 0.
    agg_style_loss = 0.
    iters = 0
    for batch_id, (x, _) in enumerate(train_loader):
      if x.size(0) != args.batch_size:
        print("=> Skip incomplete batch")
        continue
      iters += 1

      optimizer.zero_grad()
      x = utils.preprocess_batch(x).cuda()
      y = transformer(x)

      if (batch_id + 1) % 1000 == 0:
        idx = (batch_id + 1) // 1000
        utils.tensor_save_bgrimage(y.data[0],
          os.path.join(args.save_model_dir, "out_%d.png" % idx),
          True)
        utils.tensor_save_bgrimage(x.data[0],
          os.path.join(args.save_model_dir, "in_%d.png" % idx),
          True)

      xc = x.detach()

      y = utils.subtract_imagenet_mean_batch(y)
      xc = utils.subtract_imagenet_mean_batch(xc)

      features_y = vgg(y)
      features_xc = vgg(center_crop(xc, y.size(2), y.size(3)))
      
      #content target
      f_xc_c = features_xc[2].detach()
      # content
      f_c = features_y[2]

      content_loss = args.content_weight * mse_loss(f_c, f_xc_c)

      style_loss = 0.
      for m in range(len(features_y)):
        gram_s = gram_style[m]
        gram_y = utils.gram_matrix(features_y[m])
        batch_style_loss = 0
        for n in range(gram_y.shape[0]):
          batch_style_loss += args.style_weight * mse_loss(gram_y[n], gram_s[0])
        style_loss += batch_style_loss / gram_y.shape[0]

      total_loss = content_loss + style_loss

      total_loss.backward()
      optimizer.step()
      agg_content_loss += content_loss.data
      agg_style_loss += style_loss.data

      mesg = "{}\tEpoch {}:\t[{}/{}]\tcontent: {:.6f}\tstyle: {:.6f}\ttotal: {:.6f}".format(
        time.ctime(), e + 1, batch_id + 1, len(train_loader),
        agg_content_loss / iters,
        agg_style_loss / iters,
        (agg_content_loss + agg_style_loss) / iters)
      print(mesg)
      agg_content_loss = agg_style_loss = 0.0
      iters = 0

    # save model
    save_model_filename = "epoch_" + str(e) + "_" + str(args.content_weight) + "_" + str(args.style_weight) + ".model"
    save_model_path = os.path.join(args.save_model_dir, save_model_filename)
    torch.save(transformer.state_dict(), save_model_path)

  print("\nDone, trained model saved at", save_model_path)


def check_paths(args):
  try:
    if not os.path.exists(args.vgg_model_dir):
      os.makedirs(args.vgg_model_dir)
    if not os.path.exists(args.save_model_dir):
      os.makedirs(args.save_model_dir)
  except OSError as e:
    print(e)
    sys.exit(1)


def stylize(args):
  transform = transforms.Compose([
    transforms.Resize(256),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.mul(255))])
  ds = ImageFolder(args.input_dir, transform=transform)
  dl = DataLoader(ds, batch_size=1)
  print("=> Load from model file %s" % args.model_path)
  net = TransformerNet(args.pad_type)
  net.load_state_dict(torch.load(args.model_path))
  net = net.eval().cuda()
  
  utils.process_dataloader(args, net, dl)
  utils.generate_video(args, dl)


def main():
  main_arg_parser = argparse.ArgumentParser(description="parser for training the stylization networks.")
  subparsers = main_arg_parser.add_subparsers(
    title="subcommands", dest="subcommand")

  train_arg_parser = subparsers.add_parser("train",
    help="parser for training arguments")
  # loss
  train_arg_parser.add_argument("--content-weight",
    type=float, default=1.0,
    help="weight for content-loss, default is 1.0")
  train_arg_parser.add_argument("--style-weight",
    type=float, default=10.0,
    help="weight for style-loss, default is 10.0")
  # paths
  train_arg_parser.add_argument("--dataset",
    type=str, default="/home/share/datasets/coco/train2017",
    help="path to the DAVIS dataset")
  train_arg_parser.add_argument("--vgg-model-dir",
    type=str, default="pretrained/",
    help="directory for vgg, if model is not present in the directory it is downloaded")
  train_arg_parser.add_argument("--save-model-dir",
    type=str, default="exprs",
    help="path to folder where trained model will be saved.")
  train_arg_parser.add_argument("--style-image",
    type=str, default="data/styles/starry_night.jpg",
    help="path to style-image")
  # model config
  train_arg_parser.add_argument("--pad-type",
    type=str, default="interpolate-detach",
    help="interpolate-detach (default) | reflect-start | none | reflect | replicate | zero")
  # training config
  train_arg_parser.add_argument("--epochs",
    type=int, default=1,
    help="number of training epochs, default is 2.")
  train_arg_parser.add_argument("--batch-size",
    type=int, default=4,
    help="batch size for training, default is 4.")
  train_arg_parser.add_argument("--interval",
    type=int, default=1,
    help="Frame interval. Default is 1.")
  train_arg_parser.add_argument("--image-size",
    type=int, default=400,
    help="size of training images, default is 400 X 400")
  train_arg_parser.add_argument("--style-size",
    type=int, default=400,
    help="size of style-image, default is the original size of style image")
  train_arg_parser.add_argument("--seed",
    type=int, default=1234,
    help="random seed for training")
  train_arg_parser.add_argument("--lr",
    type=float, default=1e-3,
    help="learning rate, default is 0.001")
  eval_arg_parser = subparsers.add_parser("eval", help="parser for evaluation/stylizing arguments")
  # model config
  eval_arg_parser.add_argument("--model-type",
    type=str, default="sfn",
    help="sfn | rnn")
  eval_arg_parser.add_argument("--pad-type",
    type=str, default="interpolate-detach",
    help="interpolate-detach (default) | reflect-start | none | reflect | replicate | zero")
  # paths
  eval_arg_parser.add_argument("--input-dir",
    type=str, default="data/testin",
    help="Standard dataset: video_name1/*.jpg video_name2/*.jpg")
  eval_arg_parser.add_argument("--output-dir",
    type=str, default="data/testout",
    help="The output directory of generated images.")
  eval_arg_parser.add_argument("--model-path",
    type=str, default="",
    help="The path to saved model dir.")
  # others
  eval_arg_parser.add_argument("--model-name",
    type=str, default="",
    help="The model name (used in naming output videos).")
  eval_arg_parser.add_argument("--compute",
    type=int, default=1,
    help="Whether to generate new images or use existing ones.")

  args = main_arg_parser.parse_args()

  if args.subcommand is None:
    print("ERROR: specify either train or eval")
    sys.exit(1)

  if args.subcommand == "train":
    check_paths(args)
    train(args)
  elif args.subcommand == "eval":
    stylize(args)


if __name__ == "__main__":
  main()
