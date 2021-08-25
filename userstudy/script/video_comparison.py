import csv, argparse, glob, os
import numpy as np
from os.path import join as osj

video_names = ['man-bike', 'ambush_1', 'temple_1', 'bamboo_3', 'market_1', 'mountain_2', 'PERTURBED_shaman_1', 'tiger', 'wall', 'cave_3', 'market_4', 'slackline', 'cats-car', 'girl-dog', 'helicopter', 'guitar-violin', 'subway', 'gym', 'horsejump-stick', 'tandem']
styles = ["starrynight", "lamuse", "feathers", "composition"]


def main(args):
  t_ = [args.s1, args.s2]
  t_.sort()
  s1, s2 = t_
  csv_fpath = osj(f"{args.output}", f"{args.net}_{s1}_{s2}_video.csv")
  rng = np.random.RandomState(args.seed)
  lists = []

  for v in video_names:
    for s in styles:
      s1_name = f"{args.input}/{v}_{args.net}_{s1}_{s}_{args.pad}.mp4"
      s2_name = f"{args.input}/{v}_{args.net}_{s2}_{s}_{args.pad}.mp4"
      if not os.path.exists(s1_name) or not os.path.exists(s2_name):
        print(f"!> {s1_name} not exists!")
        continue
      lists.append([s1_name, s2_name])
  
  rng.shuffle(lists)
  for i in range(len(lists)):
    rng.shuffle(lists[i])
  
  csv_writer = csv.writer(open(csv_fpath, "w"), dialect='excel')
  header = ["video_A_path", "video_B_path"]
  csv_writer.writerow(header)

  for p in lists:
      csv_writer.writerow(p)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()

  parser.add_argument("--s1", default="c-fdb",
    help="Compare loss subject 1.")
  parser.add_argument("--s2", default="none",
    help="Compare loss subject 2.")
  parser.add_argument("--net", default="sfn",
    help="Type of network.")
  parser.add_argument("--pad", default="interpolate-detach",
    help="Type of padding.")
  parser.add_argument("--input", default="static/expr/video",
    help="Input directory.")
  parser.add_argument("--output", default="static/expr/csv",
    help="Output directory.")
  parser.add_argument("--seed", default=1997, type=int)
  args = parser.parse_args()

  if args.s1 == "all":
    exprs = [
      ["c-fdb", "ofb", "sfn"],
      ["c-fdb", "ofb", "rnn"],
      ["c-fdb", "p-fdb", "sfn"]]
    for s1, s2, net in exprs:
      args.s1, args.s2, args.net = s1, s2, net
      main(args)
  else:
    main(args)





  
