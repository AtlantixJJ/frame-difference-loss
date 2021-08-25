import csv, argparse, cv2, glob
import numpy as np
from os.path import join as osj

SEED = {
  "rnn_c-fdb_ofb_video" : 1397,
  "sfn_c-fdb_ofb_video" : 11973,
  "sfn_c-fdb_p-fdb_video" : 91851,
}


def extract_frames(dic):
  for k, v in dic.items():
    indice = np.unique(np.array(v))
    video = cv2.VideoCapture(k)  
    n_frame = video.get(cv2.CAP_PROP_FRAME_COUNT)
    for idx in indice:
      fp = k.replace("video", "frame").replace(
        ".mp4", f"_{idx:04d}.jpg")
      video.set(cv2.CAP_PROP_POS_FRAMES, idx % n_frame)
      frame = video.read()[1]
      cv2.imwrite(fp, frame)


if __name__ == "__main__":
  files = glob.glob("static/expr/csv/*video.csv")
  files.sort()
  for f in files:
    csv_reader = csv.reader(open(f, "r"), dialect='excel')
    header = next(csv_reader)
    for k, v in SEED.items():
      if k in f:
        break
    rng = np.random.RandomState(v)
    lists = []
    dic = {}
    while True:
      try:
        vp1, vp2 = next(csv_reader)
      except StopIteration:
        break
      
      idx = rng.randint(0, 10000)
      if vp1 not in dic:
        dic[vp1] = [idx]
      else:
        dic[vp1].append(idx)
      if vp2 not in dic:
        dic[vp2] = [idx]
      else:
        dic[vp2].append(idx)
      fp1 = vp1.replace("video", "frame").replace(".mp4", f"_{idx:05d}.jpg")
      fp2 = vp2.replace("video", "frame").replace(".mp4", f"_{idx:05d}.jpg")
      lists.append([fp1, fp2])
    rng.shuffle(lists)
    for i in range(len(lists)):
      rng.shuffle(lists[i])
    
    csv_fpath = f.replace("video", "frame")
    csv_writer = csv.writer(open(csv_fpath, "w"), dialect='excel')
    header = ["frame_A_path", "frame_B_path"]
    csv_writer.writerow(header)
    for p in lists:
        csv_writer.writerow(p)

  extract_frames(dic)