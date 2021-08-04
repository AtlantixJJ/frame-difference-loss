import os
import glob
basedir = "frame_compare/"
msra_files = glob.glob(basedir + "*msra*.jpg")
sfn_files = glob.glob(basedir + "*sfn*.jpg")
msra_files.sort()
sfn_files.sort()
def write_list(p, l):
  f = open(p, "w")
  for it in l:
    f.write("static/data/"+ it + "\n")
write_list("frame_list1.txt", msra_files)
write_list("frame_list2.txt", sfn_files)