# Write list and form csv
import os
import glob
import sys
import csv
import numpy as np

mp4list=glob.glob(sys.argv[1] + '*.mp4')
mp4list.sort()

os.system("rm list[0-9].txt")
lists = [[], [], []]

arg = "https://raw.githubusercontent.com/AtlantixJJ/VideoStableVideo%s/master/data/video_frames%dint/"
base_arg = "https://raw.githubusercontent.com/AtlantixJJ/VideoStableVideo%s/master/data/video_none/"

if len(sys.argv) >= 3:
    arg = arg % (sys.argv[2], 1) # 1 or 2
    base_arg = base_arg % sys.argv[2]
else:
    arg = arg % ("", 1) # 1 or 2
    base_arg = base_arg % ""

def proc1(l):
    ind = l.rfind("/")
    return base_arg+l[ind+1:]

def procn(l):
    ind = l.rfind("/")
    return arg+l[ind+1:]

ctrl_lists = [[], [], []]
for i in range(3):
    if i == 0:
        ctrl_lists[i] = [proc1(l) for l in open("ctrl_list%d.txt" % (i+1)).readlines()]
    else:
        ctrl_lists[i] = [procn(l) for l in open("ctrl_list%d.txt" % (i+1)).readlines()]
print(ctrl_lists[0])

for mp4file in mp4list:
    # each list should have 80 videos, and they are sorted
    if 'diff' in mp4file.split('_'):
        os.system('echo %s >> %s'%(mp4file,'list1.txt'))
        lists[0].append(procn(mp4file))
    elif 'flow' in mp4file.split('_'):
        os.system('echo %s >> %s'%(mp4file,'list2.txt'))
        lists[1].append(procn(mp4file))
    elif 'zero' in mp4file.split('_'):
        os.system('echo %s >> %s'%(mp4file,'list3.txt'))
        lists[2].append(procn(mp4file))

# ctrl_list1.txt : none
# ctrl_list2.txt : sfn
# ctrl_list3.txt : vsfn

# form pair
def add_pair_from_list(list1, list2):
    pair = []
    for l1, l2 in zip(list1, list2):
        pair.append((l1, l2))
    return pair

tot_pair = []

for i in range(3):
    for j in range(i+1, 3):
        tot_pair.extend(add_pair_from_list(lists[i], lists[j]))

tot_pair.extend(add_pair_from_list(ctrl_lists[0], ctrl_lists[1]))

# swith pair
for i in range(len(tot_pair)):
    if np.random.uniform() < 0.5:
        tot_pair[i] = tot_pair[i][1], tot_pair[i][0]

# shuffle order
s = np.random.RandomState(1092411)
s.shuffle(tot_pair)

outfile = "exam.csv"
out = open(outfile, 'w')
csv_writer = csv.writer(out, dialect='excel')
header = ["video_A_url", "video_B_url"]
csv_writer.writerow(header)

for p in tot_pair:
    csv_writer.writerow(p)