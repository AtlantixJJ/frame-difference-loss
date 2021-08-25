import os, glob, csv
import numpy as np

def get_style_url(name):
    styles = ["lamuse", "candy"]
    style_url = [
        "https://github.com/AtlantixJJ/VideoStableData/raw/master/images/la_muse_tr.jpg",
        "https://github.com/AtlantixJJ/VideoStableData/raw/master/images/candy_tr.jpg"]

    for i, s in enumerate(styles):
        if s in name:
            return style_url[i]

def pair_list(l1, l2):
    l = []
    for a, b in zip(l1, l2):
        l.append((a, b))
    return l

def func1():
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

def func2():
    basedir = "/home/xujianjing/userstudy/static/data/VideoStableData/"
    dir2 = "data/frame_compare_msra_sfn_comb_none/"
    baseurl = "https://raw.githubusercontent.com/AtlantixJJ/VideoStableData/master/data/frame_compare_msra_sfn_comb_none/"
    file_type = ["msra", "diff", "comb", "none"]
    files = [glob.glob(basedir + dir2 + ("*%s*" % t)) for t in file_type]
    for f in files: f.sort()
    for i in range(len(files)):
        for j in range(len(files[i])):
            ind = files[i][j].rfind("/")
            files[i][j] = files[i][j][ind+1:]
    for t in file_type:
        print(basedir + dir2 + ("*%s*" % t)) 
    l = []
    for i in range(len(file_type)):
        for j in range(i+1, len(file_type)):
            l.extend(pair_list(files[i], files[j]))
    print(len(l))
    print(l[0][0])
    print(l[0][1])

    # shuffle
    rng = np.random.RandomState(1314)
    rng.shuffle(l)
    for i in range(len(l)):
        if np.random.rand() < 0.5:
            l[i] = (l[i][1], l[i][0])

    outfile = basedir + "frame_compare_msra_sfn_comb_none.csv"
    out = open(outfile, 'w')
    csv_writer = csv.writer(out, dialect='excel')
    header = ["image_A_url", "image_B_url", "image_url"]
    csv_writer.writerow(header)

    for it in l:
        p = [0,0,0]
        p[0] = baseurl + it[0]
        p[1] = baseurl + it[1]
        p[2] = get_style_url(it[0])
        csv_writer.writerow(p)

func2()