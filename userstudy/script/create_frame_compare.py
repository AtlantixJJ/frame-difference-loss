"""
Select candy lamuse SFN, MSRA comparison.
"""
import glob, os
import numpy as np
from os.path import join as osj

testexamples = ['davis_man-bike', 'sintel_ambush_1', 'sintel_bamboo_3', 'sintel_market_1', 'sintel_mountain_2', 'sintel_PERTURBED_shaman_1', 'sintel_tiger', 'sintel_temple_1', 'sintel_wall', 'sintel_cave_3', 'sintel_market_4', 'davis_slackline', 'davis_cats-car', 'davis_girl-dog', 'davis_helicopter', 'davis_guitar-violin', 'davis_subway', 'davis_gym', 'davis_horsejump-stick', 'davis_tandem']
model = ["sfn_diff_candy", "sfn_diff_lamuse", "sfn_comb_candy", "sfn_comb_lamuse", "sfn_none_candy", "sfn_none_lamuse"]
dir1 = osj("/", "home", "xujianjing", "testdata", "%s", "%s")
dir2 = osj("D:\\", "past", "msra", "%s", "%s")

def func1():
    basedir = dir2
    model = ["candy", "lamuse"]
    destdir = osj("D:\\", "past", "msra", "dest")
    rng = np.random.RandomState(1314)
    paths = []
    for m in model:
        for f in testexamples:
            png_path = osj(basedir, "*.png")#.replace("\\", "\\\\")
            jpg_path = osj(basedir, "*.jpg")#.replace("\\", "\\\\")

            img_names = glob.glob(png_path % (m, f)) + glob.glob(jpg_path % (m, f))
            img_names.sort()
            total_num = len(img_names)

            rng.shuffle(img_names)
            selected = img_names[:3]
            paths.extend(selected)

            for p in selected:
                index = p.rfind("\\")
                index = p[index+1:]
                os.system("copy %s temp\\%s" % (p, f + "_msra_" + m + "_" + index))

basedir = dir1
# frame quality comparison
# feat : none (=)
# feat : msra
# feat : sfn
# (done) sfn : none
# sfn  : msra (*)
# none : msra (*=)
msra_dir = osj("/", "home", "xujianjing", "userstudy", "static", "data", "frame_compare")
destdir = osj("/", "home", "xujianjing", "userstudy", "static", "data", "VideoStableData", "data", "frame_compare_msra_sfn_comb_none")
os.system("mkdir " + destdir)
paths = []
for i,m in enumerate(model):
    if i % 2 == 0: rng = np.random.RandomState(1314)
    path_model = []
    for f in testexamples:
        png_path = osj(basedir, "*.png")#.replace("\\", "\\\\")
        jpg_path = osj(basedir, "*.jpg")#.replace("\\", "\\\\")

        img_names = glob.glob(png_path % (f, m)) + glob.glob(jpg_path % (f, m))
        img_names.sort()
        total_num = len(img_names)

        rng.shuffle(img_names)
        selected = img_names[:1]
        path_model.extend(selected)

        for p in selected:
            index = p.rfind("/")
            index = p[index+1:]
            name = osj(destdir, f + "_" + m + "_" + index)
            # print(p + " " + name)
            os.system("cp %s %s" % (p, name))
            if "diff" in m:
                # cp msra
                p = osj(msra_dir, f + "_" + m.replace("sfn_diff", "msra") + "_" + index)
                name = osj(destdir, f + "_" + m.replace("sfn_diff", "msra") + "_" + index)
                print(p + " " + name)
                os.system("cp %s %s" % (p, name))
    paths.append(path_model)

