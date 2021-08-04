"""
Write all the data list
"""
ctrl_list1 = """davis_man-bike_sfn_none_lamuse_stylized.mp4
davis_guitar-violin_sfn_none_lamuse_stylized.mp4
davis_cats-car_sfn_none_composition_stylized.mp4
sintel_temple_1_sfn_none_starrynight_stylized.mp4
davis_slackline_sfn_none_composition_stylized.mp4
davis_girl-dog_sfn_none_feathers_stylized.mp4
sintel_market_4_sfn_none_lamuse_stylized.mp4
davis_helicopter_sfn_none_lamuse_stylized.mp4
sintel_wall_sfn_none_lamuse_stylized.mp4
sintel_market_1_sfn_none_starrynight_stylized.mp4
davis_subway_sfn_none_feathers_stylized.mp4
sintel_ambush_1_sfn_none_feathers_stylized.mp4
sintel_bamboo_3_sfn_none_feathers_stylized.mp4
sintel_cave_3_sfn_none_lamuse_stylized.mp4
davis_horsejump-stick_sfn_none_starrynight_stylized.mp4
sintel_mountain_2_sfn_none_feathers_stylized.mp4
davis_tandem_sfn_none_lamuse_stylized.mp4
sintel_PERTURBED_shaman_1_sfn_none_composition_stylized.mp4
sintel_tiger_sfn_none_feathers_stylized.mp4
davis_gym_sfn_none_starrynight_stylized.mp4""".split("\n")

ctrl_list2 = """davis_man-bike_sfn_flow_lamuse_stylized.mp4
davis_guitar-violin_sfn_diff_lamuse_stylized.mp4
davis_cats-car_sfn_diff_composition_stylized.mp4
sintel_temple_1_sfn_flow_starrynight_stylized.mp4
davis_slackline_sfn_flow_composition_stylized.mp4
davis_girl-dog_sfn_diff_feathers_stylized.mp4
sintel_market_4_sfn_flow_lamuse_stylized.mp4
davis_helicopter_sfn_diff_lamuse_stylized.mp4
sintel_wall_sfn_flow_lamuse_stylized.mp4
sintel_market_1_sfn_diff_starrynight_stylized.mp4
davis_subway_sfn_flow_feathers_stylized.mp4
sintel_ambush_1_sfn_diff_feathers_stylized.mp4
sintel_bamboo_3_sfn_diff_feathers_stylized.mp4
sintel_cave_3_sfn_diff_lamuse_stylized.mp4
davis_horsejump-stick_sfn_flow_starrynight_stylized.mp4
sintel_mountain_2_sfn_flow_feathers_stylized.mp4
davis_tandem_sfn_flow_lamuse_stylized.mp4
sintel_PERTURBED_shaman_1_sfn_diff_composition_stylized.mp4
sintel_tiger_sfn_flow_feathers_stylized.mp4
davis_gym_sfn_diff_starrynight_stylized.mp4""".split("\n")

import os, glob, sys
import numpy as np
rng = np.random.RandomState(1)
dirs = ["static/data/fdb_ofb_none_sfn_rnn_adain_data", "static/data/msra_sfn_video_compare_data"]

def get_list(base_dir, filters=['diff', 'msra']):
    list1, list2 = [], []
    mp4list = glob.glob(base_dir + '/*.mp4')
    mp4list.sort()
    for mp4file in mp4list:
        filename = mp4file.split('/')[-1]
        if filters[0] in filename: list1.append(mp4file)
        elif filters[1] in filename: list2.append(mp4file)
    return list1, list2

def write_list(name, l):
    with open(name, "w") as f:
        for line in l:
            f.write(line + "\n")

list1, list2 = get_list(dirs[0], ['diff', 'flow'])
ctrl_list1 = [dirs[0] + '/' + n for n in ctrl_list1]
ctrl_list2 = [dirs[0] + '/' + n for n in ctrl_list2]
list1.extend(ctrl_list1)
list2.extend(ctrl_list2)
rng.shuffle(list1)
rng.shuffle(list2)
write_list(dirs[0] + "_list1.txt", list1)
write_list(dirs[0] + "_list2.txt", list2)

list1, list2 = get_list(dirs[1], ['diff', 'msra'])
rng.shuffle(list1)
rng.shuffle(list2)
write_list(dirs[1] + "_list1.txt", list1)
write_list(dirs[1] + "_list2.txt", list2)