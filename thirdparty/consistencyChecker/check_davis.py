import tqdm, os, glob, subprocess
from os.path import join as osj

DATA_DIR = "data/DAVIS"
FLOW_DIR = "Flows/480p"
OCC_DIR = "Occlusion/480p"
C_PATH = "thirdparty/consistencyChecker/consistencyChecker"

flow_folders = glob.glob(osj(DATA_DIR, FLOW_DIR, "*"))
flow_folders.sort()

args_list = []
for folder in flow_folders:
    flow_files = glob.glob(osj(folder, "*.flo"))
    f_files = list(filter(lambda x: "forward" in x, flow_files))
    b_files = list(filter(lambda x: "backward" in x, flow_files))
    f_files.sort()
    b_files.sort()
    occ_folder = folder.replace("Flows", "Occlusions")
    if not os.path.exists(occ_folder):
        os.makedirs(occ_folder)
    for i in range(len(f_files)):
        occ_file = f_files[i].replace("Flows", "Occlusions").replace("forward", "reliable").replace(".flo", ".pgm")
        args_list.append([f_files[i], b_files[i], occ_file])

for args in tqdm.tqdm(args_list):
    res = subprocess.run([C_PATH] + args)