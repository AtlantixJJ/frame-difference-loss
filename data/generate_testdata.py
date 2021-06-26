import os

if not os.path.exists("testin"):
    os.mkdir("testin")

davis_vnames = ["cats-car","girl-dog","guitar-violin","gym","helicopter","horsejump-stick","man-bike","slackline","subway","tandem"]
davis_fpath = "DAVIS/JPEGImages/480p"
sintel_vnames = ["ambush_1","bamboo_3","market_1","mountain_2","PERTURBED_shaman_1","tiger","ambush_3","cave_3","temple_1","wall"]
sintel_fpath = "MPI-Sintel-complete/test/final"

for vn in davis_vnames:
    os.system(f"cp -r {davis_fpath}/{vn} testin")
for vn in sintel_vnames:
    os.system(f"cp -r {sintel_fpath}/{vn} testin")