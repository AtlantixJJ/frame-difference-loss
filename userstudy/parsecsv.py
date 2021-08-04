# Parse from csv
import os
import glob
import sys
import csv, pprint
import numpy as np
import pprint

# sfn_comb_diff: Batch_3452048_batch_results
# Frame: Batch_3452051_batch_results
# MSRA:COMB Batch_3477710_batch_results
csvfile = open("document/videostability_sfn_flow_none_4_Batch_3869707_batch_results.csv", 'r')
csv_reader = csv.reader(csvfile, dialect='excel')
header = csv_reader.next()

cnt = 0
compare = {}
position_compare = {}
video_compare = {}

def get_video_name(name):
    for ts in ['none', 'diff', 'comb', 'msra', 'flow', 'zero']:
        name = name.replace(ts, "")
    for model in ["sfn", "rnn", "adain"]:
        name = name.replace(model, "")
    
    name = name.replace("___", "_")
    return name

def get_type(name):
    #ts = ['diff', 'flow']
    ts = ['none', 'diff', 'comb', 'msra', 'flow', 'zero']
    name = name[name.rfind("/")+1:]
    for t in ts:
        if t in name:
            return t

def get_choice(left, right, name):
    if name == "optionA":
        return left
    else:
        return right

def add_key_item(key, choice, dic):
    try:
        dic[key]
    except KeyError:
        dic[key] = {}

    try:
        dic[key][choice]
    except KeyError:
        dic[key][choice] = 0
    
    dic[key][choice] += 1

def get_name(name):
    ind = name.rfind("/")
    name = name[ind+1:]
    name = name.replace("sintel_", "").replace("davis_", "")
    name = name.replace("_stylized.mp4", "")
    return name.strip()

def find_column(col_name, header):
    for i, n in enumerate(header):
        if col_name in n:
            return i

count_dic = {}

left_ind = header.index("Input.video_A_url")
right_ind = header.index("Input.video_B_url")
opt_ind = header.index("Answer.choice")
reject = header.index("RejectionTime")

while True:
    try:
        l = csv_reader.next()
    except StopIteration:
        break
    
    if "PDT" in l[reject]:
      continue
    
    leftKey = get_type(l[left_ind])
    rightKey = get_type(l[right_ind])

    leftName = get_name(l[left_ind])
    rightName = get_name(l[right_ind])

    commonName = get_video_name(leftName)

    if sys.argv[1] not in leftName: continue
    
    cnt += 1

    tmp_ = [leftName, rightName]
    tmp_.sort()
    choice_name = tmp_[0] + "_" + tmp_[1]

    choice = get_choice(leftKey, rightKey, l[opt_ind])
    tmp_ = [leftKey, rightKey]
    tmp_.sort()
    choice_type = str(tmp_)

    add_key_item(choice_type, choice, compare)
    add_key_item(choice_type, leftKey, position_compare)
    add_key_item(commonName, choice, video_compare)

    if choice in tmp_[0]:
        try:
            count_dic[choice_name] += 1
        except KeyError:
            count_dic[choice_name] = 1

print("Total: %d" % cnt)
print("Result")

def print_dic(dic):
    print("|type|choice A|choice B|")
    print("|:--|:--|:--|")
    def procprint(key):
        key = key.replace("\'", "").replace("[", "").replace("]", "")
        key = key.replace(",", " v.s.")
        return key

    def procv(v):
        v = v.replace("\'", "").replace("{", "").replace("}", "")
        v = v.replace(",", "\t|")
        return v

    for k,v in dic.items():
        a = "| " + procprint(k) + " | " + procv(pprint.pformat(v)) + "\t|" 
        print(a)

def print_latex(dic):
    def procprint(key):
        key = key.replace("\'", "").replace("[", "").replace("]", "")
        key = key.replace(",", " v.s.")
        return key

    s = ""
    for v in dic.values()[::-1]:
        if "comb" in v.keys():
            s += "comb "
            val = v["comb"]
        elif "flow" in v.keys():
            s += "flow "
            val = v["flow"]
        if val > 600:
            div = 16.
        else:
            div = 4.
        s += "%.2f" % (val / div)
        s += "\% & "
    print(s)

def save_csv(filename, dic):
    def procprint(key):
        key = key.replace("\'", "").replace("[", "").replace("]", "")
        key = key.replace(",", " v.s.")
        return key

    f = open(filename, "w")
    heads = []
    lines = []
    for key, v in dic.items()[::-1]:
        vals = []
        for k in v.keys():
            if k not in heads:
                heads.append(k)
            vals.append(v[k])
        total = sum(vals)
        vals = vals + [float(v) / total for v in vals]
        lines.append(",".join([key] + [str(v) for v in vals]) + "\n")
    heads = ["type"] + heads + [h + "_ratio" for h in heads]
    f.write(",".join(heads) + "\n")
    for l in lines:
        f.write(l)
    f.close()

print_dic(compare)
save_csv("video_compare.csv", video_compare)

def get_mid(dic):
    l = 0
    for k, v in dic.items():
        if "diff" in k and "flow" in k:
            l += v
            print(k, v)

# get_mid(count_dic)

# man-bike_vsfn_diff_composition_flow
# https://github.com/AtlantixJJ/VideoStableData/raw/master/data/video_frames1int/davis_man-bike_vsfn_flow_composition_stylized.mp4

def parse_print_my():
    count_dic = {}
    compare = {}
    position_compare = {}
    cnt = 0
    drop_same = 0

    # read score json
    with open(args.input, 'r') as f:
        lines = f.readlines()
    records = [eval(line) for line in lines]

    for record in records:
        # validate result
        if 'wav1' not in record.keys():
            continue
        cnt += 1

        base_score = record['scores']
        for score in record['scores'][1:]:
            if base_score != score:
                break
        else:
            drop_same += 1
            continue

        for wav1, wav2, score, ip in zip(record['wav1'], record['wav2'], record['scores'], record['ip']):
            l = [wav1, wav2, score]

            leftKey = get_type(l[-3])
            rightKey = get_type(l[-2])

            leftName = get_name(l[-3])
            rightName = get_name(l[-2])
            tmp_ = [leftName, rightName]
            tmp_.sort()
            choice_name = tmp_[0] + "_" + other_name(tmp_[1])

            choice = [leftKey, rightKey][score]
            tmp_ = [leftKey, rightKey]
            tmp_.sort()
            choice_type = str(tmp_)
            
            add_key_item(choice_type, choice, compare)
            add_key_item(choice_type, leftKey, position_compare)

            if choice in tmp_[0]:
                try:
                    count_dic[choice_name] += 1
                except KeyError:
                    count_dic[choice_name] = 1
    
    print(compare)
