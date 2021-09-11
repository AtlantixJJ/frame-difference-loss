import sys, copy
from pymongo import MongoClient
import numpy as np
import util

client = util.client
db = client.userstudy
collection = db.score
userdb = db.user
summary = db.summary

def sanity_check(dic):
    flag = True
    idx = np.array(dic["index"])
    # sequential
    flag &= (idx[1:] == idx[:-1] + 1).all()
    # same length for options and choice
    flag &= len(dic["optionA"]) == len(dic["optionB"])
    flag &= len(dic["optionA"]) == len(dic["choice"])
    return flag

style_names = ['starrynight', 'lamuse', 'feathers', 'composition', 'candy', 'udnie', 'mosaic']
def get_style(name):
    flag = False
    for i, s in enumerate(style_names):
        if s in name:
            if flag:
                print(name)
            flag = True
            idx = i
            style = s
    if not flag:
        print(name)
    return idx, style

def inc_key_choice(key, choice, dic):
    try:
        dic[key]
    except KeyError:
        dic[key] = {}

    try:
        dic[key][choice]
    except KeyError:
        dic[key][choice] = 0
    
    dic[key][choice] += 1

# return the loss of video
def get_loss(name):
    ts = ['none', 'p-fdb', 'c-fdb', 'ofb']
    name = name[name.rfind("/")+1:]
    for t in ts:
        if t in name:
            return t

def get_model(s1, s2):
    if "rnn" in s1 or "rnn" in s2:
        return "rnn"
    return "sfn"

def get_compare_type(video, model, lossA, lossB):
    l = [lossA, lossB]
    l.sort()
    compare = " <- ".join(l)
    return f"{video}/{model}/{compare}"

def get_exprs():
    cursor = collection.find()
    # totally 16 experiments
    compare_loss_dic = [{} for _ in range(16)]
    style_compare_dic = [
        copy.deepcopy(compare_loss_dic)
        for _ in range(len(style_names))]
    for _ in range(cursor.count()):
        obj = cursor.next()
        if not sanity_check(obj):
            print("!> Error in id={} expr={}".format(
                obj["id"], obj["expr_id"]))
            continue
        expr_id = obj["expr_id"]
        lossA = get_loss(obj["optionA"][0])
        lossB = get_loss(obj["optionB"][0])
        model = get_model(obj["optionA"][0], obj["optionB"][0])
        video = "video" if ".mp4" in obj["optionA"][0] else "frame"
        compare_type = get_compare_type(video, model, lossA, lossB)
        for optionA, optionB, choice in zip(obj["optionA"], obj["optionB"], obj["choice"]):
            lossA = get_loss(optionA)
            lossB = get_loss(optionB)
            style_idx, style = get_style(optionB)
            chosen = lossA if choice == 0 else lossB
            inc_key_choice(
                compare_type,
                chosen,
                compare_loss_dic[expr_id])
            inc_key_choice(
                style + "/" + compare_type,
                chosen,
                style_compare_dic[style_idx][expr_id])

    # store to db
    res = [["expr id", "video/frame", "model", "name", "ratio", "total", "styles"]]
    for expr_id in range(16):
        dic = compare_loss_dic[expr_id]
        if dic == {}:
            continue

        compare_type = list(dic.keys())[0]
        dic = list(dic.values())[0]
        total = sum(list(dic.values()))
        video, model, cm = compare_type.split("/")
        left = cm.split(" <- ")[0]
        if left not in dic.keys():
            dic[left] = 0
        ratio = dic[left] / float(total)

        # get style dic
        style_dic = {}
        for i, s in enumerate(style_names):
            dic = style_compare_dic[i][expr_id]
            if dic == {}:
                continue
            dic = list(dic.values())[0]
            if left not in dic.keys():
                dic[left] = 0
            style_dic[s] = dic[left] / float(sum(list(dic.values())))

        summary.update(
            {"expr_id" : expr_id},
            {
                "video/frame" : video,
                "model" : model,
                "name" : compare_type,
                left : ratio,
                "total" : total,
                "styles" : style_dic
            })
            
        str_style_dic = str(style_dic)[1:-2].replace("\'", "")
        res.append([
            str(expr_id),
            video,
            model,
            compare_type,
            f"{ratio:.3f}",
            str(int(total)),
            str_style_dic])

    return res


def get_full_record():
    cursor = collection.find()
    # totally 16 experiments
    keys = ['id', 'expr_id', 'optionA', 'optionB', 'choice', 'time', 'ip']
    res = [keys]
    for _ in range(cursor.count()):
        obj = cursor.next()
        if not sanity_check(obj):
            print("!> Error in id={} expr={}".format(
                obj["id"], obj["expr_id"]))
            continue
        for i in range(len(obj["time"])):
            a = []
            for key in keys:
                if type(obj[key]) is list:
                    a.append(str(obj[key][i]))
                else:
                    a.append(str(obj[key]))
            res.append(a)
    return res


if __name__ == "__main__":
    import util
    with open("expr_record.csv", "w") as f:
        f.write(util.format_to_csv(get_exprs()))
    with open("full_record.csv", "w") as f:
        f.write(util.format_to_csv(get_full_record()))