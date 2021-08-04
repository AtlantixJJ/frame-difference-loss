#!/usr/bin/env python
# encoding: utf-8
from __future__ import division
import json
import argparse
import time
import os

parser = argparse.ArgumentParser(description='parse the BSON file.')
parser.add_argument('--input', dest='input', type=str)
args = parser.parse_args()

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
    ts = ['none', 'diff', 'comb', 'msra', 'flow', 'zero']
    name = name[name.rfind("/")+1:]
    for t in ts:
        if t in name:
            return t

def get_compare_type(lossA, lossB):
    l = [lossA, lossB]
    l.sort()
    return " ".join(l)

os.system('mongoexport -d userstudy -c score -o score --port 7771')
record_string = open("userstudy").readlines()
compare_loss_dic = {} # flow, none

for s in record_string:
    record = json.loads(s)
    for optionA, optionB, choice in zip(record["optionA"], record["optionB"], record["choice"]):
        lossA, lossB = get_loss(optionA), get_loss(optionB)
        compare_type = get_compare_type(lossA, lossB)
        chosen = lossA if choice == 0 else lossB
        inc_key_choice(compare_type, chosen, compare_loss_dic)
        print(f"{optionA}")
        print(f"{optionB}")
        print(f"{compare_type} {chosen} {choice}")

print(compare_loss_dic)
 

