# Parse from csv
import os
import glob
import sys
import csv
import numpy as np
import pprint

csvfile = open("exam1.csv", 'r')
csv_reader = csv.reader(csvfile, dialect='excel')
header = csv_reader.next()

position_compare = {}

def get_type(name):
    ts = ['none', 'diff', 'flow', 'zero']
    for t in ts:
        if t in name:
            return t

def add_key_item(key, item, dic):
    try:
        position_compare[key_name]
    except KeyError:
        position_compare[key_name] = {}
    
    try:
        position_compare[key_name][leftKey] += 1
    except KeyError:
        position_compare[key_name][leftKey] = 1

while True:
    try:
        l = csv_reader.next()
    except StopIteration:
        break
    
    leftKey, rightKey = get_type(l[0]), get_type(l[1])
    tmp = [leftKey, rightKey]
    tmp.sort()
    key_name = tmp[0] + "_" + tmp[1]
    add_key_item(key_name, leftKey, position_compare)

print(position_compare)
