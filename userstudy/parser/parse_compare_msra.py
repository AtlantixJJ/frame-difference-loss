#!/usr/bin/env python
# encoding: utf-8
"""
Parse the result of video stability comparison with MSRA method.
"""
from __future__ import division
import json
import argparse
import time
import os

parser = argparse.ArgumentParser(description='parse the BSON file.')
parser.add_argument('--input', dest='input', type=str)

datasets = ['davis', 'sintel']
styles = ['candy', 'lamuse']
models = ['msra', 'sfn']

def extract_class(s):
    labels = {}
    ind = s.rfind("/")
    s = s[ind+1:]
    for dataset in datasets:
        if dataset in s:
            labels['dataset'] = dataset
            break
    for style in styles:
        if style in s:
            labels['style'] = style
            break
    for model in models:
        if model in s:
            labels['model'] = model
            break
    return labels


def count_none_table(label_score, name):

    score_table = {}
    for model in models:
        score_table[model] = [0.0, 0]
    for example in label_score:
        wl1, wl2, score = example

        if name not in [wl1['work'], wl2['work']]:
            continue

        for model in models:
            if wl1['model'] == model:
                score_table[model][0] += score if wl1['work'] == name else 1-score
                score_table[model][1] += 1
                break
    return score_table


def count_table(label_score, lambdas):
    score_table = {}
    for model in models:
        score_table[model] = [0.0, 0]
    for example in label_score:
        wl1, wl2, score = example
        if not all([l(wl1) for l in lambdas]):
            continue
        #print(wl1, wl2, score)

        for model in models:
            score_table[model][0] += score if wl1['model'] == model else 1-score
            score_table[model][1] += 1
    return score_table


def parse_print():
    with open(args.input, 'r') as f:
        lines = f.readlines()

    records = [eval(line) for line in lines]
    record_dict = {}
    record_len_dict = {}

    ip_dict = {}
    drop_same = 0

    label_score = []

    for record in records:
        if 'wav1' not in record.keys():
            continue
        base_score = record['scores']
        print(base_score)
        for score in record['scores'][1:]:
            if base_score != score:
                break
            else:
                drop_same += 1
                continue
        print(record['ip'])
        for wav1, wav2, score, ip in zip(record['wav1'], record['wav2'], record['scores'], record['ip']):
            wav1_label = extract_class(wav1)
            wav2_label = extract_class(wav2)
            print(wav1_label)

            if len(wav1_label) == 3 and len(wav2_label) == 3:
                label_score.append([wav1_label, wav2_label, score])
            else:
                print(wav1, wav2)
                print(wav1_label, wav2_label)

    normal_score = list(label_score)
    print('\nNormal count.')
    print('FDB < MSRA')
    table = count_table(normal_score, [])
    print(table)

    for key in table.keys():
        print(key, 'True: {} All: {} Percent: {:.3f}'.format(table[key][0], table[key][1], table[key][0] / (1e-6 + table[key][1])))

    for style in styles:
        print('\nCount by style: {}'.format(style))
        table = count_table(normal_score, lambdas=[lambda x: x['style'] == style])

        for key in table.keys():
            print(key, 'True: {} All: {} Percent: {:.3f}'.format(table[key][0], table[key][1], table[key][0] / (1e-6 + table[key][1])))

    for dataset in datasets:
        print('\nCount by dataset: {}'.format(dataset))
        table = count_table(normal_score, lambdas=[lambda x: x['dataset'] == dataset])

        for key in table.keys():
            print(key, 'True: {} All: {} Percent: {:.3f}'.format(table[key][0], table[key][1], table[key][0] / (1e-6 + table[key][1])))

if __name__ == '__main__':
    global args
    args = parser.parse_args()

    count = 0
    while True:
        print('Count: {}'.format(count))
        # this port need to be the same as mongod port
        os.system('mongoexport --port 27018 -d {} -c score -o {}'.format(args.input, args.input))
        parse_print()
        time.sleep(60)

        count += 1