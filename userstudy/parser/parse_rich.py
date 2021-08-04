#!/usr/bin/env python
# encoding: utf-8

from __future__ import division
import json
import argparse
import time
import os

parser = argparse.ArgumentParser(description='parse the BSON file.')
parser.add_argument('--input', dest='input', type=str)

# datasets = ['davis', 'sintel']
datasets = ['davis']
styles = ['candy', 'lamuse']
works = ['bs', 'sfndiff']
# models = ['rnn', 'adain', 'sfn']
# models = ['sfn']


def extract_class(s):
    labels = {}
    for dataset in datasets:
        if dataset in s:
            labels['dataset'] = dataset
            break
    for style in styles:
        if style in s:
            labels['style'] = style
            break
    for work in works:
        if work in s:
            labels['work'] = work
            break
    return labels


def count_none_table(label_score, name, lambdas):

    score_table = dict((x, 0.0) for x in name)
    score_table.update({'difference': 0.0, 'sum': 0.0, 'per1': 0.0, 'per2': 0.0})
    for example in label_score:
        wl1, wl2, score = example

        if not all([l(wl1) for l in lambdas]):
            continue

        score_table[wl1['work']] += score
        score_table[wl2['work']] += 1 - score

    score_table['difference'] = score_table[name[0]] - score_table[name[1]]
    score_table['sum'] = score_table[name[0]] + score_table[name[1]]
    score_table['per1'] = score_table[name[0]] / score_table['sum']
    score_table['per2'] = score_table[name[1]] / score_table['sum']
    return score_table


def count_table(label_score, name, lambdas):

    score_table = {}
    for example in label_score:
        wl1, wl2, score = example
        if not all([l(wl1) for l in lambdas]):
            continue

        for model in models:
            if wl1['model'] == model:
                score_table[model][wl1['work']] += score
                score_table[model][wl2['work']] += 1 - score
                break
    for model in models:
        score_table[model]['difference'] = score_table[model][name[0]] - score_table[model][name[1]]
        score_table[model]['sum'] = score_table[model][name[0]] + score_table[model][name[1]]
        score_table[model]['per1'] = score_table[model][name[0]] / score_table[model]['sum']
        score_table[model]['per2'] = score_table[model][name[1]] / score_table[model]['sum']
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
        # print(record)
        if 'wav1' not in record.keys():
            continue
        base_score = record['scores']
        for score in record['scores'][1:]:
            if base_score != score:
                break
        else:
            drop_same += 1
            continue
        if len(record['scores']) < 24:
            continue

        # record['wav1']
        # print(zip(record['wav1'], record['wav2'], record['scores'], record['ip']))
        for wav1, wav2, score, ip in zip(record['wav1'], record['wav2'], record['scores'], record['ip']):
        # for wav1, wav2, score, ip in zip(record['wav1'], record['wav2'], record['scores'], record['ip']):
            # if level < 3:
            #    continue

            wav1_label = extract_class(wav1)
            wav2_label = extract_class(wav2)

            if len(wav1_label) == 2 and len(wav2_label) == 2:
                label_score.append([wav1_label, wav2_label, score])
            else:
                print(wav1, wav2)
                print(wav1_label, wav2_label)

    # None vs Flow
    # Flow vs Diff

    print('\n\nCount: Flow and Diff.')
    table = count_none_table(label_score, ('bs', 'sfndiff'), lambdas=[])
    print('bs: {}  sfndiff: {}'.format(table['bs'], table['sfndiff']))
    print('Difference: {}  Sum: {}  Perc1: {:.3f}  Perc2: {:.3f}'.format(
        table['difference'], table['sum'], table['per1'], table['per2']))

    for style in styles:
        print('\nCount by style: {}'.format(style))
        table = count_none_table(label_score, name=('bs', 'sfndiff'), lambdas=[lambda x: x['style'] == style])
        print('bs: {}  sfndiff: {}'.format(table['bs'], table['sfndiff']))
        print('Difference: {}  Sum: {}  Perc1: {:.3f}  Perc2: {:.3f}'.format(
                table['difference'], table['sum'], table['per1'], table['per2']))


if __name__ == '__main__':
    global args
    args = parser.parse_args()

    count = 0
    while True:
        print('Count: {}'.format(count))
        os.system('mongoexport -d {} -c score -o {}'.format(args.input, args.input))
        parse_print()
        time.sleep(60)

        count += 1