#!/usr/bin/env python
# encoding: utf-8

from __future__ import division
import json
import argparse
import time
import os

parser = argparse.ArgumentParser(description='parse the BSON file.')
parser.add_argument('--input', dest='input', type=str)

datasets = ['davis', 'sintel']
styles = ['composition', 'feathers', 'starrynight', 'lamuse']
works = ['flow', 'diff', 'none']
models = ['rnn', 'adain', 'sfn']


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
                score_table[model][0] += 1 - score if wl1['work'] == name else score
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

        for model in models:
            if wl1['model'] == model:
                score_table[model][0] += score if wl1['work'] == 'diff' else 1 - score
                score_table[model][1] += 1
                break
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
        # record['wav1']
        # print(zip(record['wav1'], record['wav2'], record['scores'], record['ip']))
        for wav1, wav2, score, ip in zip(record['wav1'], record['wav2'], record['scores'], record['ip']):
        # for wav1, wav2, score, ip in zip(record['wav1'], record['wav2'], record['scores'], record['ip']):
            # if level < 3:
            #    continue

            wav1_label = extract_class(wav1)
            wav2_label = extract_class(wav2)

            if len(wav1_label) == 4 and len(wav2_label) == 4:
                label_score.append([wav1_label, wav2_label, score])
            else:
                print(wav1, wav2)
                print(wav1_label, wav2_label)

    none_score = filter(lambda x: x[0]['work'] == 'none' or x[1]['work'] == 'none', label_score)
    normal_score = filter(lambda x: x[0]['work'] != 'none' and x[1]['work'] != 'none', label_score)
    none_score = list(none_score)
    normal_score = list(normal_score)

    print('Control count.')
    print('None < Others')
    print('Compare with {}.'.format('diff'))
    table = count_none_table(none_score, 'diff')

    for key in table.keys():
        print(key, 'True: {} All: {} Percent: {:.3f}'.format(table[key][1] - table[key][0], table[key][1], 1 - table[key][0] / (1e-6 + table[key][1])))

    print('Compare with {}.'.format('flow'))
    # none_score = filter(lambda x: x[0]['work'] == 'none' or x[1]['work'] == 'none', label_score)
    table = count_none_table(none_score, 'flow')
    print(table)

    for key in table.keys():
        print(key, 'True: {} All: {} Percent: {:.3f}'.format(table[key][1] - table[key][0], table[key][1], 1 - table[key][0] / (1e-6 + table[key][1])))

    print('\nNormal count.')
    print('Diff < Flow')
    table = count_table(normal_score, [])
    print(table)

    for key in table.keys():
        print(key, 'True: {} All: {} Percent: {:.3f}'.format(table[key][1] - table[key][0], table[key][1], 1 - table[key][0] / (1e-6 + table[key][1])))

    for style in styles:
        print('\nCount by style: {}'.format(style))
        table = count_table(normal_score, lambdas=[lambda x: x['style'] == style])

        for key in table.keys():
            print(key, 'True: {} All: {} Percent: {:.3f}'.format(table[key][1] - table[key][0], table[key][1], 1 - table[key][0] / (1e-6 + table[key][1])))

    for dataset in datasets:
        print('\nCount by dataset: {}'.format(dataset))
        table = count_table(normal_score, lambdas=[lambda x: x['dataset'] == dataset])

        for key in table.keys():
            print(key, 'True: {} All: {} Percent: {:.3f}'.format(table[key][1] - table[key][0], table[key][1], 1 - table[key][0] / (1e-6 + table[key][1])))

    for dataset in datasets:
        for style in styles:
            print('\nCount by dataset and style: {}, {}'.format(dataset, style))
            table = count_table(normal_score, lambdas=[lambda x: x['dataset'] == dataset and x['style'] == style])

            for key in table.keys():
                print(key, 'True: {} All: {} Percent: {:.3f}'.format(table[key][1] - table[key][0], table[key][1], 1 - table[key][0] / (1e-6 + table[key][1])))


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