#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
@project: PyCharm
@file: compute_sore.py
@author: Shengqiang Zhang
@time: 2020/10/30 13:31
@mail: sqzhang77@gmail.com
"""

import os
from nmt_bleu import compute_bleu
from nmt_rouge import rouge
import json
import subprocess


FILE_NAME = "conala2doc_3_beam.json"


references_blue_list = []
candidates_bleu_list = []

references_rouge_list = []
candidates_rouge_list = []


PREDICTIONS_FILE_NAME = FILE_NAME.split(".")[0] + "_predictions.txt"
REFERENCES_FILE_NAME  = FILE_NAME.split(".")[0] + "_references.txt"

if __name__ == '__main__':



    with open("tmp/" + FILE_NAME, 'r') as f:
        all_data = f.readlines()
    all_data = [line.strip() for line in all_data]

    for line in all_data:
        line_json = json.loads(line)

        predictions = ' '.join(line_json['predictions'])
        references = ' '.join(line_json['references'])

        candidates_bleu_list.append(predictions.split(' '))
        references_blue_list.append([references.split(' ')])

        candidates_rouge_list.append(predictions)
        references_rouge_list.append(references)

    assert len(references_blue_list) == len(candidates_bleu_list), '1. must be euqal.'
    assert len(references_rouge_list) == len(candidates_rouge_list), '2. must be euqal.'


    with open("tmp/" + PREDICTIONS_FILE_NAME, mode='w', encoding='utf-8') as f:
        for line in candidates_rouge_list:
            f.write(line)
            f.write('\n')

    with open("tmp/" + REFERENCES_FILE_NAME, mode='w', encoding='utf-8') as f:
        for line in references_rouge_list:
            f.write(line)
            f.write('\n')


    # meteor
    meteor_cmd = ['java', '-Xmx2G', '-jar', 'c2nl/eval/meteor/meteor-1.5.jar', "tmp/" + PREDICTIONS_FILE_NAME, "tmp/" + REFERENCES_FILE_NAME, '-l', 'en', '-norm']
    os.system(' '.join(meteor_cmd))
    print('↑↑↑↑↑↑↑meteor↑↑↑↑↑↑↑\n')

    # nmt bleu
    print('nmt corpus bleu4: {}'.format(compute_bleu(references_blue_list, candidates_bleu_list, max_order=4)[0]))
    print('\n')

    # nmt rouge
    rouge_score = rouge(candidates_rouge_list, references_rouge_list)
    print('nmt rouge_1/f_score: {}'.format(rouge_score['rouge_1/f_score']))
    print('nmt rouge_1/r_score: {}'.format(rouge_score['rouge_1/r_score']))
    print('nmt rouge_1/p_score: {}'.format(rouge_score['rouge_1/p_score']))
    print('\n')
    print('nmt rouge_2/f_score: {}'.format(rouge_score['rouge_2/f_score']))
    print('nmt rouge_2/r_score: {}'.format(rouge_score['rouge_2/r_score']))
    print('nmt rouge_2/p_score: {}'.format(rouge_score['rouge_2/p_score']))
    print('\n')
    print('nmt rouge_l/f_score: {}'.format(rouge_score['rouge_l/f_score']))
    print('nmt rouge_l/r_score: {}'.format(rouge_score['rouge_l/r_score']))
    print('nmt rouge_l/p_score: {}'.format(rouge_score['rouge_l/p_score']))