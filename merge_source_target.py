#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
@project: PyCharm
@file: merge_source_target.py
@author: Shengqiang Zhang
@time: 2020/12/24 16:58
@mail: sqzhang77@gmail.com
"""

import collections

if __name__ == '__main__':

    for type in ["atis", "conala", "wikisql"]:

        with open("dataset/" + type + "/train_source.txt", mode="r", encoding="utf-8") as f:
            train_source = f.readlines()
            train_source = [line.strip() for line in train_source]

        with open("dataset/" + type + "/train_target.txt", mode="r", encoding="utf-8") as f:
            train_target = f.readlines()
            train_target = [line.strip() for line in train_target]




        with open("dataset/" + type + "/eval_source.txt", mode="r", encoding="utf-8") as f:
            eval_source = f.readlines()
            eval_source = [line.strip() for line in eval_source]

        with open("dataset/" + type + "/eval_target.txt", mode="r", encoding="utf-8") as f:
            eval_target = f.readlines()
            eval_target = [line.strip() for line in eval_target]




        with open("dataset/" + type + "/test_source.txt", mode="r", encoding="utf-8") as f:
            test_source = f.readlines()
            test_source = [line.strip() for line in test_source]

        with open("dataset/" + type + "/test_target.txt", mode="r", encoding="utf-8") as f:
            test_target = f.readlines()
            test_target = [line.strip() for line in test_target]




        with open("dataset/" + type + "/" + "train.csv", mode="w", encoding="utf-8") as f:
            for index, line in enumerate(train_target):
                f.write("{}-+|.-+.|{}".format(train_target[index], train_source[index]))
                f.write("\n")

        with open("dataset/" + type + "/" + "eval.csv", mode="w", encoding="utf-8") as f:
            for index, line in enumerate(eval_target):
                f.write("{}-+|.-+.|{}".format(eval_target[index], eval_source[index]))
                f.write("\n")

        with open("dataset/" + type + "/" + "test.csv", mode="w", encoding="utf-8") as f:
            for index, line in enumerate(test_target):
                f.write("{}-+|.-+.|{}".format(test_target[index], test_source[index]))
                f.write("\n")




        # 将词典写入文件
        vocab_counter = collections.Counter()
        for line in train_source:
            for word in line.split(' '):
                vocab_counter.update([word])

        for line in train_target:
            for word in line.split(' '):
                vocab_counter.update([word])




        # 将词典写入文件
        print("writing {} vocab file...".format(type))
        with open("dataset/" + type + "/" + "vocab", mode='w', encoding='utf-8') as writer:
            writer.write("<pad>" + '\n')
            writer.write("<unk>" + '\n')
            writer.write("<s>" + '\n')
            writer.write("</s>" + '\n')
            for word, count in vocab_counter.most_common(50000000):
                writer.write(word + '\n')
        print("finished writing {} vocab file".format(type))

