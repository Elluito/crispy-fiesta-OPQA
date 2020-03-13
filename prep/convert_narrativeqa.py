# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from pprint import pprint
import json
import re
import numpy as np
import numpy.random as npr
import random
import  time
from os import path

from tqdm import tqdm
import numpy as np
from collections import Counter
from nltk.tokenize import TweetTokenizer
from nltk.tokenize import word_tokenize
import json
import os
import gzip
import csv
from utilities import *
# from utils import *
from nltk.stem.porter import PorterStemmer
import argparse
import operator
import random
from nltk.corpus import stopwords
# from nus_utilities import *
# from common_v2 import *
import sys
from rouge import Rouge


#################################################################### ESTO IVA EN UTILS PERO LO TRAJE AQUI###################################################################




def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)


def write_json(obj, path):
    with open(path, 'w') as f:
        return json.dump(obj, f)

def create_idict(dict):
    return {v: k for (k, v) in dict.items()}


def choice(objs, size, replace=True, p=None):
    all_inds = range(len(objs))
    inds = npr.choice(all_inds, size=size, replace=replace, p=p)
    return [objs[ind] for ind in inds]


def locate(context, span):
    for i in range(len(context) - len(span) + 1):
        if context[i:i+len(span)] == span:
            return i
    print(context)
    print(span)
    raise Exception('error, cannot match span in context')


def replace(l, ws, wt):
    new_l = []
    for w in l:
        if w == ws:
            new_l.append(wt)
        else:
            new_l.append(w)
    return new_l

def mkdir_if_not_exist(path):
    if path == '':
        return
    if not os.path.exists(path):
        os.makedirs(path)


class Checkpoint(object):
    def __init__(self, dirname):
        self.dirname = dirname
        mkdir_if_not_exist(dirname)


    def log(self, it, obj):
        write_json(obj, os.path.join(self.dirname, 'checkpoint_%d' % it))


color2num = dict(
    gray=30,
    red=31,
    green=32,
    yellow=33,
    blue=34,
    magenta=35,
    cyan=36,
    white=37,
    crimson=38
)


class Timer(object):
    def __init__(self, name=None, output=sys.stdout):
        self.name = name
        if output and type(output) == str:
            self.output = open(output, 'w')
        else:
            self.output = output

    def __enter__(self):
        if self.name:
            print >>self.output, colorize('[%s]\t' % self.name, 'green'),
        print >>self.output, colorize('Start', 'green')
        self.tstart = time.time()
        self.output.flush()

    def __exit__(self, type, value, traceback):
        if self.name:
            print >>self.output, colorize('[%s]\t' % self.name, 'green'),
        print >>self.output, colorize('Elapsed: %s' % (time.time() - self.tstart),
                                      'green')
        self.output.flush()









###############################################################################################################################################################################
import sys
import io
import pandas as pd
# sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding = 'utf-8')
# sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding = 'utf-8')
# sys.dont_write_bytecode = True

""" Script to convert NarrativeQA to Answer Span-prediction
Following the original paper, we use the best rouge-L matching score
for span selection.
"""

# Load passages / summaries
passages = {}
with open('../corpus/narrativeqa/third_party/wikipedia/summaries.csv','r',encoding="utf8") as f:
    reader = csv.reader(f, delimiter=',')
    # cosa = pd.read_csv(f)
    # reader.next()
    reader.__next__()
    for r in reader:
        passages[r[0]] = r[3]

print("Collected {} Passages".format(len(passages)))

def find_sub_list(sl,l):
    results=[]
    sll=len(sl)
    for ind in (i for i,e in enumerate(l) if e==sl[0]):
        if l[ind:ind+sll]==sl:
            results.append((ind,ind+sll-1))
    return results

max_span = 6

fp = '../corpus/narrativeqa/qaps.csv'
stoplist = set(['the','a','.',','])
train, dev, test = [],[],[]
ignored_train, ignored_eval = 0, 0

with open(fp, 'r',encoding="utf8") as f:
    reader = csv.reader(f, delimiter=',')
    reader.__next__()   # Skip Header
    for idx, r in tqdm(enumerate(reader)):
        # if(idx>500):
        #     break
        # data = {}
        _id = r[0]
        set_type = r[1]
        # print(set_type)
        question = r[5]
        answer1 = r[6]
        answer2 = r[7]
        passage =passages[_id]
        p = passage.split(' ')
        results = {}
        for i in range(0, max_span):
            ngrams = get_ngrams(i+1, p)
            ngrams = [' '.join(x) for x in ngrams]
            for n in ngrams:
                if n not in stoplist and n!="...":
                    r1 =Rouge().get_scores(n,answer1)[0]["rouge-2"]["f"]
                    r2 =Rouge().get_scores(n,answer2)[0]["rouge-2"]["f"]

                    if(r1>0 or r2>0) :
                        results[n] = r1 if r1>r2 else r2
        sorted_results = sorted(results.items(),
                        key=operator.itemgetter(1), reverse=True)

        new_results = []
        for s in sorted_results:
            if(s[0].lower() in stoplist):
                continue
            else:
                new_results.append(s[0])
        if(len(new_results)==0):
            if(set_type=='train'):
                ignored_train +=1
                continue
            else:
                # dummy ans for dev/test
                rand = random.randint(0, len(p)-2)
                choosen_ans = p[rand:rand+1]
                ignored_eval +=1
        else:
            choosen_ans = new_results[0].split(' ')
        spans = find_sub_list(choosen_ans, p)
        # train_labels = [new_results[0], spans]
        span_ans = p[spans[0][0]:spans[0][1]+1]
        ans_str = ' '.join(choosen_ans)
        assert(' '.join(span_ans)==ans_str)

        answers = [ans_str, spans]
        # print(train_labels)
        data = {
            '_id':_id,
            'question.tokens':question,
            'ground_truths':[answer1, answer2],
            'context.tokens':passage,
            'answers':answers
        }
        if(set_type=='train'):
            train.append(data)
        elif(set_type=='valid'):
            dev.append(data)
        elif(set_type=='test'):
            test.append(data)

print("Train={} Dev={} Test={}".format(
                            len(train), len(dev),
                            len(test)))
print("Ignored Train={} Ignored Eval={}".format(ignored_train,
                                            ignored_eval))


with open('../corpus/narrativeqa/train.json', 'w') as f:
    f.write(json.dumps(train,  indent=4,))

with open('../corpus/narrativeqa/dev.json', 'w') as f:
    f.write(json.dumps(dev,  indent=4,))

with open('../corpus/narrativeqa/test.json', 'w') as f:
    f.write(json.dumps(test,  indent=4,))
