# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tqdm import tqdm
import numpy as np
from collections import Counter
from nltk.tokenize import TweetTokenizer
from nltk.tokenize import word_tokenize
import json
import os
import gzip
# from utilities import *
# from utils import *
from nltk.stem.porter import PorterStemmer
import argparse
# from nus_utilities import *
# from common_v2 import *

import sys






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







from tqdm import tqdm
import numpy as np
from collections import Counter
from nltk.tokenize import TweetTokenizer
from nltk.tokenize import word_tokenize
import json
import gzip
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from collections import defaultdict
import sys

# sys.dont_write_bytecode = True
''' Utilities for prep scripts
'''

tweet_tokenizer = TweetTokenizer()
porter_stemmer = PorterStemmer()

def find_sub_list(sl,l):
    results=[]
    sll=len(sl)
    for ind in (i for i,e in enumerate(l) if e==sl[0]):
        if l[ind:ind+sll]==sl:
            results.append((ind,ind+sll-1))
    return results

import gzip
import io
import json

def fast_load(path):
	'''
	Read js file:
	key ->  unicode keys
	string values -> unicode value
	'''
	print("Fast Loading {}".format(path))
	try:
		gz = gzip.open(path, 'rb')
		f = io.BufferedReader(gz)
		# with gzip.open(path, 'r') as f:
		return json.loads(f.read())
	except:
		print("Can't find Gzip. loading pure json")
		with open(path, 'r') as f:
			return json.loads(f.read())

def get_ngrams(n, text):
  """Calcualtes n-grams.

  Args:
    n: which n-grams to calculate
    text: An array of tokens

  Returns:
    A set of n-grams
  """
  ngram_set = set()
  text_length = len(text)
  max_index_ngram_start = text_length - n
  for i in range(max_index_ngram_start + 1):
    ngram_set.add(tuple(text[i:i + n]))
  return ngram_set

def tweet_processer(x):
    if('@' in x):
        return '@USER'
    elif('http' in x):
        return '@URL'
    else:
        return x

def tylib_tokenize(x, setting='split', lower=False,
                tweet_process=False):
    ''' All tokenizer in one. A convenient wrapper.

    Supported - 'split','nltk_tweet'

    TODO:'treebank','nltk_word'

    Args:
        x: `list`. list of words
        setting: `str` supports different tokenizers

    Returns:
        Tokenized output `list`

    '''
    if(setting=='split'):
        tokens = x.split(' ')
    elif(setting=='nltk_tweet'):
        tokens = tweet_tokenizer.tokenize(x)
    elif(setting=='nltk'):
        tokens = word_tokenize(x)
    if(lower):
        tokens = [x.lower() for x in tokens]
    if(tweet_process):
        tokens = [tweet_processer(x) for x in tokens]
    return tokens

def word_to_index(word, word_index, unk_token=1):
    ''' Maps word to index.

    Arg:
        word: `str`. Word to be converted
        word_index: `dict`. dictionary of word-index mapping
        unk_token: `int`. token to label if OOV

    Returns:
        idx: `int` Index of word converted
    '''
    try:
        idx = word_index[word]
    except:
        idx = 1
    return idx

porter_stemmer = PorterStemmer()
from nltk.corpus import wordnet as wn

stemmer = porter_stemmer

def compute_dfs(docs):
  word2df = defaultdict(float)
  for doc in docs:
    for w in set(doc):
      word2df[w] += 1.0
  num_docs = len(docs)
  for w, value in word2df.items():
    word2df[w] = np.math.log(num_docs / value)
  return word2df

def compute_overlap_features(questions, answers, word2df=None, stoplist=None):
  word2df = word2df if word2df else {}
  stoplist = stoplist if stoplist else set()
  feats_overlap = []
  for question, answer in zip(questions, answers):
    q_set = set([q for q in question if q not in stoplist])
    a_set = set([a for a in answer if a not in stoplist])
    word_overlap = q_set.intersection(a_set)
    overlap = float(len(word_overlap)) / (len(q_set) + len(a_set))
    word_overlap = q_set.intersection(a_set)
    df_overlap = 0.0
    for w in word_overlap:
      df_overlap += word2df[w]
    df_overlap /= (len(q_set) + len(a_set))
    feats_overlap.append(np.array([
                         overlap,
                         df_overlap,
                         ]))
  return np.array(feats_overlap)

def all_overlap_features(q, a, word2df):
    stoplist = set(stopwords.words('english'))
    w1, df1 = overlap_feats(q, a, word2df, stoplist=None)
    w2, df2 = overlap_feats(q, a, word2df, stoplist=stoplist)
    return [w1, df1, w2, df2]

def overlap_feats(question, answer, word2df, stoplist=None):
    stoplist = stoplist if stoplist else set()
    question = question.split(' ')
    answer = answer.split(' ')
    # question = [x.lower() for x in question]
    # answer = [x.lower() for x in answer]
    q_set = set([q for q in question if q not in stoplist])
    a_set = set([a for a in answer if a not in stoplist])
    # print(q_set)
    # print(a_set)
    word_overlap = q_set.intersection(a_set)
    overlap = float(len(word_overlap)) / (len(q_set) + len(a_set))
    word_overlap = q_set.intersection(a_set)
    df_overlap = 0.0
    for w in word_overlap:
      df_overlap += word2df[w]
    df_overlap /= (len(q_set) + len(a_set))
    return overlap, df_overlap

def build_em_feats(data, stem=False, lower=False):
    em_left, em_right = [],[]
    for x in tqdm(data):
        em1, em2 = exact_match_feats(x[0], x[1], stem=stem, lower=lower)
        em_left.append(em1)
        em_right.append(em2)
    return em_left, em_right

def exact_match_feats(q1, q2, stem=False, lower=False):
    """ builds exact match features

    Pass in tokens.
    """
    if(lower):
        q1 = [x.lower() for x in q1]
        q2 = [x.lower() for x in q2]
    if(stem):
        q1 = [porter_stemmer.stem(x) for x in q1]
        q2 = [porter_stemmer.stem(x) for x in q2]
    a_em = []
    b_em = []
    for a in q1:
        check_b = [x for x in q2 if a==x]
        if(len(check_b)>0):
            a_em.append(1)
        else:
            a_em.append(0)
    for b in q2:
        check_a = [x for x in q1 if b==x]
        if(len(check_a)>0):
            b_em.append(1)
        else:
            b_em.append(1)
    return a_em, b_em


def sequence_to_indices(seq, word_index, unk_token=1):
    ''' Converts sequence of text to indices.

    Args:
        seq: `list`. list of list of words
        word_index: `dict`. dictionary of word-index mapping

    Returns:
        seq_idx: `list`. list of list of indices

    '''
    # print(seq)
    seq_idx = [word_to_index(x, word_index, unk_token=unk_token) for x in seq]
    return seq_idx

def build_word_index(words, min_count=1, vocab_count=None,
                        extra_words=['<pad>','<unk>'],
                        lower=True):
    ''' Builds Word Index

    Takes in all words in corpus and returns a word_index

    Args:
        words: `list` a list of words in the corpus.
        min_count: `int` min number of freq to be included in index
        extra_words: `list` list of extra tokens such as pad or unk
            tokens

    Returns:
        word_index `dict` built word index
        index_word `dict` inverrted word index

    '''

    # Build word counter

    # lowercase
    if(lower):
        words = [x.lower() for x in words]

    word_counter = Counter(words)

    # Select words above min Count
    if(vocab_count is not None):
        words = [x[0] for x in word_counter.most_common()[:vocab_count]]
    else:
        words = [x[0] for x in word_counter.most_common() if x[1]>min_count]

    # Build Word Index with extra words
    word_index = {w:i+len(extra_words) for i, w in enumerate(words)}
    for i, w in enumerate(extra_words):
        word_index[w] = i

    # Builds inverse index
    index_word = {word:index for index, word in word_index.items()}

    print(index_word[0])
    print(index_word[1])
    print(index_word[2])

    return word_index, index_word

import io
import codecs

def load_vectors(fname):
    print(fname)
    # fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    # n, d = map(int, fin.readline().split())
    data = {}
    with open(fname, 'r') as f:
        lines = f.readlines()
        for line in tqdm(lines):
            # print(line)
            tokens = line.rstrip().split(' ')
            # print(tokens[0])
            data[tokens[0].decode('utf-8')] = np.array(tokens[1:])
    return data

def build_embeddings(word_index, index_word, num_extra_words=2,
                    emb_types=[('glove',300)],
                    base_dir='../', out_dir='./',
                    init_type='zero', init_val=0.01, normalize=False,
                    check_subtokens=False):
    ''' Builds compact glove embeddings for initializing

    Args:
        word_index: `dict` of words and indices
        index_word: `dict` inverted dictionary
        num_extra_words: `int` number of extra words (unk, pad) etc.
        emb_types:  `list` of tuples. ('glove,300'),('tweets',100)
            supports both tweets and glove (commoncrawl adaptations)
        base_dir: `str` file path of where to get the embeddings from
        out_dir: `str` file path to where to store the embeddings
        init_type: `str` normal, unif or zero (how to init unk)
        init_val: `float` this acts as std for normal distribution and
            min/max val for uniform distribution.

    Returns:
        Saves the embedding to directory

    '''

    # Setup default paths
    print('Loading {} types of embeddings'.format(len(emb_types)))

    tweet_path = '{}/twitter_glove/'.format(base_dir)
    glove_path = '{}/glove_embeddings/'.format(base_dir)
    fast_text = './embed/'

    for _emb_type in emb_types:
        emb_type, dimensions = _emb_type[0], _emb_type[1]
        print(emb_type)
        print(dimensions)
        glove = {}
        if(emb_type=='tweets'):
            # dimensions = 100
            emb_path = 'glove.twitter.27B.{}d.txt'.format(dimensions)
            emb_path = tweet_path + emb_path
        elif(emb_type=='fasttext_chinese'):
            emb_path = './embed/cc.zh.300.vec'
        elif(emb_type=='glove'):
            if(dimensions==300):
                # dimensions = 300
                emb_path = 'glove.840B.{}d.txt'.format(dimensions)
                emb_path = glove_path + emb_path
            else:
                emb_path = 'glove.6B.{}d.txt'.format(dimensions)
                emb_path = glove_path + emb_path

        print("Loading Glove Embeddings...")

        # Load word embeddings
        # Please place glove in correct place!
        if('fasttext' in emb_type):
            glove = load_vectors(emb_path)
        else:
            with open(emb_path, 'r',encoding="utf8") as f:
                lines = f.readlines()
                for l in tqdm(lines):
                    vec = l.split(' ')
                    word = vec[0]
                    vec = vec[1:]
                    # print(word)
                    glove[word] = np.array(vec)


        print('glove size={}'.format(len(glove)))

        print("Finished making glove dictionary")
        matrix = []
        oov_words = []
        for i in range(num_extra_words):
            matrix.append(np.zeros((dimensions)).tolist())


        oov = 0
        def _get_word(word):
            for each in (word, word.lower(), word.capitalize(), word.upper()):
                if each in glove:
                    return glove[each]
            return 1

        all_alpha = list(string.ascii_letters) + list(string.digits)
        all_alpha = {key:1 for key in all_alpha}
        def _get_word_simple(word):
            try:
                return glove[word]
            except:
                if(check_subtokens):
                    word = [x for x in word if x not in all_alpha]
                    # print(word)
                    vec = None
                    for w in word:
                        try:
                            v = glove[w]
                            if(vec is None):
                                vec = v
                            else:
                                vec = np.mean(vec, v)
                        except:
                            continue
                    if(vec is None):
                        return 1
                    else:
                        return vec
                return 1

        for i in tqdm(range(num_extra_words, len(word_index))):
            word = index_word[i]
            if(emb_type=='glove'):
                vec = _get_word(word)
            else:
                vec = _get_word_simple(word)
            if(vec==1):
                oov +=1
                oov_words.append(word)
                if(init_type=='unif'):
                    # uniform distribution
                    vec = np.random.uniform(low=-init_val,high=init_val,
                                size=(dimensions))
                elif(init_type=='normal'):
                    # normal distribution
                    vec = np.random.normal(0, init_val,
                                size=(dimensions))
                elif(init_type=='zero'):
                    # zero vectors
                    vec = np.zeros((dimensions))
                matrix.append(vec.tolist())
            else:
                # vec = glove[word]
                matrix.append(vec.tolist())



        matrix = np.stack(matrix)
        matrix = np.reshape(matrix,(len(word_index), dimensions))
        matrix = matrix.astype(np.float)

        print(matrix.shape)

        # if(normalize):
        #     norm = np.linalg.norm(matrix, axis=1).reshape((-1, 1))
        #     matrix = matrix / norm

        # print(oov_words)
        with open('{}/oov.txt'.format(out_dir), 'w',encoding="utf8") as f:
            for w in oov_words:
                f.write(w +'\n')
        print(matrix.shape)
        print(len(word_index))
        print("oov={}".format(oov))

        print("Finished building and writing...")

        # env['glove'] = matrix
        np.save('{}/emb_{}_{}.npy'.format(out_dir, emb_type,
                                        dimensions), matrix)
        print("Saved to file..")



def dictToFile(dict, path, use_zip=True):
    ''' Writes to gz format

    Args:
        dict `dict` file to save
        path `str` path to save it

    Returns:
        Nothing. Saves file to directory

    '''
    print("Writing to {}".format(path))
    if(use_zip==False):
        with open(path, 'w') as f:
            f.write(json.dumps(dict))
    else:
        with gzip.open(path, 'w') as f:
            f.write(json.dumps(dict))


def dictFromFileUnicode(path):
    ''' Reads File from Path

    Args:
        path: `str` path to load file from

    Returns:
        Loaded file (dict obj)
    '''
    print("Loading {}".format(path))
    with gzip.open(path, 'r') as f:
        return json.loads(f.read())





##########################################################################################################









sys.dont_write_bytecode = True

parser = argparse.ArgumentParser()
ps = parser.add_argument
ps("--mode", dest="mode", type=str,  default='all', help="mode")
ps("--vocab_count", dest="vocab_count", type=int,
    default=0, help="set >0 to activate")
args =  parser.parse_args()
mode = args.mode


def word_level_em_features(s1, s2, lower=True, stem=True):
    em1 = []
    em2 = []
    #print(s1)
    #print(s2)
    s1 = s1.split(' ')
    s2 = s2.split(' ')
    if(lower):
        s1 = [x.lower() for x in s1]
        s2 = [x.lower() for x in s2]
    if(stem):
        s1 = [PorterStemmer().stem(x) for x in s1]
        s2 = [PorterStemmer().stem(x) for x in s2]
    for w1 in s1:
        if(w1 in s2):
            em1.append(1)
        else:
            em1.append(0)
    for w2 in s2:
        if(w2 in s1):
            em2.append(1)
        else:
            em2.append(0)
    return em1, em2

def convert_paragraph(para):
    words = []
    context = para['context.tokens']

    qid = para['_id']
    question = para['question.tokens']
    words += question.split(' ')
    answers = para['answers']
    # print(answers)
    try:
        label_start = answers[1][0][0]
        label_length = len(answers[0].split(' '))
    except:
        label_start, label_length = -1, -1
    words += context.split(' ')
    ground_truths = para['ground_truths']
    data = [context, question, label_start, label_length, qid, ground_truths]
    # print(data)
    return data, words

def load_set(fp, datatype='train'):
    parsed_file = load_json(fp)
    # print(parsed_file)
    all_words = []
    all_data = []
    all_feats = []
    # print(parsed_file[0])
    for p in tqdm(parsed_file, desc='parsing file'):
        pdata, words = convert_paragraph(p)
        qem, pem =  word_level_em_features(pdata[1], pdata[0])
        all_words += words
        all_data.append(pdata)
        all_feats.append([pem, qem])
        # print(qem)
    # print(' Collected {} words'.format(len(all_words)))
    return all_words, all_data, all_feats


train_words, train_data, train_feats = load_set('train.json')
dev_words, dev_data, dev_feats = load_set('dev.json')
test_words, test_data, test_feats = load_set('test.json')

all_words = train_words + dev_words + test_words

if(args.vocab_count>0):
    print("Using Vocab Count of {}".format(args.vocab_count))
    word_index, index_word = build_word_index(all_words, min_count=0,
                                                vocab_count=args.vocab_count,
                                                lower=True)
else:
    word_index, index_word = build_word_index(all_words, min_count=0,
                                                lower=True)

print("Vocab Size={}".format(len(word_index)))

# Convert passages to tokens
# passages = dict(train_passage.items() + test_passage.items() + dev_passage.items())

fp = './datasets/NarrativeQA/'

if not os.path.exists(fp):
    os.makedirs(fp)

build_embeddings(word_index, index_word,
  out_dir=fp,
  init_type='zero', init_val=0.01,
  emb_types=[('glove',300)],
  normalize=False)

passages = {}

env = {
    'train':train_data,
    'test':test_data,
    'dev':dev_data,
    'passages':passages,
    'word_index':word_index
}

feature_env = {
    'train':train_feats,
    'test':test_feats,
    'dev':dev_feats
    }

dictToFile(env,'./datasets/NarrativeQA/env.gz'.format(mode))
dictToFile(feature_env,'./datasets/NarrativeQA/feats.gz'.format(mode))
