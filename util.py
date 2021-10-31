# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
# Modifications Copyright 2017 Abigail See
# Modifications made 2018 by Logan Lebanoff
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#		 http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""This file contains some utility functions"""

import os
import numpy as np
from absl import flags
import itertools
from nltk.corpus import stopwords
import inspect, re
import string
from spacy.tokens import Doc
from spacy.lang.en import English
import sys
import unicodedata

if sys.version_info >= (3, 0):
    python_version = 3
else:
    python_version = 2

nlp = English()
FLAGS = flags.FLAGS

stop_words = set(stopwords.words('english'))


def flatten_list_of_lists(list_of_lists):
    return list(itertools.chain.from_iterable(list_of_lists))

def is_list_type(obj):
    return isinstance(obj, (list, tuple, np.ndarray))

def get_first_item(lst):
    if not is_list_type(lst):
        return lst
    for item in lst:
        result = get_first_item(item)
        if result is not None:
            return result
    return None

def remove_period_ids(lst, vocab):
    first_item = get_first_item(lst)
    if first_item is None:
        return lst
    if vocab is not None and type(first_item) == int:
        period = vocab.word2id(data.PERIOD)
    else:
        period = '.'

    if is_list_type(lst[0]):
        return [[item for item in inner_list if item != period] for inner_list in lst]
    else:
        return [item for item in lst if item != period]


def is_stopword_punctuation(word):
    if word in stop_words or word in ('<s>', '</s>'):
        return True
    is_punctuation = [ch in string.punctuation for ch in word]
    if all(is_punctuation):
        return True
    return False

def remove_stopwords_punctuation(sent):
    try:
        new_sent = [token for token in sent if not is_stopword_punctuation(token)]
    except:
        a=0
    return new_sent


def show_callers_locals():
    """Print the local variables in the caller's frame."""
    callers_local_vars = list(inspect.currentframe().f_back.f_back.f_back.f_locals.items())
    return callers_local_vars

def varname(my_var):
    callers_locals = show_callers_locals()
    return [var_name for var_name, var_val in callers_locals if var_val is my_var]

def reorder(l, ordering):
    return [l[i] for i in ordering]

def create_dirs(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def reshape_like(to_reshape, thing_with_shape):
    res = []
    if len(to_reshape) != len(flatten_list_of_lists(thing_with_shape)):
        print('Len of to_reshape (' + str(len(to_reshape)) + ') does not equal len of thing_with_shape (' + str(len(flatten_list_of_lists(thing_with_shape))) + ')')
        raise Exception('error')
    idx = 0
    for lst in thing_with_shape:
        list_to_add = []
        for _ in lst:

            try:
                list_to_add.append(to_reshape[idx])
            except:
                a=0
                raise
            idx += 1
        res.append(list_to_add)
    return res


def lemmatize_sent_tokens(article_sent_tokens):
    article_sent_tokens_lemma = [[t.lemma_ for t in Doc(nlp.vocab, words=[token for token in sent])] for sent in article_sent_tokens]
    return article_sent_tokens_lemma


def num_lines_in_file(file_path):
    with open(file_path) as f:
        num_lines = sum(1 for line in f)
    return num_lines



from unidecode import unidecode
def remove_non_ascii(text):
    return unidecode(text)

def huggingface_tokenize(text, tokenizer):
    tokens = [bytearray([tokenizer.byte_decoder[ch] for ch in token]).decode("utf-8") for token in tokenizer.tokenize(text)]
    return tokens

def get_bytes(token, tokenizer):
    return [tokenizer.byte_decoder[ch] for ch in token]

def huggingface_convert_ids_to_text(ids, tokenizer):
    tokens = tokenizer.convert_ids_to_tokens(ids)
    prev_errored_bytes = []
    decoded_tokens = []
    for token_idx, token in enumerate(tokens):
        my_bytes = get_bytes(token, tokenizer)
        if prev_errored_bytes:
            my_bytes = prev_errored_bytes + my_bytes
        my_bytearray = bytearray(my_bytes)
        try:
            decoded_token = my_bytearray.decode('utf-8')
            decoded_tokens.append(decoded_token)
            prev_errored_bytes = []
        except:
            prev_errored_bytes = my_bytes
    # tokens = [bytearray(get_bytes(token, tokenizer)).decode("utf-8", errors='replace') for token in tokens]
    text = ''.join(decoded_tokens)
    return text

def gpt2_tokenize(sents, tokenizer):
    sent_tokens = [tokenizer.encode((' ' + sent) if sent[0] != ' ' else sent) for sent in sents]
    return sent_tokens

def xlnet_preprocess_text(inputs):
    remove_space = True
    keep_accents = False
    do_lower_case = False
    if remove_space:
        outputs = " ".join(inputs.strip().split())
    else:
        outputs = inputs
    outputs = outputs.replace("``", '"').replace("''", '"')

    if not keep_accents:
        outputs = unicodedata.normalize("NFKD", outputs)
        outputs = "".join([c for c in outputs if not unicodedata.combining(c)])
    if do_lower_case:
        outputs = outputs.lower()

    return outputs

to_skip = ['ADVERTISEMENTSkip',
           '................................................................',
           '.......... .......... .......... .......... .......... .......... .......... .......... .......... .......... .......... .......... .......... .......... .......... .......... .......... ..........',
           ]
def fix_wcep_problems(raw_article_sents):
    new_raw_article_sents = []
    for sent in raw_article_sents:
        if sent in to_skip:
            continue
        sent = xlnet_preprocess_text(sent)
        sent = sent.replace('\u200e', '')
        sent = remove_non_ascii(sent)
        sent = sent.replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ')
        sent = sent.replace("''", '"')
        if sent == '':
            continue
        if sent[0] == ' ':
            sent = sent[1:]
        if sent == '':
            continue
        if sent[-1] == ' ':
            sent = sent[:-1]
        new_raw_article_sents.append(sent)
    return new_raw_article_sents

def assert_lengths_equal(v1, v2):
    if len(v1) != len(v2):
        out_str = ''
        out_str += str(varname(v1)) + ' (' + str(len(v1)) + '), '
        out_str += str(varname(v2)) + ' (' + str(len(v2)) + '), '
        print('Lengths are unequal: ' + out_str)
        # import pdb; pdb.set_trace()
        raise Exception('Lengths are unequal: ' + out_str)

def file_contents(file_name):
    with open(file_name) as f:
        text = f.read()
    return text






