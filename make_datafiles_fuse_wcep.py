# -*- coding: utf8 -*-

"""
Script to convert multi-document inputs to TensorFlow examples which can be sent to the PG-MMR model.
"""

import glob
import shutil
import nltk
import os
from bs4 import BeautifulSoup
import io
from absl import flags
from absl import app
import json
from tqdm import tqdm
import util
FLAGS = flags.FLAGS

def read_jsonl(path):
    with open(path) as f:
        for line in f:
            yield json.loads(line)

def move_from_quote_end_to_sent_end(tokens, cur_end_idx):
    spots_left_to_check = 5
    new_end_idx = cur_end_idx
    for i in range(1, spots_left_to_check+1):
        if cur_end_idx+i >= len(tokens):
            break
        if tokens[cur_end_idx+i] == '.':
            new_end_idx = cur_end_idx+i
            break
    return new_end_idx

def sent_tokenize_paragraph(tokens):
  sents = []
  while len(tokens) > 0:
      idx = next((i for i in range(len(tokens)) if (tokens[i] == '.' or tokens[i] == '?' or tokens[i] == '!')), len(tokens)-1)
      if tokens[idx] == '?':
          is_part_of_quote = False
          if idx+1 < len(tokens) and tokens[idx+1] == "'":
              idx = idx + 1
              is_part_of_quote = True
          if idx+1 < len(tokens) and tokens[idx+1] == "''":
              idx = idx + 1
              is_part_of_quote = True
          if is_part_of_quote:
              idx = move_from_quote_end_to_sent_end(tokens, idx)
      else:
          if idx+1 < len(tokens) and tokens[idx+1] == "'":
              idx = idx + 1
          if idx+1 < len(tokens) and tokens[idx+1] == "''":
              idx = idx + 1
      sent = tokens[:idx+1]
      if len(sent) > 0:
          sents.append(' '.join(sent))
      tokens = tokens[idx+1:]
  return sents

def sent_tokenize(text):
    sents = nltk.tokenize.sent_tokenize(text)
    return sents

def tokenize(text):
    text = text.strip().replace('\n\n', '\n').replace('\n\n', '\n').replace('\n\n', '\n').replace('\n\n', '\n').replace('\n\n', '\n')
    paragraphs = text.split('\n')
    raw_article_sents = []
    article_sents_tokenized = []
    for p in paragraphs:
        # tokens = list(parser.tokenize(p))

        # # doc = nlp(p)
        # # tokens = list(doc.tokens)
        # # # sents = nltk.tokenize.sent_tokenize(p.strip())
        # tokens = nltk.tokenize.word_tokenize(p.strip())
        # # # tokens = ["'" if token == "`" else token for token in tokens]
        # # # print(tokens)

        raw_sents = sent_tokenize(p.strip())
        tokenized_sents = [' '.join(nltk.tokenize.word_tokenize(sent)) for sent in raw_sents]
        raw_article_sents.extend(raw_sents)
        article_sents_tokenized.extend(tokenized_sents)
    return raw_article_sents, article_sents_tokenized

def process_dataset(dataset_name, out_data_path, in_data_path):
    if FLAGS.dataset_split == 'all':
        if FLAGS.dataset_name == 'duc_2004':
            dataset_splits = ['test']
        else:
            dataset_splits = ['test', 'val', 'train']
    else:
        dataset_splits = [FLAGS.dataset_split]
    for dataset_split in dataset_splits:
        out_dir = os.path.join(out_data_path, dataset_name, dataset_split)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        article_dir = os.path.join(in_data_path, dataset_split + '.jsonl')
        out_idx = 1
        num_sents_ne_1 = 0
        with open(os.path.join(out_dir, 'articles.tsv'), 'wb') as f_art, open(os.path.join(out_dir, 'summaries.tsv'), 'wb') as f_sum, \
            open(os.path.join(out_dir, 'articles_raw.tsv'), 'wb') as f_art_raw, open(os.path.join(out_dir, 'summaries_raw.tsv'), 'wb') as f_sum_raw:
            for ex_idx, ex in enumerate(tqdm(read_jsonl(article_dir), total=10200)):
                summary = ex['summary'].strip() # human-written summary
                articles_ = ex['articles'] # cluster of articles
                if FLAGS.dataset_name == 'wcep_mds':
                    articles = [art['text'] for art in articles_[:10]]
                else:
                    articles = [articles_[0]['text']]

                summary = summary.replace('’', "'").replace('”', '"').replace('‘', "'").replace('“', '"').replace('—', '-').replace('–', '-')
                summary_sents, summary_sents_tokenized = tokenize(summary)

                raw_article_sents_list = []
                article_sents_tokenized_list = []
                for raw_article in articles:
                    article = raw_article.replace('’', "'").replace('”', '"').replace('‘', "'").replace('“', '"').replace('—', '-').replace('–', '-')
                    raw_article_sents, article_sents_tokenized = tokenize(article)
                    raw_article_sents_list.append(raw_article_sents)
                    article_sents_tokenized_list.append(article_sents_tokenized)
                if len(summary_sents) != 1:
                    num_sents_ne_1 += 1
                    a=0
                # raw_article_sents_list.append(fixed_sentences)
                # out_str = '\t\t'.join(['\t'.join(sents) for sents in raw_article_sents_list])
                # f_art.write((out_str + '\n').encode('utf-8'))
                # out_str = '\t\t'.join(['\t'.join(sents) for sents in abstract_sents_list])
                # f_sum.write((out_str + '\n').encode('utf-8'))
                article_line = '\t\t'.join(['\t'.join(article_sents_tokenized) for article_sents_tokenized in article_sents_tokenized_list]) + '\n'
                abstract_line = '\t'.join(summary_sents_tokenized) + '\n'
                article_raw_line = '\t\t'.join(['\t'.join(raw_article_sents) for raw_article_sents in raw_article_sents_list]) + '\n'
                abstract_raw_line = '\t'.join(summary_sents) + '\n'
                f_art.write(article_line.encode())
                f_sum.write(abstract_line.encode())
                f_art_raw.write(article_raw_line.encode())
                f_sum_raw.write(abstract_raw_line.encode())
                out_idx += 1
        print(num_sents_ne_1)

def main(unused_argv):
    if len(unused_argv) != 1:  # prints a message if you've entered flags incorrectly
        raise Exception("Problem with flags: %s" % unused_argv)
    if FLAGS.dataset_name == '':
        raise Exception('Must specify which dataset to convert.')
    process_dataset(FLAGS.dataset_name, FLAGS.out_data_path, FLAGS.in_data_path)


if __name__ == '__main__':
    flags.DEFINE_string('dataset_name', 'wcep_mds',
                        'Which dataset to convert from raw data to tf examples')
    flags.DEFINE_string('dataset_split', 'all',
                        'Which dataset to convert from raw data to tf examples')
    flags.DEFINE_string('out_data_path', 'data/processed', 'Where to put output tf examples')
    flags.DEFINE_string('in_data_path', os.path.expanduser('~') + '/data/wcep-mds/data', 'Path to raw DUC data.')
    flags.DEFINE_string('custom_dataset_path', 'example_custom_dataset/',
                        'Path to custom dataset. Format of custom dataset must be:\n'
                        + 'One file for each topic...\n'
                        + 'Distinct articles will be separated by one blank line (two carriage returns \\n)...\n'
                        + 'Each sentence of the article will be on its own line\n'
                        + 'After all articles, there will be one blank line, followed by \'<SUMMARIES>\' without the quotes...\n'
                        + 'Distinct summaries will be separated by one blank line...'
                        + 'Each sentence of the summary will be on its own line'
                        + 'See the directory example_custom_dataset for an example')
    app.run(main)

























