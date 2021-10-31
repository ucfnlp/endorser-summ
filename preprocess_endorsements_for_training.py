# -*- coding: utf8 -*-

"""
Script to convert multi-document inputs to TensorFlow examples which can be sent to the PG-MMR model.
"""

import os
from absl import flags
from absl import app
from tqdm import tqdm
import util
import numpy as np
import glob
import rouge_functions
import nltk
np.random.seed(123)
from transformers import RobertaTokenizer
tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-large')

FLAGS = flags.FLAGS


def convert_singpairmix_to_fairseq_examples(dataset_name, processed_data_dir, split='all'):
    exp_name = FLAGS.endorse_method
    if FLAGS.consolidation_method == 'sequential':
        exp_name += '_' + 'sequential'
    log_dir = FLAGS.dataset_name + '/' + exp_name
    out_dir = os.path.join('endorse_data', dataset_name + '_' + exp_name)
    util.create_dirs(out_dir)
    util.create_dirs(out_dir + '-bin')
    if split == 'all':
        if dataset_name == 'duc_2004' or dataset_name == 'tac_2011':
            output_dataset_splits = ['test']
            input_dataset_splits = ['test']
        else:
            output_dataset_splits = ['test', 'val', 'train']
            input_dataset_splits = ['test', 'val', 'train']
    else:
        output_dataset_splits = [split]
        input_dataset_splits = [split]
    for output_dataset_split, input_dataset_split in zip(output_dataset_splits, input_dataset_splits):
        highlight_split = 'valid' if output_dataset_split == 'val' else output_dataset_split
        processed_data_path = os.path.join(processed_data_dir, dataset_name, input_dataset_split)
        articles_file = 'articles'
        summaries_file = 'summaries'
        if FLAGS.dataset_name != 'duc_2004' and FLAGS.dataset_name != 'tac_2011':
            articles_file += '_raw'
            summaries_file += '_raw'
        articles_file += '.tsv'
        summaries_file += '.tsv'
        articles_path = os.path.join(processed_data_path, articles_file)
        abstracts_path = os.path.join(processed_data_path, summaries_file)
        endorse_scores_file = log_dir + '/' + 'endorse_scores_' + input_dataset_split + '.txt'
        token_ids_file = log_dir + '/' + 'token_ids_' + input_dataset_split + '.txt'
        bart_summaries_file = dataset_name + '/' + 'orig_bart_summaries_test.txt'
        total = util.num_lines_in_file(articles_path)

        if not os.path.exists(endorse_scores_file) or util.file_contents(endorse_scores_file).strip() == '':
            print('Putting all individual scores files together into one file...')
            indiv_endorse_scores_dir = log_dir + '/' + 'endorse_scores_' + input_dataset_split + '_all'
            endorse_files = glob.glob(os.path.join(indiv_endorse_scores_dir, '*'))
            if len(endorse_files) != total:
                raise Exception('len(endorse_files) != total')
            writer_endorse = open(endorse_scores_file, 'w')
            for i in range(total):
                endorse_file = indiv_endorse_scores_dir + '/ex_%05d.txt' % i
                with open(endorse_file) as f:
                    line = f.read().strip()
                    writer_endorse.write(line + '\n')
            writer_endorse.close()

            indiv_token_ids_dir = log_dir + '/' + 'token_ids_' + input_dataset_split + '_all'
            token_ids_files = glob.glob(os.path.join(indiv_token_ids_dir, '*'))
            if len(token_ids_files) != total:
                raise Exception('len(token_ids) != total')
            writer_token_ids = open(token_ids_file, 'w')
            for i in range(total):
                token_ids_file = indiv_token_ids_dir + '/tokenids_%05d.txt' % i
                with open(token_ids_file) as f:
                    line = f.read().strip()
                    writer_token_ids.write(line + '\n')
            writer_token_ids.close()

        with open(token_ids_file) as f:
            token_ids_text = f.read().strip()
        if not os.path.exists(token_ids_file) or token_ids_text == '':
            stringmatch_token_ids_file = FLAGS.dataset_name + '/' + exp_name + '/' + 'token_ids_' + input_dataset_split + '.txt'
            with open(stringmatch_token_ids_file) as f:
                text = f.read()
            with open(token_ids_file, 'w') as f:
                f.write(text)

        f_art = open(articles_path)
        f_abs = open(abstracts_path)
        f_endorse_scores = open(endorse_scores_file)
        f_token_ids = open(token_ids_file)
        writer_source = open(os.path.join(out_dir, output_dataset_split + '.source'), 'w')
        writer_target = open(os.path.join(out_dir, output_dataset_split + '.target'), 'w')
        writer_endorse = open(os.path.join(out_dir + '-bin', highlight_split + '.endorse'), 'w')
        writer_sentsalience = open(os.path.join(out_dir + '-bin', highlight_split + '.sentsalience'), 'w')
        writer_sourceids = open(os.path.join(out_dir + '-bin', highlight_split + '.src'), 'w')
        writer_targetids = open(os.path.join(out_dir + '-bin', highlight_split + '.tgt'), 'w')
        writer_rougescores = open(os.path.join(out_dir + '-bin', highlight_split + '.rougescores'), 'w')

        f_bart_summaries = open(bart_summaries_file)
        examples_bart_summaries = [summaries.strip().split('\n\n') for summaries in f_bart_summaries.read().strip().split('\n\n\n\n')]
        for example_idx in tqdm(range(total)):
            docs = f_art.readline().strip().split('\t\t')
            if FLAGS.dataset_name != 'duc_2004' and FLAGS.dataset_name != 'tac_2011':
                groundtruth_summ_sents = f_abs.readline().strip().split('\t')
                groundtruth_summaries_texts = [' '.join(groundtruth_summ_sents)]
                summ_tokenids = tokenizer.encode(' '.join(groundtruth_summ_sents))
            else:
                groundtruth_summaries = f_abs.readline().strip().split('\t\t')
                groundtruth_summaries_texts = [summ.strip().replace('\t', ' ') for summ in groundtruth_summaries]
                groundtruth_summ_sents = groundtruth_summaries[0].strip().split('\t')
                summ_tokenids = tokenizer.encode(' '.join(groundtruth_summ_sents))

            doc_scores_list = f_endorse_scores.readline().strip().split('\t')
            raw_sent_token_ids_list = f_token_ids.readline().strip().split('\t\t')

            def process_example(docs, groundtruth_summaries_texts, groundtruth_summ_sents, summ_tokenids, doc_scores_list, raw_sent_token_ids_list):
                # Remove invalid docs that are empty string
                valid_doc_indices = [doc_idx for doc_idx, doc in enumerate(docs) if doc.strip() != '']
                docs = [doc for doc_idx, doc in enumerate(docs) if doc_idx in valid_doc_indices]

                raw_article_sents_list = [art.strip().split('\t') for art in docs]
                raw_article_sents_list = [util.fix_wcep_problems(raw_article_sents) for raw_article_sents in raw_article_sents_list]

                if len(doc_scores_list) != len(docs):
                    print(len(doc_scores_list))
                    print(len(docs))
                    raise Exception('len(doc_scores_list) != len(docs)')
                source_endorse_ids_list = [list(map(int, doc_scores.strip().split(' '))) for doc_scores in doc_scores_list]
                source_endorse_ids_flat = util.flatten_list_of_lists(source_endorse_ids_list)
                sent_token_ids_list_maybe_too_long = [[list(map(int, doc_scores.strip().split(' '))) for doc_scores in raw_token_ids_list.strip().split('\t')] for raw_token_ids_list in raw_sent_token_ids_list]

                # TODO: shouldn't need this, because they should match
                sent_token_ids_list = []
                for doc_idx, sent_token_ids_maybe_too_long in enumerate(sent_token_ids_list_maybe_too_long):
                    sent_token_ids = []
                    len_ids = 0
                    for tokens in sent_token_ids_maybe_too_long:
                        tokens_to_add = []
                        for token in tokens:
                            # if len_ids < 1022:
                            if len_ids < len(source_endorse_ids_list[doc_idx]):        # TODO: shouldn't need this, because they should match. It should be if len_ids < 1022
                                tokens_to_add.append(token)
                                len_ids += 1
                        if len(tokens_to_add) > 0:
                            sent_token_ids.append(tokens_to_add)
                    sent_token_ids_list.append((sent_token_ids))

                sent_token_ids = util.flatten_list_of_lists(sent_token_ids_list)
                token_ids_flat = util.flatten_list_of_lists(sent_token_ids)
                if len(source_endorse_ids_flat) > len(token_ids_flat):      # TODO: shouldn't need this, because they should match
                    source_endorse_ids_flat = source_endorse_ids_flat[:len(token_ids_flat)]
                util.assert_lengths_equal(source_endorse_ids_flat, token_ids_flat)
                source_sent_endorse_ids = util.reshape_like(source_endorse_ids_flat, sent_token_ids)

                outside_range = [my_id for my_id in source_endorse_ids_flat if my_id < 0 or my_id > 10]
                if len(outside_range) > 0:
                    print(source_endorse_ids_flat)
                    raise Exception('len(outside_range) > 0')

                out_tokens_ids = []
                out_ids = []
                out_endorse_ids = []
                out_sent_salience_ids = []
                if FLAGS.sent_sel_method == 'first':
                    cur_sent_idx = 0
                    max_sent_idx = max([len(article_sent_tokenids) for article_sent_tokenids in sent_token_ids_list])
                    should_break = False
                    while len(out_endorse_ids) < FLAGS.maxlen - 1 and cur_sent_idx < max_sent_idx:
                        for art_idx, article_sent_tokenids in enumerate(sent_token_ids_list):
                            if cur_sent_idx >= len(article_sent_tokenids):
                                continue
                            tokens = article_sent_tokenids[cur_sent_idx]
                            if len(out_endorse_ids) + len(tokens) >= FLAGS.maxlen - 1:
                                should_break = True
                                break
                            out_tokens_ids.append(tokens)
                            out_ids.extend(tokens)
                            if len(tokens) <= 2:
                                my_endorse_ids = [(cur_sent_idx % 10) + 1] * len(tokens)    # we add 1 because 0 is the pad_idx
                            else:
                                my_endorse_ids = [(cur_sent_idx % 10) + 1] * (len(tokens) - 2)
                                my_endorse_ids = [1] + my_endorse_ids + [1]
                            if len(my_endorse_ids) != len(tokens):
                                print(my_endorse_ids)
                                print(len(my_endorse_ids))
                                print(tokens)
                                print(len(tokens))
                                raise Exception('len(my_endorse_ids) != len(tokens)')
                            out_endorse_ids.extend(my_endorse_ids)
                            out_sent_salience_ids.extend([(cur_sent_idx % 10) + 1] * len(tokens))
                        if should_break:
                            break
                        cur_sent_idx += 1
                elif FLAGS.sent_sel_method == 'max_salience':

                    sent_endorse_scores = [sum(scores) for scores in source_sent_endorse_ids]
                    sorted_indices = np.argsort(sent_endorse_scores)[::-1]
                    selected_indices = []
                    for cur_sent_idx in sorted_indices:
                        tokens = sent_token_ids[cur_sent_idx]
                        if len(tokens) >= FLAGS.maxlen - 2:
                            cur_sent_idx += 1
                            continue
                        if len(util.flatten_list_of_lists(out_endorse_ids)) + len(tokens) >= FLAGS.maxlen - 1:
                            break
                        out_tokens_ids.append(tokens)
                        out_endorse_ids.append(source_sent_endorse_ids[cur_sent_idx])
                        out_sent_salience_ids.append([sent_endorse_scores[cur_sent_idx]] * len(tokens))
                        selected_indices.append(cur_sent_idx)
                        cur_sent_idx += 1

                    # Reorder selected sentences so that they are back in chronological order
                    new_order = np.argsort(selected_indices)
                    out_tokens_ids = util.reorder(out_tokens_ids, new_order)
                    out_endorse_ids = util.reorder(out_endorse_ids, new_order)
                    out_sent_salience_ids = util.reorder(out_sent_salience_ids, new_order)

                    # Convert to flat lists
                    out_ids = util.flatten_list_of_lists(out_tokens_ids)
                    out_endorse_ids = util.flatten_list_of_lists(out_endorse_ids)
                    out_sent_salience_ids = util.flatten_list_of_lists(out_sent_salience_ids)

                source_text = ''.join([util.huggingface_convert_ids_to_text(ids, tokenizer) for ids in out_tokens_ids])[1:]
                source_endorse_ids_text = ' '.join(list(map(str, out_endorse_ids)))
                source_sent_salience_ids_text = ' '.join(list(map(str, out_sent_salience_ids)))
                source_ids_text = ' '.join(list(map(str, out_ids)))
                target_text = ' '.join(groundtruth_summ_sents)
                target_ids_text = ' '.join(list(map(str, summ_tokenids)))

                if FLAGS.coeff:
                    endorse_thresholds = [0, 1, 2]
                    rouge_scores = []
                    for endorse_threshold in endorse_thresholds:
                        source = [token_id for token_idx, token_id in enumerate(out_ids) if out_endorse_ids[token_idx] >= endorse_threshold]
                        threshold_source_text = ''.join([util.huggingface_convert_ids_to_text(ids, tokenizer) for ids in source])[1:]
                        threshold_source_text = ' '.join(nltk.tokenize.word_tokenize(threshold_source_text))
                        if FLAGS.dataset_name != 'duc_2004' and FLAGS.dataset_name != 'tac_2011':
                            threshold_target_texts = [target_text]
                        else:
                            threshold_target_texts = groundtruth_summaries_texts
                        threshold_target_texts = [' '.join(nltk.tokenize.word_tokenize(t)) for t in threshold_target_texts]
                        rouge_score = rouge_functions.multi_rouge_1(threshold_source_text, threshold_target_texts, metric='r')
                        rouge_scores.append(rouge_score)
                    rouge_scores_text = ' '.join(['%.2f' % rouge_score for rouge_score in rouge_scores])
                    writer_rougescores.write(rouge_scores_text + '\n')


                util.assert_lengths_equal(out_endorse_ids, out_ids)

                if len(out_ids) > 1024:
                    raise Exception('len(source_ids) > 1024: ' + str(len(out_ids)))

                writer_source.write(source_text + '\n')
                writer_endorse.write(source_endorse_ids_text + '\n')
                writer_sentsalience.write(source_sent_salience_ids_text + '\n')
                writer_sourceids.write(source_ids_text + '\n')
                writer_target.write(target_text + '\n')
                writer_targetids.write(target_ids_text + '\n')

            process_example(docs, groundtruth_summaries_texts, groundtruth_summ_sents, summ_tokenids, doc_scores_list, raw_sent_token_ids_list)


        writer_source.close()
        writer_target.close()
        writer_endorse.close()
        writer_sentsalience.close()
        writer_sourceids.close()
        writer_targetids.close()


def main(unused_argv):
    if len(unused_argv) != 1:  # prints a message if you've entered flags incorrectly
        raise Exception("Problem with flags: %s" % unused_argv)
    if FLAGS.dataset_name == '':
        raise Exception('Must specify which dataset to convert.')
    convert_singpairmix_to_fairseq_examples(FLAGS.dataset_name, FLAGS.line_by_line_data_path,
                                       split=FLAGS.dataset_split)


if __name__ == '__main__':
    flags.DEFINE_string('dataset_name', 'wcep_mds', 'Which dataset to use. Can be {wcep_mds, duc_2004, tac_2011}')
    flags.DEFINE_string('dataset_split', 'all', 'Which dataset split to use. Must be one of {train, val, test, all}')
    flags.DEFINE_string('line_by_line_data_path', 'data/processed',
                        'Where the data is, to be converted to bart input.')
    flags.DEFINE_string('sent_sel_method', 'max_salience', 'How to select which sentences will be added to the megadoc. Can be {max_salience, first}')
    flags.DEFINE_string('endorse_method', 'bertscore', 'Can be {pyramid, bertscore, stringmatch}')
    flags.DEFINE_string('consolidation_method', 'reciprocal', 'Can be {reciprocal, sequential}')
    flags.DEFINE_integer('maxlen', 1024, '')
    flags.DEFINE_boolean('coeff', True, '')
    flags.DEFINE_boolean('leave_one_out', False, 'Which dataset split to use. Must be one of {train, val, test, all}')
    app.run(main)
























