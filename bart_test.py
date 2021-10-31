import torch
from tqdm import tqdm
import os
from absl import flags
from absl import app
import sys
import itertools
import pyrouge
import util

flags.DEFINE_string('hypo_file', 'test.hypo', 'Where abstractive summaries will be saved to')
flags.DEFINE_string('dataset_name', 'wcep_mds', 'Which dataset to use. Can be {cnn_dm, xsum, duc_2004}')
flags.DEFINE_string('endorse_method', 'bertscore', 'Can be {pyramid, bertscore, stringmatch}')
flags.DEFINE_string('consolidation_method', 'reciprocal', 'Can be {reciprocal, sequential}')
flags.DEFINE_boolean('endorse_duplicate_heads', False, 'Whether to use companion heads')
flags.DEFINE_float('endorse_original_head_scale', 0.8, 'Ratio used for Companion Heads')
flags.DEFINE_integer('min_len', 55, '')
flags.DEFINE_boolean('original_bart', False, 'Whether to use original BART without endorsement')

# Flags only used when performing experiments. Can be ignored.
flags.DEFINE_string('suffix', '', 'Used for experimenting')
flags.DEFINE_boolean('noemb', False, 'Used in experiments')
flags.DEFINE_float('endorse_scaling', 1.0, 'Used in experiments')
flags.DEFINE_boolean('endorse_embedding', False, 'Used in experiments')
flags.DEFINE_boolean('endorse_scales_variable', False, 'Used in experiments')
flags.DEFINE_boolean('endorse_scales_mlp', False, 'Used in experiments')
flags.DEFINE_boolean('endorse_scales_mlp_shared', False, 'Used in experiments')

FLAGS = flags.FLAGS
FLAGS(sys.argv)

from fairseq.models.bart import BARTModel as BModel

def flatten_list_of_lists(list_of_lists):
    return list(itertools.chain.from_iterable(list_of_lists))

def main(unused_argv):

    if len(unused_argv) != 1: # prints a message if you've entered flags incorrectly
        raise Exception("Problem with flags: %s" % unused_argv)

    exp_name = FLAGS.dataset_name + '_' + FLAGS.endorse_method
    if FLAGS.consolidation_method == 'sequential':
        exp_name += '_' + 'sequential'
    test_notbin_path = 'endorse_data/' + exp_name
    if FLAGS.noemb:
        exp_name += '_' + 'noemb'
    if FLAGS.endorse_embedding:
        exp_name += '_' + 'emb'
    if FLAGS.endorse_scaling != 1.0:
        exp_name += '_scale' + str(FLAGS.endorse_scaling)
    if FLAGS.endorse_duplicate_heads:
        exp_name += '_' + 'dup'
        if FLAGS.endorse_scales_variable:
            exp_name += '_variablescale'
            if FLAGS.endorse_scales_mlp:
                exp_name += '_mlp'
                if FLAGS.endorse_scales_mlp_shared:
                    exp_name += '_shared'
        else:
            exp_name += '_origscale' + str(FLAGS.endorse_original_head_scale)
    if FLAGS.suffix != '':
        exp_name += '_' + FLAGS.suffix

    if FLAGS.original_bart:
        exp_name = FLAGS.dataset_name + '_' + 'originalbart'
        test_notbin_path = FLAGS.dataset_name
    else:
        test_path = test_notbin_path + '-bin'
    exp_name_without_dataset_name = exp_name[len(FLAGS.dataset_name) + 1:]

    if FLAGS.min_len != 55 and 'wcep' in FLAGS.dataset_name:
        exp_name += '_' + 'min' + str(FLAGS.min_len)

    util.create_dirs(exp_name)

    checkpoints_dir = 'models/checkpoints_' + exp_name_without_dataset_name
    data_name_or_path = 'endorse_data/' + FLAGS.dataset_name + '_' + FLAGS.endorse_method
    if FLAGS.consolidation_method == 'sequential':
        data_name_or_path += '_sequential'
    print(checkpoints_dir)
    print(data_name_or_path)
    checkpoints_file = 'checkpoint_best.pt'
    bart = BModel.from_pretrained(
        checkpoints_dir + '/',
        checkpoint_file=checkpoints_file,
        data_name_or_path=data_name_or_path + '-bin'
    )
    bart.cuda()
    bart.eval()
    bart.half()
    count = 1
    if FLAGS.dataset_name == 'duc_2004' or FLAGS.dataset_name == 'tac_2011':
        min_len, max_len, lenpen = 300, 400, 0.05
        bsz = 2
    else:
        min_len, max_len, lenpen = FLAGS.min_len, 140, 2.0
        bsz = 1
    hypo_file_name = 'test.hypo'
    hypo_file = exp_name + '/' + hypo_file_name
    if FLAGS.original_bart:
        source_file = test_notbin_path + '/test.source'
        total = util.num_lines_in_file(source_file)
        with open(source_file) as source, open(hypo_file, 'w') as fout:
            sline = source.readline().strip()
            slines = [sline]
            for sline in tqdm(source, total=total):
                if count % bsz == 0:
                    with torch.no_grad():
                            hypotheses_batch = bart.sample(slines, beam=4, lenpen=lenpen,
                                                               max_len_b=max_len, min_len=min_len, no_repeat_ngram_size=3)
                    for hypothesis in hypotheses_batch:
                        fout.write(hypothesis + '\n')
                        fout.flush()
                    slines = []
                slines.append(sline.strip())
                count += 1
            if slines != []:
                hypotheses_batch = bart.sample(slines, beam=4, lenpen=lenpen,
                                                   max_len_b=max_len, min_len=min_len, no_repeat_ngram_size=3)
                for hypothesis in hypotheses_batch:
                    fout.write(hypothesis + '\n')
                    fout.flush()
    else:
        source_file = test_path + '/test.src'
        endorse_file = test_path + '/test.endorse'
        rouge_scores_file = test_path + '/test.rougescores'
        total = util.num_lines_in_file(source_file)
        with open(source_file) as source, open(hypo_file, 'w') as fout, open(endorse_file) as f_endorse, open(rouge_scores_file) as f_rouge_scores:
            sline = source.readline().strip()
            endorse = f_endorse.readline().strip()
            rouge_scores = f_rouge_scores.readline().strip()

            slines = [sline]
            endorses = [endorse]
            rouge_scoress = [rouge_scores]
            for sline, endorse, rouge_scores in tqdm(zip(source, f_endorse, f_rouge_scores), total=total):
                if count % bsz == 0:
                    with torch.no_grad():
                            hypotheses_batch = bart.sample_endorse(slines, endorses, rouge_scoress, beam=4, lenpen=lenpen,
                                                           max_len_b=max_len, min_len=min_len, no_repeat_ngram_size=3)
                    for hypothesis in hypotheses_batch:
                        fout.write(hypothesis + '\n')
                        fout.flush()
                    slines = []
                    endorses = []
                    rouge_scoress = []

                slines.append(sline.strip())
                endorses.append(endorse.strip())
                rouge_scoress.append(rouge_scores.strip())

                count += 1
            if slines != []:
                hypotheses_batch = bart.sample_endorse(slines, endorses, rouge_scoress, beam=4, lenpen=lenpen,
                                                   max_len_b=max_len, min_len=min_len, no_repeat_ngram_size=3)
                for hypothesis in hypotheses_batch:
                    fout.write(hypothesis + '\n')
                    fout.flush()


    ref_dir = FLAGS.dataset_name + '/' + 'ref'
    ref_tokenized_dir = FLAGS.dataset_name + '/' + 'ref_tokenized'
    if not os.path.exists(ref_dir):
        os.makedirs(ref_dir)
        if not os.path.exists(ref_tokenized_dir):
            os.makedirs(ref_tokenized_dir)
        summaries_file = 'summaries'
        if FLAGS.dataset_name != 'duc_2004' and FLAGS.dataset_name != 'tac_2011':
            summaries_file += '_raw'
        summaries_file += '.tsv'
        with open('data/processed/' + FLAGS.dataset_name + '/test/' + summaries_file, 'r') as f:
            text = f.read()
        summary_sents_list = [[summ.split('\t') for summ in summs.split('\t\t')] for summs in text.strip().split('\n')]
        for example_idx, summs in enumerate(summary_sents_list):
            for summ_idx, summ in enumerate(summs):
                ref_file = os.path.join(ref_dir, "%06d_%s.%s.txt" % (
                    example_idx, 'reference', chr(ord('A') + summ_idx)))
                ref_file_tokenized = os.path.join(ref_tokenized_dir, "%06d_%s.%s.txt" % (
                    example_idx, 'reference', chr(ord('A') + summ_idx)))
                with open(ref_file, 'w') as f:
                    f.write('\n'.join(summ))
                os.system("java edu.stanford.nlp.process.PTBTokenizer %s -preserveLines > %s" % (ref_file, ref_file_tokenized))

    dec_dir = exp_name + '/' + 'dec'
    dec_tokenized_dir = exp_name + '/' + 'dec_tokenized'
    if not os.path.exists(dec_dir):
        os.makedirs(dec_dir)
    if not os.path.exists(dec_tokenized_dir):
        os.makedirs(dec_tokenized_dir)
    with open(hypo_file, 'r') as f:
        text = f.read()
    hypos = text.strip().split('\n')
    for hypo_idx, hypo in enumerate(hypos):
        decoded_file = os.path.join(dec_dir, "%06d_decoded.txt" % hypo_idx)
        decoded_file_tokenized = os.path.join(dec_tokenized_dir, "%06d_decoded.txt" % hypo_idx)
        with open(os.path.join(decoded_file), 'w') as f:
            f.write(hypo)
        os.system("java edu.stanford.nlp.process.PTBTokenizer %s -preserveLines > %s" % (decoded_file, decoded_file_tokenized))
    if 'wcep' in FLAGS.dataset_name:
        l_param = 50
    else:
        l_param = 100
    results_dict = rouge_eval(ref_tokenized_dir, dec_tokenized_dir, l_param=l_param)
    if not os.path.exists(os.path.join(exp_name, 'rouge_results')):
        os.makedirs(os.path.join(exp_name, 'rouge_results'))
    rouge_log(results_dict, os.path.join(exp_name, 'rouge_results'))


def rouge_eval(ref_dir, dec_dir, l_param=100):
    """Evaluate the files in ref_dir and dec_dir with pyrouge, returning results_dict"""
    r = pyrouge.Rouge155()
    r.model_filename_pattern = '#ID#_reference.[A-Z].txt'
    r.system_filename_pattern = '(\d+)_decoded.txt'
    r.model_dir = ref_dir
    r.system_dir = dec_dir
    rouge_args = ['-e', r._data_dir,
         '-c',
         '95',
         '-2', '4',        # This is the only one we changed (changed the max skip from -1 to 4)
         '-U',
         '-r', '1000',
         '-n', '4',
         '-w', '1.2',
         '-a',
         '-l', str(l_param)]
    rouge_args = ' '.join(rouge_args)
    rouge_results = r.convert_and_evaluate(rouge_args=rouge_args)
    return r.output_to_dict(rouge_results)


def rouge_log(results_dict, dir_to_write, prefix=None, suffix=None):
    """Log ROUGE results to screen and write to file.
    Args:
        results_dict: the dictionary returned by pyrouge
        dir_to_write: the directory where we will write the results to"""
    log_str = ""
    for x in ["1","2","l","s4","su4"]:
        log_str += "\nROUGE-%s:\n" % x
        for y in ["f_score", "recall", "precision"]:
            key = "rouge_%s_%s" % (x,y)
            key_cb = key + "_cb"
            key_ce = key + "_ce"
            val = results_dict[key]
            val_cb = results_dict[key_cb]
            val_ce = results_dict[key_ce]
            log_str += "%s: %.4f with confidence interval (%.4f, %.4f)\n" % (key, val, val_cb, val_ce)
    print(log_str) # log to screen
    results_file = os.path.join(dir_to_write, "ROUGE_results.txt")
    print(("Writing final ROUGE results to %s...", results_file))
    with open(results_file, "w") as f:
        f.write(log_str)

    print("\nROUGE-1, ROUGE-2, ROUGE-SU4 (PRF):\n")
    sheets_str = ""
    last_rouge_metric = "su4" if (FLAGS.dataset_name == 'duc_2004' or FLAGS.dataset_name == 'tac_2011' or FLAGS.dataset_name == 'wcep_mds' or FLAGS.dataset_name == 'wcep') else "l"
    for x in ["1", "2", last_rouge_metric]:
        for y in ["precision", "recall", "f_score"]:
            key = "rouge_%s_%s" % (x, y)
            val = results_dict[key] * 100
            sheets_str += "%.2f\t" % (val)
    sheets_str += "\n"
    if prefix is not None:
        sheets_str = prefix + sheets_str
    if suffix is not None:
        sheets_str = sheets_str + suffix
    print(sheets_str)
    sheets_results_file = os.path.join(dir_to_write, "sheets_results.txt")
    with open(sheets_results_file, "w") as f:
        f.write(sheets_str)
    return sheets_str




if __name__ == '__main__':
    app.run(main)