
import util
import numpy as np
import rouge_functions
import collections
import six
import pyrouge
import logging as log
import os
import sys
import rouge
try:
    reload(sys)
    sys.setdefaultencoding('utf8')
except:
    a=0
import tempfile
tempfile.tempdir = os.path.expanduser('~') + "/tmp"

def _ngrams(words, n):
    queue = collections.deque(maxlen=n)
    for w in words:
        queue.append(w)
        if len(queue) == n:
            yield tuple(queue)

def _ngram_counts(words, n):
    return collections.Counter(_ngrams(words, n))

def _ngram_count(words, n):
    return max(len(words) - n + 1, 0)

def _counter_overlap(counter1, counter2):
    result = 0
    for k, v in six.iteritems(counter1):
        result += min(v, counter2[k])
    return result

def _safe_divide(numerator, denominator):
    if denominator > 0:
        return numerator / denominator
    else:
        return 0

def _safe_f1(matches, recall_total, precision_total, alpha):
    recall_score = _safe_divide(matches, recall_total)
    precision_score = _safe_divide(matches, precision_total)
    denom = (1.0 - alpha) * precision_score + alpha * recall_score
    if denom > 0.0:
        return (precision_score * recall_score) / denom
    else:
        return 0.0

def rouge_n(peer, models, n, alpha, metric='f1'):
    """
    Compute the ROUGE-N score of a peer with respect to one or more models, for
    a given value of `n`.
    """
    if len(models) == 0:
        return 0.

    if type(models[0]) is not list:
        models = [models]

    matches = 0
    recall_total = 0
    peer_counter = _ngram_counts(peer, n)
    for model in models:
        model_counter = _ngram_counts(model, n)
        matches += _counter_overlap(peer_counter, model_counter)
        recall_total += _ngram_count(model, n)
    precision_total = len(models) * _ngram_count(peer, n)
    if metric == 'f1':
        return _safe_f1(matches, recall_total, precision_total, alpha)
    elif metric == 'precision':
        return _safe_divide(matches, precision_total)
    elif metric == 'recall':
        return _safe_divide(matches, recall_total)
    else:
        raise Exception('must be one of {f1, recall, precision}')

def rouge_1(peer, models, alpha, metric='f1'):
    """
    Compute the ROUGE-1 (unigram) score of a peer with respect to one or more
    models.
    """
    return rouge_n(peer, models, 1, alpha, metric=metric)

def rouge_2(peer, models, alpha, metric='f1'):
    """
    Compute the ROUGE-2 (bigram) score of a peer with respect to one or more
    models.
    """
    return rouge_n(peer, models, 2, alpha, metric=metric)




def make_html_safe(s):
    """Replace any angled brackets in string s to avoid interfering with HTML attention visualizer."""
    s.replace("<", "&lt;")
    s.replace(">", "&gt;")
    return s

def write_for_rouge(all_reference_sents, decoded_sents, ex_index, ref_dir, dec_dir, decoded_words=None, file_name=None, log=True):
    """Write output to file in correct format for eval with pyrouge. This is called in single_pass mode.

    Args:
        all_reference_sents: list of list of strings
        decoded_sents: list of strings
        ex_index: int, the index with which to label the files
    """

    # First, divide decoded output into sentences if we supply words instead of sentences
    if decoded_words is not None:
        decoded_sents = []
        while len(decoded_words) > 0:
            try:
                fst_period_idx = decoded_words.index(".")
            except ValueError:  # there is text remaining that doesn't end in "."
                fst_period_idx = len(decoded_words)
            sent = decoded_words[:fst_period_idx + 1]  # sentence up to and including the period
            decoded_words = decoded_words[fst_period_idx + 1:]  # everything else
            decoded_sents.append(' '.join(sent))

    # pyrouge calls a perl script that puts the data into HTML files.
    # Therefore we need to make our output HTML safe.
    decoded_sents = [make_html_safe(w) for w in decoded_sents]
    all_reference_sents = [[make_html_safe(w) for w in abstract] for abstract in all_reference_sents]

    # Write to file
    if file_name is None:
        decoded_file = os.path.join(dec_dir, "%06d_decoded.txt" % ex_index)
    else:
        decoded_file = os.path.join(dec_dir, "%s_%06d.txt" % (file_name, ex_index))

    for abs_idx, abs in enumerate(all_reference_sents):
        if file_name is None:
            ref_file = os.path.join(ref_dir, "%06d_reference.%s.txt" % (
                ex_index, chr(ord('A') + abs_idx)))
        else:
            ref_file = os.path.join(ref_dir, "%s_%06d.%s.txt" % (
                file_name, ex_index, chr(ord('A') + abs_idx)))
        with open(ref_file, "w") as f:
            for idx, sent in enumerate(abs):
                f.write(sent + "\n")
    with open(decoded_file, "w") as f:
        for idx, sent in enumerate(decoded_sents):
            f.write(sent + "\n")

    # if log:
    #     logging.info("Wrote example %i to file" % ex_index)

def rouge_eval(ref_dir, dec_dir, l_param=100):
    """Evaluate the files in ref_dir and dec_dir with pyrouge, returning results_dict"""
    r = pyrouge.Rouge155()
#   r.model_filename_pattern = '#ID#_reference.txt'
    r.model_filename_pattern = '#ID#_reference.[A-Z].txt'
    r.system_filename_pattern = '(\d+)_decoded.txt'
    r.model_dir = ref_dir
    r.system_dir = dec_dir
    log.getLogger('global').setLevel(log.WARNING) # silence pyrouge logging
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


def rouge_log(results_dict, dir_to_write, prefix=None, suffix=None, dataset_name='cnn_dm'):
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

    sheets_str = ""
    last_rouge_metric = "su4" if dataset_name == 'duc_2004' else "l"
    print("\nROUGE-1, ROUGE-2, ROUGE-%s (PRF):\n" % last_rouge_metric.upper())
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

def word_f1_log(dir_to_write, suffix):
    print(suffix)
    results_file = os.path.join(dir_to_write, "word_f1.txt")
    with open(results_file, "w") as f:
        f.write(suffix)

def get_sent_similarities(summ_sent_tokens, article_sent_tokens, only_rouge_l=False, remove_stop_words=True, metric='recall'):
    # similarity_matrix = util.rouge_l_similarity_matrix(article_sent_tokens, [summ_sent], 'recall')
    # similarities = np.squeeze(similarity_matrix, 1)

    if not only_rouge_l:
        rouge_1 = rouge_functions.rouge_1_similarity_matrix(article_sent_tokens, summ_sent_tokens, metric, remove_stop_words)
        rouge_2 = rouge_functions.rouge_2_similarity_matrix(article_sent_tokens, summ_sent_tokens, metric, False)
        # similarities = (rouge_l + rouge_1 + rouge_2) / 3.0
        similarities = (rouge_1 + rouge_2) / 2.0

    return similarities


def my_lcs(string, sub):
    """
    Calculates longest common subsequence for a pair of tokenized strings
    :param string : list of str : tokens from a string split using whitespace
    :param sub : list of str : shorter string, also split using whitespace
    :returns: length (list of int): length of the longest common subsequence between the two strings

    Note: my_lcs only gives length of the longest common subsequence, not the actual LCS
    """
    if (len(string) < len(sub)):
        sub, string = string, sub

    lengths = [[0 for i in range(0, len(sub) + 1)] for j in range(0, len(string) + 1)]

    for j in range(1, len(sub) + 1):
        for i in range(1, len(string) + 1):
            if (string[i - 1] == sub[j - 1]):
                lengths[i][j] = lengths[i - 1][j - 1] + 1
            else:
                lengths[i][j] = max(lengths[i - 1][j], lengths[i][j - 1])

    return lengths[len(string)][len(sub)]

def calc_ROUGE_L_score(candidate, reference, metric='f1'):
    """
    Compute ROUGE-L score given one candidate and references for an image
    :param candidate: str : candidate sentence to be evaluated
    :param refs: list of str : COCO reference sentences for the particular image to be evaluated
    :returns score: int (ROUGE-L score for the candidate evaluated against references)
    """
    beta = 1.2
    prec = []
    rec = []

    if len(reference) == 0:
        return 0.

    if type(reference[0]) is not list:
        reference = [reference]

    for ref in reference:
        # compute the longest common subsequence
        lcs = my_lcs(ref, candidate)
        try:
            prec.append(lcs / float(len(candidate)))
            rec.append(lcs / float(len(ref)))
        except:
            print('Candidate', candidate)
            print('Reference', ref)
            raise


    prec_max = max(prec)
    rec_max = max(rec)

    if metric == 'f1':
        if (prec_max != 0 and rec_max != 0):
            score = ((1 + beta ** 2) * prec_max * rec_max) / float(rec_max + beta ** 2 * prec_max)
        else:
            score = 0.0
    elif metric == 'precision':
        score = prec_max
    elif metric == 'recall':
        score = rec_max
    else:
        raise Exception('Invalid metric argument: %s. Must be one of {f1,precision,recall}.' % metric)
    return score

# @profile
def rouge_l_similarity(article_sents, abstract_sents, vocab, metric='f1'):
    sentence_similarity = np.zeros([len(article_sents)], dtype=float)
    abstract_sents_removed_periods = util.remove_period_ids(abstract_sents, vocab)
    for article_sent_idx, article_sent in enumerate(article_sents):
        rouge_l = calc_ROUGE_L_score(article_sent, abstract_sents_removed_periods, metric=metric)
        sentence_similarity[article_sent_idx] = rouge_l
    return sentence_similarity

def rouge_l_similarity_matrix(article_sents, abstract_sents, metric='f1'):
    sentence_similarity_matrix = np.zeros([len(article_sents), len(abstract_sents)], dtype=float)
    # abstract_sents_removed_periods = remove_period_ids(abstract_sents, vocab)
    for article_sent_idx, article_sent in enumerate(article_sents):
        abs_similarities = []
        for abstract_sent_idx, abstract_sent in enumerate(abstract_sents):
            rouge_l = calc_ROUGE_L_score(article_sent, abstract_sent, metric=metric)
            abs_similarities.append(rouge_l)
            sentence_similarity_matrix[article_sent_idx, abstract_sent_idx] = rouge_l
    return sentence_similarity_matrix

def rouge_1_similarity_matrix(article_sents, abstract_sents, metric, should_remove_stop_words):
    if should_remove_stop_words:
        article_sents = [util.remove_stopwords_punctuation(sent) for sent in article_sents]
        abstract_sents = [util.remove_stopwords_punctuation(sent) for sent in abstract_sents]
    sentence_similarity_matrix = np.zeros([len(article_sents), len(abstract_sents)], dtype=float)
    # abstract_sents_removed_periods = remove_period_ids(abstract_sents, vocab)
    for article_sent_idx, article_sent in enumerate(article_sents):
        abs_similarities = []
        for abstract_sent_idx, abstract_sent in enumerate(abstract_sents):
            rouge = rouge_functions.rouge_1(article_sent, abstract_sent, 0.5, metric=metric)
            abs_similarities.append(rouge)
            sentence_similarity_matrix[article_sent_idx, abstract_sent_idx] = rouge
    return sentence_similarity_matrix

def rouge_2_similarity_matrix(article_sents, abstract_sents, metric, should_remove_stop_words):
    if should_remove_stop_words:
        article_sents = [util.remove_stopwords_punctuation(sent) for sent in article_sents]
        abstract_sents = [util.remove_stopwords_punctuation(sent) for sent in abstract_sents]
    sentence_similarity_matrix = np.zeros([len(article_sents), len(abstract_sents)], dtype=float)
    # abstract_sents_removed_periods = remove_period_ids(abstract_sents, vocab)
    for article_sent_idx, article_sent in enumerate(article_sents):
        abs_similarities = []
        for abstract_sent_idx, abstract_sent in enumerate(abstract_sents):
            rouge = rouge_functions.rouge_2(article_sent, abstract_sent, 0.5, metric=metric)
            abs_similarities.append(rouge)
            sentence_similarity_matrix[article_sent_idx, abstract_sent_idx] = rouge
    return sentence_similarity_matrix

def get_similarity(enc_tokens, summ_tokens, vocab):
    metric = 'precision'
    summ_tokens_combined = util.flatten_list_of_lists(summ_tokens)
    importances_hat = rouge_l_similarity(enc_tokens, summ_tokens_combined, vocab, metric=metric)
    return importances_hat

evaluator = rouge.Rouge(metrics=['rouge-n'],
                       max_n=1,
                       limit_length=True,
                       length_limit=100000,
                       length_limit_type='words',
                       apply_avg=False,
                       apply_best=True,
                       alpha=0.5, # Default F1_score
                       weight_factor=1.2,
                       stemming=True)
def multi_rouge_1(candidate, references, metric='f'):
    scores = evaluator.get_scores([candidate], [references])
    score = scores['rouge-1'][metric]
    return score
















