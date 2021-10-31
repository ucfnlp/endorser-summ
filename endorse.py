import util
import torch
from tqdm import tqdm
import os
from absl import flags
from absl import app
import sys
import itertools
import numpy as np
import pyrouge
import nltk
import bert_score_utils
import transformers
from collections import Counter

flags.DEFINE_string('exp_name', '', 'Experiment name that determines where output files go')
flags.DEFINE_string('hypo_file', 'test.hypo', 'Extractive summaries using just endorsement will be placed here')
flags.DEFINE_string('dataset_name', 'wcep_mds', 'Which dataset to use. Can be {wcep_mds, duc_2004, tac_2011}')
flags.DEFINE_string('dataset_split', 'all', 'Which dataset split to use. Must be one of {train, val, test, all}')
flags.DEFINE_boolean('candidate_as_endorser', True, 'Whether to use the candidate document as an additional endorser')
flags.DEFINE_boolean('allindividual', True, '')
flags.DEFINE_boolean('mcss', True, 'Uses the MCSS algorithm for endorsement')
flags.DEFINE_boolean('orig_bart_summarize', False, 'Whether to generate summaries using BART, which will act as endorsers')
flags.DEFINE_string('endorse_method', 'bertscore', 'Can be {pyramid, bertscore, stringmatch}')
flags.DEFINE_string('consolidation_method', 'reciprocal', 'Can be {reciprocal, sequential}')
flags.DEFINE_boolean('xlnet_tokenization', True, 'Which dataset to use. Can be {cnn_dm, xsum, duc_2004}')

FLAGS = flags.FLAGS
FLAGS(sys.argv)

from fairseq.models.bart import BARTModel as BModel

max_vocab_rank = 50264

def flatten_list_of_lists(list_of_lists):
    return list(itertools.chain.from_iterable(list_of_lists))

def write_highlighted_html(html, out_dir, example_idx):
    html = '''

<button id="btnPrev" class="float-left submit-button" >Prev</button>
<button id="btnNext" class="float-left submit-button" >Next</button>
<br><br>

<script type="text/javascript">
    document.getElementById("btnPrev").onclick = function () {
        location.href = "%06d_highlighted.html";
    };
    document.getElementById("btnNext").onclick = function () {
        location.href = "%06d_highlighted.html";
    };

    document.addEventListener("keyup",function(e){
   var key = e.which||e.keyCode;
   switch(key){
      //left arrow
      case 37:
         document.getElementById("btnPrev").click();
      break;
      //right arrow
      case 39:
         document.getElementById("btnNext").click();
      break;
   }
});
</script>

''' % (example_idx - 1, example_idx + 1) + html
    # if not os.path.exists(out_dir):
    #     os.makedirs(out_dir)
    path = os.path.join(out_dir, '%06d_highlighted.html' % example_idx)
    # path = os.path.join(out_dir + '.html')
    with open(path, 'w') as f:
        f.write(html)

def start_tag_highlight(color):
    return "<span style='background-color: " + color + ";'>"

def start_tag_text_color(color):
    return "<span style='color: " + color + ";'>"

def highlight_endorsements(orig_probs, targets):
    end_tag = "</span>"
    out_str = ''

    probs = max(orig_probs) - orig_probs

    min_prob, max_prob = min(probs), max(probs)
    min_prob = min(probs)
    if max_prob == 0:
        normalized_probs = probs
    else:
        normalized_probs = ((probs - min_prob) / max_prob)
    min_color = 255
    max_color = 79
    color_values = normalized_probs * (int(min_color) - int(max_color)) + int(max_color)
    cur_char_num = 0
    if FLAGS.endorse_method == 'pyramid':
        targets = ''.join(targets)
    for color, target in zip(color_values, targets):
        out_target = target
        start_tag = start_tag_highlight('ff' + hex(int(color))[2:] + 'ff')
        out_str += start_tag + out_target + end_tag
        cur_char_num += len(target)
    out_str += '<br>'
    str_format = '%3s '

    if FLAGS.endorse_method == 'bertscore' or FLAGS.mcss:
        def my_op(p):
            return int(round(p*10))
        op = my_op
    else:
        def my_op(p):
            return int(round(p))
        op = my_op

    out_str += ' '.join([str_format % str(op(p)) for p in orig_probs])

    return out_str

def string_matching_recall(cand_tokens, summ_tokens):
    matches = np.in1d(cand_tokens, summ_tokens)
    return matches


import math
# Function to find the maximum contiguous subarray
# and print its starting and end index
def maxSubArraySum(a):
    max_so_far = -math.inf - 1
    max_ending_here = 0
    start = 0
    end = 0
    s = 0

    for i in range(0, len(a)):

        max_ending_here += a[i]

        if max_so_far < max_ending_here:
            max_so_far = max_ending_here
            start = s
            end = i

        if max_ending_here < 0:
            max_ending_here = 0
            s = i + 1

    return max_so_far, start, end+1

def scores_to_summary_highest_scoring_sent(consolidated_scores_list, consolidated_targets, cand_sent_ends_list):
    best_score = 0
    best_sent = None
    for doc_idx, scores, target, cand_end_indices in zip(range(len(consolidated_scores_list)),
                                                         consolidated_scores_list, consolidated_targets,
                                                         cand_sent_ends_list):
        for sent_idx, start, end in zip(range(len(cand_end_indices)-1), cand_end_indices[:-1], cand_end_indices[1:]):
            sent_score = sum(scores[start:end])
            if sent_score > best_score:
                best_score = sent_score
                best_sent = consolidated_targets[doc_idx][start:end]
    summary = ''.join(best_sent)
    if summary[0] == ' ':
        summary = summary[1:]
    if summary[-1] == ' ':
        summary = summary[:-1]
    return summary

def convert_scores_xlnet_to_gpt2(xlnet_endorse_scores, xlnet_tokens, gpt2_tokens, orig_text):
    xlnet_text = ''.join(xlnet_tokens)
    gpt2_text = ''.join(gpt2_tokens)
    if xlnet_text[0] == ' ':
        xlnet_text = xlnet_text[1:]
        xlnet_tokens[0] = xlnet_tokens[0][1:]
    if xlnet_text[-1] == ' ':
        xlnet_text = xlnet_text[:-1]
        xlnet_tokens[-1] = xlnet_tokens[-1][:-1]
    if gpt2_text[0] == ' ':
        gpt2_text = gpt2_text[1:]
        gpt2_tokens[0] = gpt2_tokens[0][1:]
    if gpt2_text[-1] == ' ':
        gpt2_text = gpt2_text[:-1]
        gpt2_tokens[-1] = gpt2_tokens[-1][:-1]
    if len(xlnet_text) != len(gpt2_text):
        print(orig_text)
        print(xlnet_text)
        print(gpt2_text)
    xlnet_endorse_scores_token_level = []
    for token, score in zip(xlnet_tokens, xlnet_endorse_scores):
        xlnet_endorse_scores_token_level.extend([score] * len(token))
    ch_idx = 0
    gpt2_scores = []
    for token in gpt2_tokens:
        c = Counter([1,1,2,2,3])
        c.most_common(1)
        gpt2_score = Counter(xlnet_endorse_scores_token_level[ch_idx: ch_idx + len(token)]).most_common(1)[0][0]
        gpt2_scores.append(gpt2_score)
        ch_idx += len(token)
    if len(gpt2_scores) != len(gpt2_tokens):
        raise Exception('')
    return np.array(gpt2_scores)

from unidecode import unidecode
def remove_non_ascii(text):
    return unidecode(str(text, encoding = "utf-8"))

# @profile
def main(unused_argv):

    if len(unused_argv) != 1: # prints a message if you've entered flags incorrectly
        raise Exception("Problem with flags: %s" % unused_argv)

    if FLAGS.orig_bart_summarize:
        checkpoints_dir = '../bart.large.cnn'
        data_name_or_path = 'cnn_dm'
        bart = BModel.from_pretrained(
            checkpoints_dir + '/',
            checkpoint_file='model.pt',
            data_name_or_path=data_name_or_path + '-bin'
        )
        bart.cuda()
        bart.eval()
        bart.half()
    if FLAGS.dataset_split == 'all':
        if FLAGS.dataset_name == 'duc_2004' or FLAGS.dataset_name == 'tac_2011':
            dataset_splits = ['test']
        else:
            dataset_splits = ['test', 'val', 'train']
    else:
        dataset_splits = [FLAGS.dataset_split]
    for dataset_split in dataset_splits:
        articles_file = 'articles'
        if FLAGS.dataset_name != 'duc_2004' and FLAGS.dataset_name != 'tac_2011':
            articles_file += '_raw'
        articles_file += '.tsv'
        source_file = os.path.join('data/processed', FLAGS.dataset_name, dataset_split, articles_file)
        exp_name = FLAGS.endorse_method
        if FLAGS.consolidation_method == 'sequential':
            exp_name += '_' + 'sequential'
        log_dir = FLAGS.dataset_name + '/' + exp_name
        highlight_dir = log_dir + '/' + 'endorsement_highlights'
        util.create_dirs(highlight_dir)
        hypo_file = log_dir + '/' + 'test_fragments.hypo'
        endorse_scores_file = log_dir + '/' + 'endorse_scores_' + dataset_split + '.txt'
        token_ids_file = log_dir + '/' + 'token_ids_' + dataset_split + '.txt'
        orig_bart_summary_path = FLAGS.dataset_name + '/orig_bart_summaries_' + dataset_split + '.txt'

        if FLAGS.orig_bart_summarize:
            with open(source_file) as source, open(orig_bart_summary_path, 'w') as fout:
                for sline in tqdm(source, total=11490):
                    docs = sline.strip().split('\t\t')
                    docs = [' '.join(doc.split('\t')) for doc in docs]
                    with torch.no_grad():
                        hypotheses_batch = bart.sample(docs, beam=4, lenpen=2.0, max_len_b=140, min_len=55,
                                                       no_repeat_ngram_size=3, softmax=True)

                    for hypothesis in hypotheses_batch:
                        fout.write(hypothesis + '\n\n')
                    fout.write('\n\n')


        if FLAGS.endorse_method == 'pyramid':
            with open('stuff/duc_manual_pyramid.txt') as f:
                # List (for each doc) of list (for each line) of 3-tuples (start_char_idx, end_char_idx, pyr_weight)
                pyr_data = [[list(map(int, line.strip().split())) for line in pyr_doc.strip().split('\n')] for pyr_doc in
                            f.read().strip().split('\n\n')]
        if FLAGS.endorse_method == 'bertscore':
            if FLAGS.xlnet_tokenization:
                bertscore_model_type = 'xlnet-base-cased'
            else:
                bertscore_model_type = 'roberta-large'
            xlnet_tokenizer = transformers.XLNetTokenizer.from_pretrained('xlnet-base-cased')
            tokenizer = transformers.RobertaTokenizer.from_pretrained('roberta-large')
            bertscorer = bert_score_utils.BertScorer(lang="en", verbose=False, batch_size=2, model_type=bertscore_model_type)
        elif FLAGS.endorse_method == 'stringmatch':
            bertscore_model_type = 'roberta-large'
            tokenizer = transformers.RobertaTokenizer.from_pretrained('roberta-large')
        if FLAGS.endorse_method == 'bertscore' or FLAGS.endorse_method == 'stringmatch':
            def split_and_lemmatize(clusters):
                clusters_lemmatized_sent_tokens, clusters_sents_list = [], []
                for cluster in clusters:
                    summ_sents_list = [nltk.sent_tokenize(doc) for doc in cluster]
                    summ_sent_tokens_list = [[list(map(str.lower, nltk.word_tokenize(sent))) for sent in doc] for doc in summ_sents_list]
                    lemmatized_summ_sent_tokens_list = [util.lemmatize_sent_tokens(summ_sent_tokens) for summ_sent_tokens in summ_sent_tokens_list]
                    clusters_lemmatized_sent_tokens.append(lemmatized_summ_sent_tokens_list)
                    clusters_sents_list.append(summ_sents_list)
                return clusters_lemmatized_sent_tokens, clusters_sents_list
            with open(orig_bart_summary_path) as f:
                clusters = [cluster.strip().split('\n\n') for cluster in f.read().strip().split('\n\n\n\n')]
                clusters_lemmatized_sent_tokens, clusters_sents_list = split_and_lemmatize(clusters)
        total = util.num_lines_in_file(source_file)
        if dataset_split == 'test':
            fout = open(hypo_file, 'w')
        f_endorse_scores = open(endorse_scores_file, 'w')
        f_token_ids = open(token_ids_file, 'w')
        example_idx = 0
        with open(source_file) as source:
            for sline in tqdm(source, total=total):
                summ_sents_list = clusters_sents_list[example_idx]
                summs = clusters[example_idx]
                sline = sline.replace('\x7f', '')
                docs = sline.strip().split('\t\t')

                # Remove invalid docs that are empty string. Must also remove the BART original summaries for those docs too
                valid_doc_indices = [doc_idx for doc_idx, doc in enumerate(docs) if doc.strip() != '']
                docs = [doc for doc_idx, doc in enumerate(docs) if doc_idx in valid_doc_indices]
                summ_sents_list = [doc for doc_idx, doc in enumerate(summ_sents_list) if doc_idx in valid_doc_indices]
                summs = [doc for doc_idx, doc in enumerate(summs) if doc_idx in valid_doc_indices]

                docs_sents = [doc.strip().split('\t') for doc in docs]
                docs_sents = [util.fix_wcep_problems(raw_article_sents) for raw_article_sents in docs_sents]
                docs = [' '.join(doc.split('\t')) for doc in docs]
                if 'roberta' in bertscore_model_type or 'xlnet' in bertscore_model_type:
                    docs_sents = [[' ' + sent for sent in doc] for doc in docs_sents]
                    summ_sents_list = [[' ' + sent for sent in doc] for doc in summ_sents_list]
                if FLAGS.endorse_method == 'bertscore':
                    array_of_sims, refs_sents_ends, cand_sent_ends_list, _ = bertscorer.score(docs_sents, summ_sents_list, verbose=False, batch_size=2)
                elif FLAGS.endorse_method == 'stringmatch' or FLAGS.endorse_method == 'tokenfreq':
                    summ_sent_tokens_list = [[util.huggingface_tokenize(sent, tokenizer) for sent in summ_sents] for summ_sents in summ_sents_list]
                    ref_sent_lens = [[len(tokenizer.tokenize(sent)) for sent in sents] for sents in summ_sents_list]
                    hyp_sent_lens = [[len(tokenizer.tokenize(sent)) for sent in sents] for sents in docs_sents]
                    refs_sents_ends, cand_sent_ends_list = [np.insert(np.cumsum(sents), 0, 0) for sents in ref_sent_lens], [
                        np.insert(np.cumsum(sents), 0, 0) for sents in hyp_sent_lens]
                else:
                    cand_sent_ends_list = []
                if FLAGS.endorse_method == 'tokenfreq':
                    tf = {}
                    docs_sents_tokens = [[util.huggingface_tokenize(sent, tokenizer) for sent in sents] for sents in docs_sents]
                    for sents_tokens in docs_sents_tokens:
                        for tokens in sents_tokens:
                            for token in tokens:
                                token = token.lower()
                                if token not in tf:
                                    tf[token] = 0
                                tf[token] += 1

                consolidated_scores_list = []
                consolidated_targets = []
                sent_token_ids_list = []
                out_html = ''
                for doc_idx, cand_doc in enumerate(docs):
                    if FLAGS.candidate_as_endorser:
                        endorsers = summs
                        endorser_indices = list(range(len(endorsers)))
                    else:
                        endorsers = [endorser for endorser_idx, endorser in enumerate(summs) if endorser_idx != doc_idx]
                        endorser_indices = [end_idx for end_idx in list(range(len(summs))) if end_idx != doc_idx]

                    if FLAGS.endorse_method == 'pyramid':
                        scores = np.zeros([len(''.join(decoded_targets[0]))])
                        for pyr_tuple in pyr_data[doc_idx]:
                            scores[pyr_tuple[0] : pyr_tuple[1]] = pyr_tuple[2]
                        scores_list = [scores] * len(endorsers)
                    elif FLAGS.endorse_method == 'bertscore':
                        if 'roberta' in bertscore_model_type:
                            sent_token_ids = util.gpt2_tokenize(docs_sents[doc_idx], tokenizer)
                            token_ids = util.flatten_list_of_lists(sent_token_ids)
                            decoded_targets_single = util.huggingface_tokenize(''.join(docs_sents[doc_idx]), tokenizer)
                            util.assert_lengths_equal(token_ids, decoded_targets_single)
                            sent_token_ids_list.append(sent_token_ids)
                        else:
                            sent_token_ids = util.gpt2_tokenize(docs_sents[doc_idx], tokenizer)
                            xlnet_tokens = util.flatten_list_of_lists([xlnet_tokenizer.tokenize(sent) for sent in docs_sents[doc_idx]])
                            xlnet_tokens = [token.replace(transformers.tokenization_xlnet.SPIECE_UNDERLINE, ' ') for token in xlnet_tokens]
                            decoded_targets_single = xlnet_tokens
                            gpt2_tokens = util.huggingface_tokenize(''.join(docs_sents[doc_idx]), tokenizer)
                            sent_token_ids_list.append(sent_token_ids)
                        cand_end_indices = cand_sent_ends_list[doc_idx]
                        if len(decoded_targets_single) != cand_end_indices[-1]:
                            print(len(decoded_targets_single))
                            print(cand_end_indices[-1])
                        scores_list = []
                        for ref_idx, ref_sent_ends in enumerate(refs_sents_ends):
                            scores = np.zeros([cand_end_indices[-1]])
                            for ref_sent_idx, ref_start, ref_end in zip(range(len(ref_sent_ends)-1), ref_sent_ends[:-1], ref_sent_ends[1:]):
                                for cand_sent_idx, cand_start, cand_end in zip(range(len(docs_sents[doc_idx])), cand_end_indices[:-1], cand_end_indices[1:]):
                                    my_sim = array_of_sims[doc_idx][ref_idx]
                                    my_bert_scores = my_sim[cand_start:cand_end, ref_start:ref_end] # don't include last token (usually period)
                                    recall_scores = np.max(my_bert_scores, axis=1)
                                    if 'roberta' in bertscore_model_type:
                                        normalized_recall_scores = recall_scores - 0.85
                                    else:
                                        normalized_recall_scores = recall_scores - 0.55
                                    mcss_score, mcss_start, mcss_end = maxSubArraySum(normalized_recall_scores)
                                    if 'roberta' in bertscore_model_type:
                                        min_phrase_length = 5
                                    else:
                                        min_phrase_length = 5
                                    if mcss_end - mcss_start >= min_phrase_length:
                                        scores[cand_start+mcss_start:cand_start+mcss_end] = 1
                            if FLAGS.xlnet_tokenization:
                                scores = convert_scores_xlnet_to_gpt2(scores, xlnet_tokens, gpt2_tokens, ''.join(docs_sents[doc_idx]))
                                decoded_targets_single = gpt2_tokens
                            scores_list.append(scores)
                        decoded_targets = [decoded_targets_single] * len(endorsers)
                    elif FLAGS.endorse_method == 'stringmatch':
                        sent_token_ids = util.gpt2_tokenize(docs_sents[doc_idx], tokenizer)
                        token_ids = util.flatten_list_of_lists(sent_token_ids)
                        cand_doc_sents = docs_sents[doc_idx]
                        cand_doc_sent_tokens = [util.huggingface_tokenize(sent, tokenizer) for sent in cand_doc_sents]
                        decoded_targets_single = util.flatten_list_of_lists(cand_doc_sent_tokens)
                        util.assert_lengths_equal(token_ids, decoded_targets_single)
                        sent_token_ids_list.append(sent_token_ids)
                        decoded_targets = [decoded_targets_single] * len(endorsers)

                        cand_end_indices = cand_sent_ends_list[doc_idx]

                        if len(decoded_targets_single) != cand_end_indices[-1]:
                            print(len(decoded_targets_single))
                            print(cand_end_indices[-1])
                            raise Exception('Tokenizers dont match')

                        scores_list = []
                        for ref_idx, summ_sent_tokens in enumerate(summ_sent_tokens_list):
                            scores = np.zeros([cand_end_indices[-1]])
                            for ref_sent_idx, summ_tokens in enumerate(summ_sent_tokens):
                                for cand_sent_idx, cand_start, cand_end in zip(range(len(docs_sents[doc_idx])),
                                                                               cand_end_indices[:-1],
                                                                               cand_end_indices[1:]):
                                    cand_tokens_lower = [token.lower() for token in cand_doc_sent_tokens[cand_sent_idx]]
                                    summ_tokens_lower = [token.lower() for token in summ_tokens]
                                    recall_scores = string_matching_recall(cand_tokens_lower, summ_tokens_lower).astype(int)
                                    if 'roberta' in bertscore_model_type:
                                        normalized_recall_scores = recall_scores - 0.8
                                    else:
                                        normalized_recall_scores = recall_scores - 0.5
                                    mcss_score, mcss_start, mcss_end = maxSubArraySum(normalized_recall_scores)
                                    if mcss_end - mcss_start >= 5:
                                        scores[cand_start+mcss_start:cand_start+mcss_end] = 1
                            scores_list.append(scores)
                    if (doc_idx == 0 or FLAGS.allindividual) and example_idx < 50:
                        for endorser_idx, scores, target in zip(endorser_indices, scores_list, decoded_targets):
                            cand_html = highlight_endorsements(scores, target)
                            out_html += 'CANDIDATE DOCUMENT: %d -- ENDORSEMENTS FROM DOCUMENT %d<br><br>' % (doc_idx, endorser_idx)
                            out_html += cand_html + '<br><br>------------------------------------------------------<br><br>'
                    if FLAGS.consolidation_method == 'reciprocal':
                        consolidated_scores = sum(scores_list) / len(scores_list)
                    elif FLAGS.consolidation_method == 'std':
                        consolidated_scores = np.std(scores_list, axis=0)
                    elif FLAGS.consolidation_method == 'sequential':
                        corrected_scores_list = [(sc * -1) if endorser_idx < doc_idx else sc for endorser_idx, sc in enumerate(scores_list)]
                        consolidated_scores = sum(corrected_scores_list) / len(corrected_scores_list)
                        consolidated_scores[consolidated_scores < 0] = 0
                    else:
                        raise Exception('No handling for consolidation method: ' + FLAGS.consolidation_method)
                    consolidated_target = decoded_targets[0]
                    consolidated_scores_list.append(consolidated_scores)
                    consolidated_targets.append(consolidated_target)

                if example_idx < 50:
                    out_html += '<br><br>'
                    for doc_idx, scores, target in zip(range(len(consolidated_scores_list)), consolidated_scores_list, consolidated_targets):
                        out_html += 'CANDIDATE DOCUMENT: %d -- CONSOLIDATED ENDORSEMENTS FROM ALL DOCS<br><br>' % doc_idx
                        cand_html = highlight_endorsements(scores, target)
                        out_html += cand_html + '<br><br>------------------------------------------------------<br><br>'
                    write_highlighted_html(out_html, highlight_dir, example_idx)

                if dataset_split == 'test':
                    summary = scores_to_summary_highest_scoring_sent(consolidated_scores_list, consolidated_targets, cand_sent_ends_list)
                    fout.write(summary + '\n')
                    fout.flush()

                endorse_scores_text = '\t'.join([' '.join(list(map(str, map(int, sc*10)))) for sc in consolidated_scores_list])
                f_endorse_scores.write(endorse_scores_text + '\n')
                f_endorse_scores.flush()
                token_ids_text = '\t\t'.join(['\t'.join([' '.join(list(map(str, map(int, sc)))) for sc in token_ids]) for token_ids in sent_token_ids_list])
                f_token_ids.write(token_ids_text + '\n')
                f_token_ids.flush()
                example_idx += 1

        f_endorse_scores.close()
        f_token_ids.close()

        if dataset_split == 'test':
            ref_dir = FLAGS.dataset_name + '/' + 'ref'
            ref_tokenized_dir = FLAGS.dataset_name + '/' + 'ref_tokenized'
            if not os.path.exists(ref_dir):
                os.makedirs(ref_dir)
                if not os.path.exists(ref_tokenized_dir):
                    os.makedirs(ref_tokenized_dir)
            with open(f'data/processed/{FLAGS.dataset_name}/test/summaries.tsv', 'r') as f:
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

            print('Calculating ROUGE...')
            dec_dir = log_dir + '/' + 'decoded'
            dec_tokenized_dir = log_dir + '/' + 'decoded_tokenized'
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
            results_dict = rouge_eval(ref_tokenized_dir, dec_tokenized_dir)
            rouge_log(results_dict, log_dir)


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