import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from transformers import AutoTokenizer


__all__ = ['score', 'plot_example']

class BertScorer:

    def __init__(self, model_type=None, num_layers=None, verbose=False,
              idf=False, batch_size=64, nthreads=4, all_layers=False, lang=None,
              return_hash=False, rescale_with_baseline=False):
        assert lang is not None or model_type is not None, \
            'Either lang or model_type should be specified'

        if rescale_with_baseline:
            assert lang is not None, 'Need to specify Language when rescaling with baseline'

        if model_type is None:
            lang = lang.lower()
            model_type = lang2model[lang]
        if num_layers is None:
            num_layers = model2layers[model_type]


        if model_type.startswith('scibert'):
            tokenizer = AutoTokenizer.from_pretrained(cache_scibert(model_type))
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_type)
        model = get_model(model_type, num_layers, all_layers)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model.to(device)
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def score(self, cands, refs, model_type=None, num_layers=None, verbose=False,
              idf=False, batch_size=64, nthreads=4, all_layers=False, lang=None,
              return_hash=False, rescale_with_baseline=False):
        """
        BERTScore metric.

        Args:
            - :param: `cands` (list of str): candidate sentences
            - :param: `refs` (list of str): reference sentences
            - :param: `model_type` (str): bert specification, default using the suggested
                      model for the target langauge; has to specify at least one of
                      `model_type` or `lang`
            - :param: `num_layers` (int): the layer of representation to use.
                      default using the number of layer tuned on WMT16 correlation data
            - :param: `verbose` (bool): turn on intermediate status update
            - :param: `idf` (bool or dict): use idf weighting, can also be a precomputed idf_dict
            - :param: `batch_size` (int): bert score processing batch size
            - :param: `lang` (str): language of the sentences; has to specify
                      at least one of `model_type` or `lang`. `lang` needs to be
                      specified when `rescale_with_baseline` is True.
            - :param: `return_hash` (bool): return hash code of the setting
            - :param: `rescale_with_baseline` (bool): rescale bertscore with pre-computed baseline

        Return:
            - :param: `(P, R, F)`: each is of shape (N); N = number of input
                      candidate reference pairs. if returning hashcode, the
                      output will be ((P, R, F), hashcode).
        """
        assert len(cands) == len(refs), "Different number of candidates and references"

        if not idf:
            idf_dict = defaultdict(lambda: 1.)
            # set idf for [SEP] and [CLS] to 0
            idf_dict[self.tokenizer.sep_token_id] = 0
            idf_dict[self.tokenizer.cls_token_id] = 0
        elif isinstance(idf, dict):
            if verbose:
                print('using predefined IDF dict...')
            idf_dict = idf
        else:
            if verbose:
                print('preparing IDF dict...')
            start = time.perf_counter()
            idf_dict = get_idf_dict(refs, self.tokenizer, nthreads=nthreads)
            if verbose:
                print('done in {:.2f} seconds'.format(time.perf_counter() - start))

        if verbose:
            print('calculating scores...')
        start = time.perf_counter()
        array_of_sims, refs_sents_ends, hyp_sent_ends, tokenizer = bert_cos_score_idf(self.model, refs, cands, self.tokenizer, idf_dict,
                                       verbose=verbose, device=self.device,
                                       batch_size=batch_size, all_layers=all_layers)
        return array_of_sims, refs_sents_ends, hyp_sent_ends, tokenizer
        if rescale_with_baseline:
            baseline_path = os.path.join(
                os.path.dirname(__file__),
                f'rescale_baseline/{lang}/{model_type}.tsv'
            )
            if os.path.isfile(baseline_path):
                if not all_layers:
                    baselines = torch.from_numpy(
                        pd.read_csv(baseline_path).iloc[num_layers].to_numpy()
                    )[1:].float()
                else:
                    baselines = torch.from_numpy(
                        pd.read_csv(baseline_path).to_numpy()
                    )[:, 1:].unsqueeze(1).float()

                all_preds = (all_preds - baselines) / (1 - baselines)
            else:
                print(f'Warning: Baseline not Found for {model_type} on {lang} at {baseline_path}', file=sys.stderr)

        out = all_preds[..., 0], all_preds[..., 1], all_preds[..., 2] # P, R, F

        if verbose:
            time_diff = time.perf_counter() - start
            print(f'done in {time_diff:.2f} seconds, {len(refs) / time_diff:.2f} sentences/sec')

        if return_hash:
            return tuple([out, get_hash(model_type, num_layers, idf, rescale_with_baseline)])

        return out

    def plot_example(self, candidate, reference, model_type=None, num_layers=None, lang=None,
                     rescale_with_baseline=False, fname=''):
        """
        BERTScore metric.

        Args:
            - :param: `candidate` (str): a candidate sentence
            - :param: `reference` (str): a reference sentence
            - :param: `verbose` (bool): turn on intermediate status update
            - :param: `model_type` (str): bert specification, default using the suggested
                      model for the target langauge; has to specify at least one of
                      `model_type` or `lang`
            - :param: `num_layers` (int): the layer of representation to use
            - :param: `lang` (str): language of the sentences; has to specify
                      at least one of `model_type` or `lang`. `lang` needs to be
                      specified when `rescale_with_baseline` is True.
            - :param: `return_hash` (bool): return hash code of the setting
            - :param: `rescale_with_baseline` (bool): rescale bertscore with pre-computed baseline
            - :param: `fname` (str): path to save the output plot
        """
        assert isinstance(candidate, str)
        assert isinstance(reference, str)

        assert lang is not None or model_type is not None, \
            'Either lang or model_type should be specified'

        if rescale_with_baseline:
            assert lang is not None, 'Need to specify Language when rescaling with baseline'

        if model_type is None:
            lang = lang.lower()
            model_type = lang2model[lang]
        if num_layers is None:
            num_layers = model2layers[model_type]

        if model_type.startswith('scibert'):
            tokenizer = AutoTokenizer.from_pretrained(cache_scibert(model_type))
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_type)
        model = get_model(model_type, num_layers)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model.to(device)

        idf_dict = defaultdict(lambda: 1.)
        # set idf for [SEP] and [CLS] to 0
        idf_dict[tokenizer.sep_token_id] = 0
        idf_dict[tokenizer.cls_token_id] = 0

        hyp_embedding, masks, padded_idf = get_bert_embedding([candidate], model, tokenizer, idf_dict,
                                                             device=device, all_layers=False)
        ref_embedding, masks, padded_idf = get_bert_embedding([reference], model, tokenizer, idf_dict,
                                                             device=device, all_layers=False)
        ref_embedding.div_(torch.norm(ref_embedding, dim=-1).unsqueeze(-1))
        hyp_embedding.div_(torch.norm(hyp_embedding, dim=-1).unsqueeze(-1))
        sim = torch.bmm(hyp_embedding, ref_embedding.transpose(1, 2))
        sim = sim.squeeze(0).cpu()

        # remove [CLS] and [SEP] tokens
        r_tokens = [tokenizer.decode([i]) for i in sent_encode(tokenizer, reference)][1:-1]
        h_tokens = [tokenizer.decode([i]) for i in sent_encode(tokenizer, candidate)][1:-1]
        sim = sim[1:-1,1:-1]

        if rescale_with_baseline:
            baseline_path = os.path.join(
                os.path.dirname(__file__),
                f"rescale_baseline/{lang}/{model_type}.tsv"
            )
            if os.path.isfile(baseline_path):
                baselines = torch.from_numpy(
                    pd.read_csv(baseline_path).iloc[num_layers].to_numpy()
                )[1:].float()
                sim = (sim - baselines[2].item()) / (1 - baselines[2].item())
            else:
                print(f'Warning: Baseline not Found for {model_type} on {lang} at {baseline_path}', file=sys.stderr)

        fig, ax = plt.subplots(figsize=(len(r_tokens), len(h_tokens)))
        im = ax.imshow(sim, cmap='Blues', vmin=0, vmax=1)
        fig.colorbar(im, ax=ax)

        # We want to show all ticks...
        ax.set_xticks(np.arange(len(r_tokens)))
        ax.set_yticks(np.arange(len(h_tokens)))
        # ... and label them with the respective list entries
        ax.set_xticklabels(r_tokens, fontsize=10)
        ax.set_yticklabels(h_tokens, fontsize=10)
        plt.xlabel("Reference (tokenized)", fontsize=14)
        plt.ylabel("Candidate (tokenized)", fontsize=14)
        plt.title("Similarity Matrix", fontsize=14)

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        for i in range(len(h_tokens)):
            for j in range(len(r_tokens)):
                text = ax.text(j, i, '{:.3f}'.format(sim[i, j].item()),
                               ha="center", va="center", color="k" if sim[i, j].item() < 0.5 else "w")

        fig.tight_layout()
        if fname != "":
            plt.savefig(fname, dpi=100)
            print("Saved figure to file: ", fname)
        plt.show()


import sys
import os
import torch
from math import log
from itertools import chain
from collections import defaultdict, Counter
from multiprocessing import Pool
from functools import partial
from tqdm.auto import tqdm
from torch.nn.utils.rnn import pad_sequence

from transformers import BertConfig, XLNetConfig, XLMConfig, RobertaConfig
from transformers import AutoModel, GPT2Tokenizer

from bert_score import __version__
from transformers import __version__ as trans_version

__all__ = ['model_types']

SCIBERT_URL_DICT = {
    'scibert-scivocab-uncased': 'https://s3-us-west-2.amazonaws.com/ai2-s2-research/scibert/pytorch_models/scibert_scivocab_uncased.tar',
# recommend by the SciBERT authors
    'scibert-scivocab-cased': 'https://s3-us-west-2.amazonaws.com/ai2-s2-research/scibert/pytorch_models/scibert_scivocab_cased.tar',
    'scibert-basevocab-uncased': 'https://s3-us-west-2.amazonaws.com/ai2-s2-research/scibert/pytorch_models/scibert_basevocab_uncased.tar',
    'scibert-basevocab-cased': 'https://s3-us-west-2.amazonaws.com/ai2-s2-research/scibert/pytorch_models/scibert_basevocab_cased.tar',
}

model_types = list(BertConfig.pretrained_config_archive_map.keys()) + \
              list(XLNetConfig.pretrained_config_archive_map.keys()) + \
              list(RobertaConfig.pretrained_config_archive_map.keys()) + \
              list(XLMConfig.pretrained_config_archive_map.keys()) + \
              list(SCIBERT_URL_DICT.keys())

lang2model = defaultdict(lambda: 'bert-base-multilingual-cased')
lang2model.update({
    'en': 'roberta-large',
    'zh': 'bert-base-chinese',
    'en-sci': 'scibert-scivocab-uncased',
})

model2layers = {
    'bert-base-uncased': 9,  # 0.6925188074454226
    'bert-large-uncased': 18,  # 0.7210358126642836
    'bert-base-cased-finetuned-mrpc': 9,  # 0.6721947475618048
    'bert-base-multilingual-cased': 9,  # 0.6680687802637132
    'bert-base-chinese': 8,
    'roberta-base': 10,  # 0.706288719158983
    'roberta-large': 17,  # 0.7385974720781534
    'roberta-large-mnli': 19,  # 0.7535618640417984
    'roberta-base-openai-detector': 7,  # 0.7048158349432633
    'roberta-large-openai-detector': 15,  # 0.7462770207355116
    'xlnet-base-cased': 5,  # 0.6630103662114238
    'xlnet-large-cased': 7,  # 0.6598800720297179
    'xlm-mlm-en-2048': 6,  # 0.651262570131464
    'xlm-mlm-100-1280': 10,  # 0.6475166424401905
    'scibert-scivocab-uncased': 9,
    'scibert-scivocab-cased': 9,
    'scibert-basevocab-uncased': 9,
    'scibert-basevocab-cased': 9,
    'distilroberta-base': 5,  # 0.6797558139322964
    'distilbert-base-uncased': 5,  # 0.6756659152782033
    'distilbert-base-uncased-distilled-squad': 4,  # 0.6718318036382493
    'distilbert-base-multilingual-cased': 5,  # 0.6178131050889238
    'albert-base-v1': 10,  # 0.654237567249745
    'albert-large-v1': 17,  # 0.6755890754323239
    'albert-xlarge-v1': 16,  # 0.7031844211905911
    'albert-xxlarge-v1': 8,  # 0.7508642218461096
    'albert-base-v2': 9,  # 0.6682455591837927
    'albert-large-v2': 14,  # 0.7008537594374035
    'albert-xlarge-v2': 13,  # 0.7317228357869254
    'albert-xxlarge-v2': 8,  # 0.7505160257184014
    'xlm-roberta-base': 9,  # 0.6506799445871697
    'xlm-roberta-large': 17,  # 0.6941551437476826
}


def sent_encode(tokenizer, sent):
    "Encoding as sentence based on the tokenizer"
    if isinstance(tokenizer, GPT2Tokenizer):
        # for RoBERTa and GPT-2
        return tokenizer.encode(sent.strip(), add_special_tokens=True,
                                add_prefix_space=True,
                                max_length=tokenizer.max_len)
    else:
        return tokenizer.encode(sent.strip(), add_special_tokens=True,
                                max_length=tokenizer.max_len)


def get_model(model_type, num_layers, all_layers=None):
    if model_type.startswith('scibert'):
        model = AutoModel.from_pretrained(cache_scibert(model_type))
    else:
        model = AutoModel.from_pretrained(model_type)
    model.eval()

    # drop unused layers
    if not all_layers:
        if hasattr(model, 'n_layers'):  # xlm
            assert 0 <= num_layers <= model.n_layers, \
                f"Invalid num_layers: num_layers should be between 0 and {model.n_layers} for {model_type}"
            model.n_layers = num_layers
        elif hasattr(model, 'layer'):  # xlnet
            assert 0 <= num_layers <= len(model.layer), \
                f"Invalid num_layers: num_layers should be between 0 and {len(model.layer)} for {model_type}"
            model.layer = \
                torch.nn.ModuleList([layer for layer in model.layer[:num_layers]])
        elif hasattr(model, 'encoder'):  # albert
            if hasattr(model.encoder, 'albert_layer_groups'):
                assert 0 <= num_layers <= model.encoder.config.num_hidden_layers, \
                    f"Invalid num_layers: num_layers should be between 0 and {model.encoder.config.num_hidden_layers} for {model_type}"
                model.encoder.config.num_hidden_layers = num_layers
            else:  # bert, roberta
                assert 0 <= num_layers <= len(model.encoder.layer), \
                    f"Invalid num_layers: num_layers should be between 0 and {len(model.encoder.layer)} for {model_type}"
                model.encoder.layer = \
                    torch.nn.ModuleList([layer for layer in model.encoder.layer[:num_layers]])
        elif hasattr(model, 'transformer'):  # bert, roberta
            assert 0 <= num_layers <= len(model.transformer.layer), \
                f"Invalid num_layers: num_layers should be between 0 and {len(model.transformer.layer)} for {model_type}"
            model.transformer.layer = \
                torch.nn.ModuleList([layer for layer in model.transformer.layer[:num_layers]])
        else:
            raise ValueError("Not supported")
    else:
        if hasattr(model, 'output_hidden_states'):
            model.output_hidden_states = True
        elif hasattr(model, 'encoder'):
            model.encoder.output_hidden_states = True
        elif hasattr(model, 'transformer'):
            model.transformer.output_hidden_states = True
        else:
            raise ValueError(f'Not supported model architecture: {model_type}')

    return model


def padding(arr, pad_token, dtype=torch.long):
    lens = torch.LongTensor([len(a) for a in arr])
    max_len = lens.max().item()
    padded = torch.ones(len(arr), max_len, dtype=dtype) * pad_token
    mask = torch.zeros(len(arr), max_len, dtype=torch.long)
    for i, a in enumerate(arr):
        padded[i, :lens[i]] = torch.tensor(a, dtype=dtype)
        mask[i, :lens[i]] = 1
    return padded, lens, mask


def bert_encode(model, x, attention_mask, all_layers=False):
    model.eval()
    with torch.no_grad():
        out = model(x, attention_mask=attention_mask)
    if all_layers:
        emb = torch.stack(out[-1], dim=2)
    else:
        emb = out[0]
    return emb


def process(a, tokenizer=None):
    if tokenizer is not None:
        a = sent_encode(tokenizer, a)
    return set(a)


def get_idf_dict(arr, tokenizer, nthreads=4):
    """
    Returns mapping from word piece index to its inverse document frequency.


    Args:
        - :param: `arr` (list of str) : sentences to process.
        - :param: `tokenizer` : a BERT tokenizer corresponds to `model`.
        - :param: `nthreads` (int) : number of CPU threads to use
    """
    idf_count = Counter()
    num_docs = len(arr)

    process_partial = partial(process, tokenizer=tokenizer)

    with Pool(nthreads) as p:
        idf_count.update(chain.from_iterable(p.map(process_partial, arr)))

    idf_dict = defaultdict(lambda: log((num_docs + 1) / (1)))
    idf_dict.update({idx: log((num_docs + 1) / (c + 1)) for (idx, c) in idf_count.items()})
    return idf_dict


def collate_idf(arr, tokenizer, idf_dict, device='cuda:0'):
    """
    Helper function that pads a list of sentences to hvae the same length and
    loads idf score for words in the sentences.

    Args:
        - :param: `arr` (list of str): sentences to process.
        - :param: `tokenize` : a function that takes a string and return list
                  of tokens.
        - :param: `numericalize` : a function that takes a list of tokens and
                  return list of token indexes.
        - :param: `idf_dict` (dict): mapping a word piece index to its
                               inverse document frequency
        - :param: `pad` (str): the padding token.
        - :param: `device` (str): device to use, e.g. 'cpu' or 'cuda'
    """
    arr = [sent_encode(tokenizer, a)[:1024] for a in arr]

    idf_weights = [[idf_dict[i] for i in a] for a in arr]

    pad_token = tokenizer.pad_token_id

    padded, lens, mask = padding(arr, pad_token, dtype=torch.long)
    padded_idf, _, _ = padding(idf_weights, 0, dtype=torch.float)

    padded = padded.to(device=device)
    mask = mask.to(device=device)
    lens = lens.to(device=device)
    return padded, padded_idf, lens, mask


def get_bert_embedding(all_sens, model, tokenizer, idf_dict,
                       batch_size=-1, device='cuda:0',
                       all_layers=False):
    """
    Compute BERT embedding in batches.

    Args:
        - :param: `all_sens` (list of str) : sentences to encode.
        - :param: `model` : a BERT model from `pytorch_pretrained_bert`.
        - :param: `tokenizer` : a BERT tokenizer corresponds to `model`.
        - :param: `idf_dict` (dict) : mapping a word piece index to its
                               inverse document frequency
        - :param: `device` (str): device to use, e.g. 'cpu' or 'cuda'
    """

    padded_sens, padded_idf, lens, mask = collate_idf(all_sens,
                                                      tokenizer,
                                                      idf_dict,
                                                      device=device)

    if batch_size == -1: batch_size = len(all_sens)

    embeddings = []
    with torch.no_grad():
        for i in range(0, len(all_sens), batch_size):
            batch_embedding = bert_encode(model, padded_sens[i:i + batch_size],
                                          attention_mask=mask[i:i + batch_size],
                                          all_layers=all_layers)
            embeddings.append(batch_embedding)
            del batch_embedding

    total_embedding = torch.cat(embeddings, dim=0)

    return total_embedding, mask, padded_idf


def greedy_cos_idf(ref_embedding, ref_masks, ref_idf,
                   hyp_embedding, hyp_masks, hyp_idf,
                   all_layers=False):
    """
    Compute greedy matching based on cosine similarity.

    Args:
        - :param: `ref_embedding` (torch.Tensor):
                   embeddings of reference sentences, BxKxd,
                   B: batch size, K: longest length, d: bert dimenison
        - :param: `ref_lens` (list of int): list of reference sentence length.
        - :param: `ref_masks` (torch.LongTensor): BxKxK, BERT attention mask for
                   reference sentences.
        - :param: `ref_idf` (torch.Tensor): BxK, idf score of each word
                   piece in the reference setence
        - :param: `hyp_embedding` (torch.Tensor):
                   embeddings of candidate sentences, BxKxd,
                   B: batch size, K: longest length, d: bert dimenison
        - :param: `hyp_lens` (list of int): list of candidate sentence length.
        - :param: `hyp_masks` (torch.LongTensor): BxKxK, BERT attention mask for
                   candidate sentences.
        - :param: `hyp_idf` (torch.Tensor): BxK, idf score of each word
                   piece in the candidate setence
    """
    ref_embedding.div_(torch.norm(ref_embedding, dim=-1).unsqueeze(-1))
    hyp_embedding.div_(torch.norm(hyp_embedding, dim=-1).unsqueeze(-1))

    if all_layers:
        B, _, L, D = hyp_embedding.size()
        hyp_embedding = hyp_embedding.transpose(1, 2).transpose(0, 1) \
            .contiguous().view(L * B, hyp_embedding.size(1), D)
        ref_embedding = ref_embedding.transpose(1, 2).transpose(0, 1) \
            .contiguous().view(L * B, ref_embedding.size(1), D)
    batch_size = ref_embedding.size(0)
    sim = torch.bmm(hyp_embedding, ref_embedding.transpose(1, 2))
    masks = torch.bmm(hyp_masks.unsqueeze(2).float(), ref_masks.unsqueeze(1).float())
    if all_layers:
        masks = masks.unsqueeze(0).expand(L, -1, -1, -1) \
            .contiguous().view_as(sim)
    else:
        masks = masks.expand(batch_size, -1, -1) \
            .contiguous().view_as(sim)

    masks = masks.float().to(sim.device)
    sim = sim * masks
    return sim

    word_precision = sim.max(dim=2)[0]
    word_recall = sim.max(dim=1)[0]

    hyp_idf.div_(hyp_idf.sum(dim=1, keepdim=True))
    ref_idf.div_(ref_idf.sum(dim=1, keepdim=True))
    precision_scale = hyp_idf.to(word_precision.device)
    recall_scale = ref_idf.to(word_recall.device)
    if all_layers:
        precision_scale = precision_scale.unsqueeze(0) \
            .expand(L, B, -1).contiguous().view_as(word_precision)
        recall_scale = recall_scale.unsqueeze(0) \
            .expand(L, B, -1).contiguous().view_as(word_recall)
    P = (word_precision * precision_scale).sum(dim=1)
    R = (word_recall * recall_scale).sum(dim=1)
    F = 2 * P * R / (P + R)

    hyp_zero_mask = hyp_masks.sum(dim=1).eq(2)
    ref_zero_mask = ref_masks.sum(dim=1).eq(2)

    if all_layers:
        P = P.view(L, B)
        R = R.view(L, B)
        F = F.view(L, B)

    if torch.any(hyp_zero_mask):
        print("Warning: Empty candidate sentence; Setting precision to be 0.", file=sys.stderr)
        P = P.masked_fill(hyp_zero_mask, 0.)

    if torch.any(ref_zero_mask):
        print("Warning: Empty candidate sentence; Setting recall to be 0.", file=sys.stderr)
        R = R.masked_fill(ref_zero_mask, 0.)

    F = F.masked_fill(torch.isnan(F), 0.)

    return P, R, F


def bert_cos_score_idf(model, refs_sents, hyps_sents, tokenizer, idf_dict,
                       verbose=False, batch_size=64, device='cuda:0',
                       all_layers=False):
    """
    Compute BERTScore.

    Args:
        - :param: `model` : a BERT model in `pytorch_pretrained_bert`
        - :param: `refs` (list of str): reference sentences
        - :param: `hyps` (list of str): candidate sentences
        - :param: `tokenzier` : a BERT tokenizer corresponds to `model`
        - :param: `idf_dict` : a dictionary mapping a word piece index to its
                               inverse document frequency
        - :param: `verbose` (bool): turn on intermediate status update
        - :param: `batch_size` (int): bert score processing batch size
        - :param: `device` (str): device to use, e.g. 'cpu' or 'cuda'
    """
    preds = []

    def my_sort(l):
        return sorted(list(l), key=lambda x: len(x.split(" ")))

    ref_sent_lens = [[len(tokenizer.tokenize(sent)) for sent in sents] for sents in refs_sents]
    hyp_sent_lens = [[len(tokenizer.tokenize(sent)) for sent in sents] for sents in hyps_sents]
    refs_sents_ends, hyp_sent_ends = [np.insert(np.cumsum(sents), 0, 0) for sents in ref_sent_lens], [np.insert(np.cumsum(sents), 0, 0) for sents in hyp_sent_lens]
    ref_combined = [''.join(sents) for sents in refs_sents]
    hyp_combined = [''.join(sents) for sents in hyps_sents]
    # refs = refs_sents[0]
    # hyps = hyps_sents[0]
    refs = ref_combined
    hyps = hyp_combined

    sentences = my_sort(ref_combined + hyp_combined)
    ref_embs = []
    ref_idfs = []
    hyp_embs = []
    hyp_idfs = []
    iter_range = range(0, len(sentences), batch_size)
    if verbose:
        print("computing bert embedding.")
        iter_range = tqdm(iter_range)
    stats_dict = dict()
    for batch_start in iter_range:
        sen_batch = sentences[batch_start:batch_start + batch_size]
        embs, masks, padded_idf = get_bert_embedding(sen_batch, model, tokenizer, idf_dict,
                                                     device=device, all_layers=all_layers)
        embs = embs.cpu()
        masks = masks.cpu()
        padded_idf = padded_idf.cpu()
        for i, sen in enumerate(sen_batch):
            if i == 0:
                end_indices = refs_sents_ends
            else:
                end_indices = hyp_sent_ends
            sequence_len = masks[i].sum().item()
            emb = embs[i, :sequence_len]
            idf = padded_idf[i, :sequence_len]
            # for idx, end_idx in enumerate(end_indices):
            #     if idx == 0:
            #         start_idx = 0
            #         sen = refs_sents[0][idx]
            #     else:
            #         start_idx = refs_sents_ends[idx-1]
            #         sen = hyps_sents[0][idx]
            #     my_emb = emb[start_idx:end_idx]
            #     my_idf = idf[start_idx:end_idx]
            #     if i == 0:
            #         ref_embs.append(my_emb)
            #         ref_idfs.append(my_idf)
            #     else:
            #         hyp_embs.append(my_emb)
            #         hyp_idfs.append(my_idf)
            #     stats_dict[sen] = (emb, idf)
            stats_dict[sen] = (emb, idf)

    def pad_batch_stats(sen_batch, stats_dict, device):
        stats = [stats_dict[s] for s in sen_batch]
        emb, idf = zip(*stats)
        lens = [e.size(0) for e in emb]
        emb_pad = pad_sequence(emb, batch_first=True, padding_value=2.)
        idf_pad = pad_sequence(idf, batch_first=True)

        def length_to_mask(lens):
            lens = torch.tensor(lens, dtype=torch.long)
            max_len = max(lens)
            base = torch.arange(max_len, dtype=torch.long) \
                .expand(len(lens), max_len)
            return base < lens.unsqueeze(1)

        pad_mask = length_to_mask(lens)
        return emb_pad.to(device), pad_mask.to(device), idf_pad.to(device)

    device = next(model.parameters()).device
    iter_range = range(0, len(refs), batch_size)
    if verbose:
        print("computing greedy matching.")
        iter_range = tqdm(iter_range)
    array_of_sims = []  # num_hyps x num_refs
    for hyp in hyp_combined:
        hyp_sims = []
        for ref in ref_combined:
            batch_refs = [ref]
            batch_hyps = [hyp]
            ref_stats = pad_batch_stats(batch_refs, stats_dict, device)
            hyp_stats = pad_batch_stats(batch_hyps, stats_dict, device)
            sim = greedy_cos_idf(*ref_stats, *hyp_stats, all_layers)
            hyp_sims.append(sim[0].cpu().numpy())
        array_of_sims.append(hyp_sims)
    return array_of_sims, refs_sents_ends, hyp_sent_ends, tokenizer
    for batch_start in iter_range:
        batch_refs = refs[batch_start:batch_start + batch_size]
        batch_hyps = hyps[batch_start:batch_start + batch_size]
        ref_stats = pad_batch_stats(batch_refs, stats_dict, device)
        hyp_stats = pad_batch_stats(batch_hyps, stats_dict, device)

        sim = greedy_cos_idf(*ref_stats, *hyp_stats, all_layers)
        preds.append(torch.stack((P, R, F1), dim=-1).cpu())
    preds = torch.cat(preds, dim=1 if all_layers else 0)
    return preds


def get_hash(model, num_layers, idf, rescale_with_baseline):
    msg = '{}_L{}{}_version={}(hug_trans={})'.format(
        model, num_layers, '_idf' if idf else '_no-idf', __version__, trans_version)
    if rescale_with_baseline:
        msg += "-rescaled"
    return msg


def cache_scibert(model_type, cache_folder='~/.cache/torch/transformers'):
    if not model_type.startswith('scibert'):
        return model_type

    underscore_model_type = model_type.replace('-', '_')
    cache_folder = os.path.abspath(cache_folder)
    filename = os.path.join(cache_folder, underscore_model_type)

    # download SciBERT models
    if not os.path.exists(filename):
        cmd = f'mkdir -p {cache_folder}; cd {cache_folder};'
        cmd += f'wget {SCIBERT_URL_DICT[model_type]}; tar -xvf {underscore_model_type}.tar;'
        cmd += f'rm -f {underscore_model_type}.tar ; cd {underscore_model_type}; tar -zxvf weights.tar.gz; mv weights/* .;'
        cmd += f'rm -f weights.tar.gz; rmdir weights; mv bert_config.json config.json;'
        print(cmd)
        print(f'downloading {model_type} model')
        os.system(cmd)

    # fix the missing files in scibert
    json_file = os.path.join(filename, 'special_tokens_map.json')
    if not os.path.exists(json_file):
        with open(json_file, 'w') as f:
            print(
                '{"unk_token": "[UNK]", "sep_token": "[SEP]", "pad_token": "[PAD]", "cls_token": "[CLS]", "mask_token": "[MASK]"}',
                file=f)

    json_file = os.path.join(filename, 'added_tokens.json')
    if not os.path.exists(json_file):
        with open(json_file, 'w') as f:
            print('{}', file=f)

    if 'uncased' in model_type:
        json_file = os.path.join(filename, 'tokenizer_config.json')
        if not os.path.exists(json_file):
            with open(json_file, 'w') as f:
                print('{"do_lower_case": true, "max_len": 512, "init_inputs": []}', file=f)

    return filename
