# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List

from fairseq import utils
from fairseq.data import encoders

from fairseq.data import (
    LanguagePairDataset,
)


logger = logging.getLogger(__name__)


class BARTHubInterface(nn.Module):
    """A simple PyTorch Hub interface to BART.

    Usage: https://github.com/pytorch/fairseq/tree/master/examples/BART
    """

    def __init__(self, args, task, model):
        super().__init__()
        self.args = args
        self.task = task
        self.model = model

        self.bpe = encoders.build_bpe(args)

        self.max_positions = min(utils.resolve_max_positions(
            self.task.max_positions(),
            self.model.max_positions(),
        ))

        # this is useful for determining the device
        self.register_buffer('_float_tensor', torch.tensor([0], dtype=torch.float))

    @property
    def device(self):
        return self._float_tensor.device

    def encode(self, sentence: str, *addl_sentences, no_separator=True) -> torch.LongTensor:
        """
        BPE-encode a sentence (or multiple sentences).

        Every sequence begins with a beginning-of-sentence (`<s>`) symbol.
        Every sentence ends with an end-of-sentence (`</s>`).

        Example (single sentence): `<s> a b c </s>`
        Example (sentence pair): `<s> d e f </s> 1 2 3 </s>`

        The BPE encoding follows GPT-2. One subtle detail is that the GPT-2 BPE
        requires leading spaces. For example::

            >>> bart.encode('Hello world').tolist()
            [0, 31414, 232, 2]
            >>> bart.encode(' world').tolist()
            [0, 232, 2]
            >>> bart.encode('world').tolist()
            [0, 8331, 2]
        """
        tokens = self.bpe.encode(sentence)
        if len(tokens.split(' ')) > self.max_positions - 2:
            tokens = ' '.join(tokens.split(' ')[:self.max_positions - 2])
        bpe_sentence = '<s> ' + tokens + ' </s>'
        for s in addl_sentences:
            bpe_sentence += (' </s>' if not no_separator else '')
            bpe_sentence += ' ' + self.bpe.encode(s) + ' </s>'
        tokens = self.task.source_dictionary.encode_line(bpe_sentence, append_eos=False)
        return tokens.long()

    def decode(self, tokens: torch.LongTensor):
        assert tokens.dim() == 1
        tokens = tokens.cpu().numpy()
        if tokens[0] == self.task.source_dictionary.bos():
            tokens = tokens[1:]  # remove <s>
        eos_mask = (tokens == self.task.source_dictionary.eos())
        doc_mask = eos_mask[1:] & eos_mask[:-1]
        sentences = np.split(tokens, doc_mask.nonzero()[0] + 1)
        sentences = [self.bpe.decode(self.task.source_dictionary.string(s)) for s in sentences]
        if len(sentences) == 1:
            return sentences[0]
        return sentences

    def decode_list(self, orig_tokens: torch.LongTensor):
        assert orig_tokens.dim() == 1
        tokens = orig_tokens.clone()
        tokens = tokens.cpu().numpy()
        if tokens[0] == self.task.source_dictionary.bos():
            tokens = tokens[1:]  # remove <s>
        eos_mask = (tokens == self.task.source_dictionary.eos())
        doc_mask = eos_mask[1:] & eos_mask[:-1]
        sentences = np.split(tokens, doc_mask.nonzero()[0] + 1)
        # sentences = [tokens]
        sentences = [self.bpe.decode_list(self.task.source_dictionary.string(s)) for s in sentences]
        if len(sentences) == 1:
            if orig_tokens[0] == self.task.source_dictionary.bos():
                sentences[0].insert(0, '[start] ')
            if orig_tokens[-1] == self.task.source_dictionary.eos():
                sentences[0].append(' [end]')
            return sentences[0]
        if orig_tokens[0] == self.task.source_dictionary.bos():
            sentences.insert(0, '[start] ')
        if orig_tokens[-1] == self.task.source_dictionary.eos():
            sentences.append(' [end]')
        return sentences

    def _build_sample(self, src_tokens: List[torch.LongTensor], endorse_scores=None, rouge_scores=None):
        # assert torch.is_tensor(src_tokens)
        dataset = self.task.build_dataset_for_inference(
            src_tokens,
            [x.numel() for x in src_tokens],
            endorse_scores=endorse_scores,
            rouge_scores=rouge_scores,
        )
        sample = dataset.collater(dataset)
        sample = utils.apply_to_sample(
            lambda tensor: tensor.to(self.device),
            sample
        )
        return sample

    def _build_endorse_sample(self, src_tokens: List[torch.LongTensor], target_tokens: List[torch.LongTensor]):
        # assert torch.is_tensor(src_tokens)
        dataset = LanguagePairDataset(
            src_tokens,
            [x.numel() for x in src_tokens],
            self.task.source_dictionary,
            target_tokens,
            [x.numel() for x in target_tokens],
            self.task.target_dictionary,
        )
        sample = dataset.collater(dataset)
        sample = utils.apply_to_sample(
            lambda tensor: tensor.to(self.device),
            sample
        )
        return sample

    def encode_source(self, token_ids):
        return torch.from_numpy(np.array(list(map(int, token_ids.split() + [self.task.source_dictionary.eos()]))))
    def encode_endorse(self, endorse_line):
        extra_token = 0
        return torch.from_numpy(np.array([int(item) + 1 for item in endorse_line.split()] + ([extra_token] if extra_token is not None else [])))
    def encode_rouge_scores(self, endorse_line):
        row = []
        for item in endorse_line.strip().split():
            row.append(float(item))
        if sum(row) <= 0:
            row = [1.0, 1.0, 1.0]
        return torch.from_numpy(np.array(row))

    def sample(self, sentences: List[str], beam: int = 1, verbose: bool = False, **kwargs) -> str:
        input = [self.encode(sentence) for sentence in sentences]
        hypos = self.generate(input, beam, verbose, **kwargs)
        return [self.decode(x['tokens']) for x in hypos]

    def sample_endorse(self, sentences: List[str], endorse_scores, rouge_scores, beam: int = 1, verbose: bool = False, **kwargs) -> str:
        input = [self.encode_source(sentence) for sentence in sentences]
        endorse_input = [self.encode_endorse(scores) for scores in endorse_scores]
        rouge_input = [self.encode_rouge_scores(scores) for scores in rouge_scores]
        hypos = self.endorse_generate(input, endorse_input, rouge_input, beam, verbose, **kwargs)
        decoded_out = [self.decode(x['tokens']) for x in hypos]
        decoded_out = [summ[1:] if summ[0] == ' ' else summ for summ in decoded_out]
        return decoded_out

    # @profile
    def endorsement(self, sentences: List[str], target_sentences: List[str], softmax=True, suppress_eos=True, **kwargs) -> str:
        input = [self.encode(sentence) for sentence in sentences]
        target = [self.encode(sentence) for sentence in target_sentences]
        probs = self.get_raw_probs(input, target, softmax=softmax, suppress_eos=suppress_eos)
        decoded_targets = [self.decode_list(x[1:-1]) for x in target]
        target_numpy = [tar.cpu().numpy()[1:-1] for tar in target]
        for p, t in zip(probs, decoded_targets):
            if len(p) != len(t):
                a=0
                raise Exception('len(p) != len(t): %d %d' % (len(p), len(t)))
        return probs, decoded_targets, target_numpy

    # @profile
    def get_raw_probs(self, tokens: List[torch.LongTensor], targets: List[torch.LongTensor], softmax=True, suppress_eos=True):
        sample = self._build_endorse_sample(tokens, targets)
        net_output = self.model(**sample['net_input'])
        probs = net_output[0]
        probs = probs[:,1:-1,:]
        if suppress_eos:
            probs[:,:,self.task.source_dictionary.bos()] = -np.inf
            probs[:,:,self.task.source_dictionary.eos()] = -np.inf
        if softmax:
            probs = torch.softmax(probs, -1)
        probs = [v.cpu().numpy() for _, v in sorted(zip(sample['id'].tolist(), probs))]
        return probs

    def get_probs_of_sentence(self, tokens: List[torch.LongTensor], targets: List[torch.LongTensor], softmax=True, suppress_eos=True):
        sample = self._build_endorse_sample(tokens, targets)
        net_output = self.model(**sample['net_input'])
        probs = net_output[0]
        if suppress_eos:
            probs[:,:,self.task.source_dictionary.eos()] = -np.inf
        if softmax:
            probs = torch.softmax(probs, -1)
        probs = [v for _, v in sorted(zip(sample['id'].tolist(), probs))]
        # target_word_probs = []
        # for prob, target in zip(probs, targets):
        #     my_target_word_probs = prob[np.arange(len(target)), target]
        #     target_word_probs.append(my_target_word_probs.cpu().numpy())
        target_word_ranks = []
        for prob, target in zip(probs, targets):
            my_target_word_ranks = []
            prob = prob.cpu().numpy()
            target = target.cpu().numpy()
            # sorted_probs = np.argsort(prob, -1)
            for token_idx, token in enumerate(target):
                p = prob[token_idx]
                rank = (p >= p[token]).sum()
                # idx = np.where(sorted_probs[token_idx] == token)[0][0]
                # rank = sorted_probs.shape[-1] - idx
                my_target_word_ranks.append(rank)
            target_word_ranks.append(np.array(my_target_word_ranks))
        return target_word_ranks




    def generate(self, tokens: List[torch.LongTensor], beam: int = 5, verbose: bool = False, **kwargs) -> torch.LongTensor:
        sample = self._build_sample(tokens)

        # build generator using current args as well as any kwargs
        gen_args = copy.copy(self.args)
        gen_args.beam = beam
        for k, v in kwargs.items():
            setattr(gen_args, k, v)
        generator = self.task.build_generator([self.model], gen_args)
        translations = self.task.inference_step(
            generator,
            [self.model],
            sample,
            prefix_tokens=sample['net_input']['src_tokens'].new_zeros((len(tokens), 1)).fill_(self.task.source_dictionary.bos()),
        )

        if verbose:
            src_str_with_unk = self.string(tokens)
            logger.info('S\t{}'.format(src_str_with_unk))

        def getarg(name, default):
            return getattr(gen_args, name, getattr(self.args, name, default))

        # Process top predictions
        hypos = [x[0] for x in translations]
        hypos = [v for _, v in sorted(zip(sample['id'].tolist(), hypos))]
        return hypos

    def endorse_generate(self, tokens: List[torch.LongTensor], endorse_scores, rouge_scores, beam: int = 5, verbose: bool = False, **kwargs) -> torch.LongTensor:
        sample = self._build_sample(tokens, endorse_scores=endorse_scores, rouge_scores=rouge_scores)

        # build generator using current args as well as any kwargs
        gen_args = copy.copy(self.args)
        gen_args.beam = beam
        for k, v in kwargs.items():
            setattr(gen_args, k, v)
        generator = self.task.build_generator([self.model], gen_args)
        translations = self.task.inference_step(
            generator,
            [self.model],
            sample,
            prefix_tokens=sample['net_input']['src_tokens'].new_zeros((len(tokens), 1)).fill_(self.task.source_dictionary.bos()),
        )

        if verbose:
            src_str_with_unk = self.string(tokens)
            logger.info('S\t{}'.format(src_str_with_unk))

        def getarg(name, default):
            return getattr(gen_args, name, getattr(self.args, name, default))

        # Process top predictions
        hypos = [x[0] for x in translations]
        hypos = [v for _, v in sorted(zip(sample['id'].tolist(), hypos))]
        return hypos

    def extract_features(self, tokens: torch.LongTensor, return_all_hiddens: bool = False) -> torch.Tensor:
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)
        if tokens.size(-1) > min(self.model.max_positions()):
            raise ValueError('tokens exceeds maximum length: {} > {}'.format(
                tokens.size(-1), self.model.max_positions()
            ))
        tokens.to(device=self.device),
        prev_output_tokens = tokens.clone()

        prev_output_tokens[:, 0] = tokens.gather(
            1,
            (tokens.ne(self.task.source_dictionary.pad()).sum(dim=1)- 1).unsqueeze(-1),
        ).squeeze()

        prev_output_tokens[:, 1:] = tokens[:, :-1]
        features, extra = self.model(
            src_tokens=tokens,
            src_lengths=None,
            prev_output_tokens=prev_output_tokens,
            features_only=True,
            return_all_hiddens=return_all_hiddens,
        )
        if return_all_hiddens:
            # convert from T x B x C -> B x T x C
            inner_states = extra['inner_states']
            return [inner_state.transpose(0, 1) for inner_state in inner_states]
        else:
            return features  # just the last layer's features

    def register_classification_head(
        self, name: str, num_classes: int = None, embedding_size: int = None, **kwargs
    ):
        self.model.register_classification_head(
            name, num_classes=num_classes, embedding_size=embedding_size, **kwargs
        )

    def predict(self, head: str, tokens: torch.LongTensor, return_logits: bool = False):
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)
        features = self.extract_features(tokens.to(device=self.device))
        sentence_representation = features[
            tokens.eq(self.task.source_dictionary.eos()), :
        ].view(features.size(0), -1, features.size(-1))[:, -1, :]

        logits = self.model.classification_heads[head](sentence_representation)
        if return_logits:
            return logits
        return F.log_softmax(logits, dim=-1)
