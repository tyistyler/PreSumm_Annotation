#!/usr/bin/env python
""" Translator Class and builder """
from __future__ import print_function
import codecs
import os
import math

import torch

from tensorboardX import SummaryWriter

from others.utils import rouge_results_to_str, test_rouge, tile
from translate.beam import GNMTGlobalScorer


def build_predictor(args, tokenizer, symbols, model, logger=None):
    scorer = GNMTGlobalScorer(args.alpha, length_penalty='wu')
    translator = Translator(args, model, tokenizer, symbols, global_scorer=scorer, logger=logger)
    return translator


class Translator(object):
    """
    Uses a model to translate a batch of sentences.


    Args:
       model (:obj:`onmt.modules.NMTModel`):
          NMT model to use for translation
       fields (dict of Fields): data fields
       beam_size (int): size of beam to use
       n_best (int): number of translations produced
       max_length (int): maximum length output to produce
       global_scores (:obj:`GlobalScorer`):
         object to rescore final translations
       copy_attn (bool): use copy attention during translation
       cuda (bool): use cuda
       beam_trace (bool): trace beam search for debugging
       logger(logging.Logger): logger.
    """

    def __init__(self,
                 args,
                 model,
                 vocab,
                 symbols,
                 global_scorer=None,
                 logger=None,
                 dump_beam=""):
        self.logger = logger
        self.cuda = args.visible_gpus != '-1'

        self.args = args
        self.model = model
        self.generator = self.model.generator
        self.vocab = vocab
        self.symbols = symbols
        self.start_token = symbols['BOS']
        self.end_token = symbols['EOS']

        self.global_scorer = global_scorer
        self.beam_size = args.beam_size
        self.min_length = args.min_length
        self.max_length = args.max_length

        self.dump_beam = dump_beam

        # for debugging
        self.beam_trace = self.dump_beam != ""
        self.beam_accum = None

        tensorboard_log_dir = args.model_path

        self.tensorboard_writer = SummaryWriter(tensorboard_log_dir, comment="Unmt")

        if self.beam_trace:
            self.beam_accum = {
                "predicted_ids": [],
                "beam_parent_ids": [],
                "scores": [],
                "log_probs": []}

    def _build_target_tokens(self, pred):
        # vocab = self.fields["tgt"].vocab
        tokens = []
        for tok in pred:
            tok = int(tok)
            tokens.append(tok)
            if tokens[-1] == self.end_token:
                tokens = tokens[:-1]
                break
        tokens = [t for t in tokens if t < len(self.vocab)]
        tokens = self.vocab.DecodeIds(tokens).split(' ')
        return tokens

    def from_batch(self, translation_batch):
        batch = translation_batch["batch"]
        assert (len(translation_batch["gold_score"]) ==
                len(translation_batch["predictions"]))
        batch_size = batch.batch_size

        preds, pred_score, gold_score, tgt_str, src =  translation_batch["predictions"],translation_batch["scores"],translation_batch["gold_score"],batch.tgt_str, batch.src

        translations = []
        for b in range(batch_size):
            pred_sents = self.vocab.convert_ids_to_tokens([int(n) for n in preds[b][0]])
            pred_sents = ' '.join(pred_sents).replace(' ##','')
            gold_sent = ' '.join(tgt_str[b].split())
            # translation = Translation(fname[b],src[:, b] if src is not None else None,
            #                           src_raw, pred_sents,
            #                           attn[b], pred_score[b], gold_sent,
            #                           gold_score[b])
            # src = self.spm.DecodeIds([int(t) for t in translation_batch['batch'].src[0][5] if int(t) != len(self.spm)])
            raw_src = [self.vocab.ids_to_tokens[int(t)] for t in src[b]][:500]
            raw_src = ' '.join(raw_src)
            translation = (pred_sents, gold_sent, raw_src)
            # translation = (pred_sents[0], gold_sent)
            translations.append(translation)

        return translations

    def translate(self,
                  data_iter, step,
                  attn_debug=False):

        self.model.eval()
        gold_path = self.args.result_path + '.%d.gold' % step
        can_path = self.args.result_path + '.%d.candidate' % step
        self.gold_out_file = codecs.open(gold_path, 'w', 'utf-8')
        self.can_out_file = codecs.open(can_path, 'w', 'utf-8')

        # raw_gold_path = self.args.result_path + '.%d.raw_gold' % step
        # raw_can_path = self.args.result_path + '.%d.raw_candidate' % step
        # self.gold_out_file = codecs.open(gold_path, 'w', 'utf-8')
        # self.can_out_file = codecs.open(can_path, 'w', 'utf-8')

        raw_src_path = self.args.result_path + '.%d.raw_src' % step
        self.src_out_file = codecs.open(raw_src_path, 'w', 'utf-8')

        # pred_results, gold_results = [], []
        ct = 0
        with torch.no_grad():
            for batch in data_iter:
                # self.args.recall_eval=False
                if(self.args.recall_eval):
                    gold_tgt_len = batch.tgt.size(1)
                    self.min_length = gold_tgt_len + 20
                    self.max_length = gold_tgt_len + 60
                # ???????????????token
                batch_data = self.translate_batch(batch)
                # print("batch_data={}".format(batch_data))
                
                # [(pred_sents, gold_sent, raw_src), (pred_sents, gold_sent, raw_src),...]
                translations = self.from_batch(batch_data)
                for trans in translations:
                    pred, gold, src = trans
                    pred_str = pred.replace('[unused0]', '').replace('[unused3]', '').replace('[PAD]', '').replace('[unused1]', '').replace(r' +', ' ').replace(' [unused2] ', '<q>').replace('[unused2]', '').strip()
                    gold_str = gold.strip()
                    # print("self.args.recall_eval={}".format(self.args.recall_eval))   # False
                    if(self.args.recall_eval):
                        _pred_str = ''
                        gap = 1e3
                        for sent in pred_str.split('<q>'):
                            can_pred_str = _pred_str+ '<q>'+sent.strip()
                            can_gap = math.fabs(len(_pred_str.split())-len(gold_str.split()))
                            # if(can_gap>=gap):
                            if(len(can_pred_str.split())>=len(gold_str.split())+10):
                                pred_str = _pred_str
                                break
                            else:
                                gap = can_gap
                                _pred_str = can_pred_str
                        # pred_str = ' '.join(pred_str.split()[:len(gold_str.split())])
                    # self.raw_can_out_file.write(' '.join(pred).strip() + '\n')
                    # self.raw_gold_out_file.write(' '.join(gold).strip() + '\n')
                    self.can_out_file.write(pred_str + '\n')
                    self.gold_out_file.write(gold_str + '\n')
                    self.src_out_file.write(src.strip() + '\n')
                    ct += 1
                self.can_out_file.flush()
                self.gold_out_file.flush()
                self.src_out_file.flush()

        self.can_out_file.close()
        self.gold_out_file.close()
        self.src_out_file.close()

        if (step != -1):
            rouges = self._report_rouge(gold_path, can_path)
            self.logger.info('Rouges at step %d \n%s' % (step, rouge_results_to_str(rouges)))
            if self.tensorboard_writer is not None:
                self.tensorboard_writer.add_scalar('test/rouge1-F', rouges['rouge_1_f_score'], step)
                self.tensorboard_writer.add_scalar('test/rouge2-F', rouges['rouge_2_f_score'], step)
                self.tensorboard_writer.add_scalar('test/rougeL-F', rouges['rouge_l_f_score'], step)

    def _report_rouge(self, gold_path, can_path):
        self.logger.info("Calculating Rouge")
        results_dict = test_rouge(self.args.temp_dir, can_path, gold_path)
        return results_dict

    def translate_batch(self, batch, fast=False):
        """
        Translate a batch of sentences.

        Mostly a wrapper around :obj:`Beam`.

        Args:
           batch (:obj:`Batch`): a batch from a dataset object
           data (:obj:`Dataset`): the dataset object
           fast (bool): enables fast beam search (may not support all features)

        Todo:
           Shouldn't need the original dataset.
        """
        with torch.no_grad():
            return self._fast_translate_batch(
                batch,
                self.max_length,
                min_length=self.min_length)

    def _fast_translate_batch(self,
                              batch,
                              max_length,
                              min_length=0):
        # TODO: faster code path for beam_size == 1.

        # TODO: support these blacklisted features.
        assert not self.dump_beam

        beam_size = self.beam_size                  # 5
        batch_size = batch.batch_size                 # 6
        src = batch.src                        # [6, 512]
        segs = batch.segs                       # [6, 512]
        mask_src = batch.mask_src                   # [6, 512]
        
        print("src={}".format(src.size()))
        
        src_features = self.model.bert(src, segs, mask_src)
        # ?????????cache
        dec_states = self.model.decoder.init_decoder_state(src, src_features, with_cache=True)
        device = src_features.device

        # Tile states and memory beam_size times.
        # tile->???[6, 512]??????[30, 512]???????????????????????????
        dec_states.map_batch_fn(
            lambda state, dim: tile(state, beam_size, dim=dim))

        src_features = tile(src_features, beam_size, dim=0)    # [30, 512, 768]
        batch_offset = torch.arange(
            batch_size, dtype=torch.long, device=device)     # [0, 1, 2, 3, 4, 5]
        beam_offset = torch.arange(
            0,
            batch_size * beam_size,
            step=beam_size,
            dtype=torch.long,
            device=device)                     # [0, 5, 10, 15, 20, 25]
        # print("self.start_token={}".format(self.start_token))    # 1
        # alive_seq???decoder???????????????
        alive_seq = torch.full(
            [batch_size * beam_size, 1],
            self.start_token,
            dtype=torch.long,
            device=device)                     # [30, 1]
        
        # Give full probability to the first beam on the first step.
        topk_log_probs = (
            torch.tensor([0.0] + [float("-inf")] * (beam_size - 1),
                         device=device).repeat(batch_size)) # [30]
        # print("topk_log_probs={}".format(topk_log_probs.size()))
        # Structure that holds finished hypotheses.
        hypotheses = [[] for _ in range(batch_size)]  # noqa: F812

        results = {}
        results["predictions"] = [[] for _ in range(batch_size)]  # noqa: F812
        results["scores"] = [[] for _ in range(batch_size)]  # noqa: F812
        results["gold_score"] = [0] * batch_size
        results["batch"] = batch
        # print("max_length={}".format(max_length))          # 150
        for step in range(max_length):
            # print("step={}".format(step))
            # alive_seq??????step?????????????????????????????????????????????????????????decoder?????????
            decoder_input = alive_seq[:, -1].view(1, -1)    # [1, 30]
            # Decoder forward.
            decoder_input = decoder_input.transpose(0,1)     # [30, 1]
            # decoder_input????????????token????????????????????????dec_states???????????????token????????????token??????????????????self-attention????????????????????????
            # print(dec_states.cache["layer_0"]["self_keys"].size())  # neural.py???MultiHeadedAttention???dec_states.cache["layer_0"]["self_keys"]????????????
            dec_out, dec_states = self.model.decoder(decoder_input, src_features, dec_states, step=step)
            # print("dec_out={}".format(dec_out.size()))      [30, 1, 768]
            # print(dec_states.cache["layer_0"]["self_keys"].size())
            # Generator forward.
            log_probs = self.generator.forward(dec_out.transpose(0,1).squeeze(0))   # [30, 30522]
            vocab_size = log_probs.size(-1)
            # print("self.end_token={}".format(self.end_token))  # 2
            # print("min_length={}".format(min_length))      # 50
            if step < min_length:
                # <EOS>???????????????????????????
                log_probs[:, self.end_token] = -1e20
            # Multiply probs by the beam probability.
            # print(topk_log_probs.view(-1).unsqueeze(1))      # [30, 1]
            log_probs += topk_log_probs.view(-1).unsqueeze(1)  # ????????????tile??????repeat?????????value, [30, 30522]
            # print("log_probs={}".format(log_probs))

            alpha = self.global_scorer.alpha           #0.95
            length_penalty = ((5.0 + (step + 1)) / 6.0) ** alpha

            # Flatten probs into a list of possibilities.
            curr_scores = log_probs / length_penalty      # [30, 30522]

            if(self.args.block_trigram):              # True
                cur_len = alive_seq.size(1)
                # print("alive_seq={}".format(alive_seq.size()))
                if(cur_len>3):
                    # print("???????????????3???alive_seq={}".format(alive_seq.size()))
                    for i in range(alive_seq.size(0)):
                        fail = False
                        words = [int(w) for w in alive_seq[i]]
                        words = [self.vocab.ids_to_tokens[w] for w in words]
                        words = ' '.join(words).replace(' ##','').split()
                        if(len(words)<=3):
                            continue
                        trigrams = [(words[i-1],words[i],words[i+1]) for i in range(1,len(words)-1)]
                        trigram = tuple(trigrams[-1])
                        if trigram in trigrams[:-1]:
                            fail = True
                        if fail:
                            # ???i???????????????-10e20
                            # ???i=4?????????????????????????????????4???beam???????????????curr_scores.reshape(-1, beam_size * vocab_size)
                            # ??????????????????0???batch??????30522???value???????????????????????????30522*4???value??????????????????top5???index
                            curr_scores[i] = -10e20
            
            curr_scores = curr_scores.reshape(-1, beam_size * vocab_size)   # [6, 152610]
            # print("curr_scores={}".format(curr_scores))
            # print("curr_scores={}".format(curr_scores[1]))
            # ?????????beam_size??????????????????score?????????id
            topk_scores, topk_ids = curr_scores.topk(beam_size, dim=-1)
            # print("topk_scores={}".format(topk_scores.size())) # [6, 5]
            # print("topk_ids={}".format(topk_ids.size()))    # [6, 5]
            # Recover log probs.
            topk_log_probs = topk_scores * length_penalty

            # Resolve beam origin and true word ids.
            # topk_beam_index = topk_ids.div(vocab_size)          # RuntimeError: "index_select_out_cuda_impl" not implemented for 'Float'
            topk_beam_index = topk_ids // vocab_size           # topk_ids???????????????beam
            topk_ids = topk_ids.fmod(vocab_size)              # ??????????????????vocab??????id

            # Map beam_index to batch_index in the flat representation.
            # beam_offset=[0, 5, 10, 15, 20, 25]
            # ?????????topk_id??????????????????beam?????????30??????
            batch_index = (topk_beam_index + beam_offset[:topk_beam_index.size(0)].unsqueeze(1))
            # print("topk_beam_index={}".format(topk_beam_index))
            # print("beam_offset={}".format(beam_offset[:topk_beam_index.size(0)].unsqueeze(1)))
            # print("batch_index={}".format(batch_index))     # [6, 5]
            select_indices = batch_index.view(-1)         # [30]-->????????????????????????indices[0,0,0,0,0,5,5,...,25,25]
            # print("select_indices={}".format(select_indices))
            """
            alive_seq.index_select(0, select_indices)-->[30, 1]
            ?????????alive_seq??????select_indices????????????????????????????????????
            """
            # Append last prediction.--??????beam search
            alive_seq = torch.cat([alive_seq.index_select(0, select_indices), topk_ids.view(-1, 1)], -1)
            # if step==1:
            #   print("alive_seq={}".format(alive_seq))
            #   exit()
            """
            alive_seq-step=0
            tensor([[1, 2745], [1, 2610], [1, 1996], [1, 2417], [1, 2047],
                [1, 1000], [1, 1996], [1, 2047], [1, 3063], [1, 2198],
                [1, 3539], [1, 2585], [1, 7610], [1, 7232], [1, 1996],
                [1, 12642], [1, 1996], [1, 3033], [1, 2047], [1, 14295],
                [1, 2852], [1, 1996], [1, 2070], [1, 7435], [1, 2966],
                [1, 1996], [1, 2009], [1, 2686], [1, 1037], [1, 2034]])
            """
            is_finished = topk_ids.eq(self.end_token)       # [6, 5]
            if step + 1 == max_length:
                is_finished.fill_(1)              #??????????????????True
            # End condition is that the top beam is finished.
            end_condition = is_finished[:, 0].eq(1)
            # Save finished hypotheses.
            # print("is_finished.any()={}".format(is_finished.any()))
            # any()-->????????? False???????????? False????????????????????? True???????????? True???
            if is_finished.any():                 # False
                """
                tensor([[False, False, False, False, False],
                    [False, False, False, False, False],
                    [False, False, False, False, False],
                    [False, False, False, False, False],
                    [False, False, False, False, False],
                    [False, False,  True, False, False]]
                """
                # print("is_finished={}".format(is_finished))
                # print("alive_seq={}".format(alive_seq.size())) # [30, 26]
                predictions = alive_seq.view(-1, beam_size, alive_seq.size(-1)) # [6, 5, 26]
                for i in range(is_finished.size(0)):
                    b = batch_offset[i]         # batch_offset=[0,1,2,3,4,5]
                    if end_condition[i]:
                        """
                        [[False, False, False, False, False],
                        [True, False, False, False, False],
                        [False, False, False, False, False],
                        [False, False,  True, False,  True],
                        [False, False, False, False, False],
                        [False, False,  True, False, False]]
                        """
                        is_finished[i].fill_(1)
                    finished_hyp = is_finished[i].nonzero().view(-1)    # tensor([])
                    # print("is_finished[i].nonzero()={}".format(is_finished[i].nonzero())) # ??????????????????
                    # Store finished hypotheses for this batch.
                    for j in finished_hyp:
                        # print("j={}".format(j))   # j=2, b=5, i=5
                        hypotheses[b].append((
                            topk_scores[i, j],
                            predictions[i, j, 1:])) # predictions??????????????????
                    # If the batch reached the end, save the n_best hypotheses.
                    if end_condition[i]:         # false
                        best_hyp = sorted(
                            hypotheses[b], key=lambda x: x[0], reverse=True)
                        score, pred = best_hyp[0]

                        results["scores"][b].append(score)
                        results["predictions"][b].append(pred)
                # exit()
                non_finished = end_condition.eq(0).nonzero().view(-1)
                # If all sentences are translated, no need to go further.
                if len(non_finished) == 0:
                    break
                # Remove finished batches for the next step.
                # ??????finished????????????????????????????????????????????????????????????????????????(??????)m?????????????????????m???????????????index???????????????
                # ????????????????????????index_select???????????????for????????????b???i???j????????????????????????index???????????????
                topk_log_probs = topk_log_probs.index_select(0, non_finished)
                batch_index = batch_index.index_select(0, non_finished)
                batch_offset = batch_offset.index_select(0, non_finished)
                alive_seq = predictions.index_select(0, non_finished) \
                    .view(-1, alive_seq.size(-1))
            # Reorder states.
            select_indices = batch_index.view(-1)         # topk???index
            # print("select_indices={}".format(select_indices))
            src_features = src_features.index_select(0, select_indices)
            dec_states.map_batch_fn(
                lambda state, dim: state.index_select(dim, select_indices))
            # if step == 3:
            #   exit()
        return results


class Translation(object):
    """
    Container for a translated sentence.

    Attributes:
        src (`LongTensor`): src word ids
        src_raw ([str]): raw src words

        pred_sents ([[str]]): words from the n-best translations
        pred_scores ([[float]]): log-probs of n-best translations
        attns ([`FloatTensor`]) : attention dist for each translation
        gold_sent ([str]): words from gold translation
        gold_score ([float]): log-prob of gold translation

    """

    def __init__(self, fname, src, src_raw, pred_sents,
                 attn, pred_scores, tgt_sent, gold_score):
        self.fname = fname
        self.src = src
        self.src_raw = src_raw
        self.pred_sents = pred_sents
        self.attns = attn
        self.pred_scores = pred_scores
        self.gold_sent = tgt_sent
        self.gold_score = gold_score

    def log(self, sent_number):
        """
        Log translation.
        """

        output = '\nSENT {}: {}\n'.format(sent_number, self.src_raw)

        best_pred = self.pred_sents[0]
        best_score = self.pred_scores[0]
        pred_sent = ' '.join(best_pred)
        output += 'PRED {}: {}\n'.format(sent_number, pred_sent)
        output += "PRED SCORE: {:.4f}\n".format(best_score)

        if self.gold_sent is not None:
            tgt_sent = ' '.join(self.gold_sent)
            output += 'GOLD {}: {}\n'.format(sent_number, tgt_sent)
            output += ("GOLD SCORE: {:.4f}\n".format(self.gold_score))
        if len(self.pred_sents) > 1:
            output += '\nBEST HYP:\n'
            for score, sent in zip(self.pred_scores, self.pred_sents):
                output += "[{:.4f}] {}\n".format(score, sent)

        return output
