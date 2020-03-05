# -*- coding: utf-8 -*-
# @Author: gump
# @Date:   2018-12-15 14:11:08
# @Contact: ganleilei@westlake.edu.cn

import torch
import torch.nn as nn
import torch.autograd as autograd
import numpy as np
import random
import torch.nn.functional as F
from utils import functions

CUDA = torch.cuda.is_available()
seed_num = 10
torch.cuda.manual_seed(seed_num)
torch.manual_seed(seed_num)
torch.cuda.manual_seed_all(seed_num)
np.random.seed(seed_num)
random.seed(seed_num)


class Attention(nn.Module):
    def __init__(self, data, input_size, output_size=200, attention_type='general'):
        """
        character hidden attention with candidate words vectors

        :param attention_type: dot, general, concat
        """
        super(Attention, self).__init__()
        if attention_type not in ['dot', 'general', 'concat']:
            raise ValueError('invalid attention type: %s' % attention_type)

        self.window_size = 4
        self.input_size = input_size
        self.output_size = output_size

        # for cross domain
        self.cross_domain = data.cross_domain
        self.pos_to_idx = data.pos_to_idx
        self.use_tencent_dic = data.use_tencent_dic if hasattr(data, "use_tencent_dic") else False

        if self.use_tencent_dic:
            self.tencent_vec, self.tencent_dic = functions.load_tencent_dic()

        self.pos_bem_to_idx = {}
        for pos in self.pos_to_idx:
            self.pos_bem_to_idx[pos + '_b'] = len(self.pos_bem_to_idx)
            self.pos_bem_to_idx[pos + '_m'] = len(self.pos_bem_to_idx)
            self.pos_bem_to_idx[pos + '_e'] = len(self.pos_bem_to_idx)
        print('pos bem to idx: ', self.pos_bem_to_idx)

        if hasattr(data, "pos_embed_dim"):
            self.pos_embed = nn.Embedding(len(self.pos_bem_to_idx), data.pos_embed_dim)

        self.token_replace_prob = data.token_replace_prob
        if len(self.token_replace_prob) > 0:
            print('token replace prob: ', self.token_replace_prob['家庭###nn'])

        self.attention_type = attention_type
        if self.attention_type == 'general':
            self.linear_W = nn.Linear(self.input_size, self.output_size, bias=False)

        elif self.attention_type == 'concat':
            self.linear_W = nn.Linear(self.input_size, self.output_size, bias=False)
            self.v_a = torch.randon(1, self.output_size)

    def match_all_candidate_words(self, cur_sentence, cur_sentence_pos, external_pos):
        # find all candidate words in sentence
        all_candidate_words = dict()
        cur_sentence_len = len(cur_sentence)

        for char_j in range(cur_sentence_len):
            for k in range(1, self.window_size):
                if char_j + k + 1 > cur_sentence_len:
                    break

                candidate_word = cur_sentence[char_j:char_j + k + 1]
                candidate_pos = ''
                candidate_word_pos = ''

                if (char_j, char_j + k) in cur_sentence_pos:
                    candidate_pos = cur_sentence_pos[(char_j, char_j + k)]
                    candidate_word_pos = candidate_word + '###' + candidate_pos

                '''
                if not self.cross_domain:
                    if self.use_tencent_dic and candidate_word in self.tencent_dic:
                        all_candidate_words[(char_j, char_j + k)] = candidate_word
                else:
                '''

                if len(external_pos) > 0:
                    # when predicting, first search word if in tencent lexicon
                    if self.use_tencent_dic and candidate_word in self.tencent_dic:
                        all_candidate_words[(char_j, char_j + k)] = candidate_word
                    elif candidate_word in external_pos and external_pos[candidate_word] in self.pos_to_idx:
                        all_candidate_words[(char_j, char_j + k)] = external_pos[candidate_word]

                else:
                    # when training and in cross domain mode, replace word with pos tag randomljy
                    if candidate_pos in self.pos_to_idx and candidate_word_pos in self.token_replace_prob \
                            and random.random() < self.token_replace_prob[candidate_word_pos]:
                        all_candidate_words[(char_j, char_j + k)] = candidate_pos
                    elif self.use_tencent_dic and candidate_word in self.tencent_dic:
                        all_candidate_words[(char_j, char_j + k)] = candidate_word

        return all_candidate_words

    def forward(self, input_context, batch_x, word_text, word_seq_len, batch_pos, external_pos):
        """
        enhance character embedding with related words vector
        """
        local_batch_vec = torch.Tensor().cuda()
        batch_size = batch_x.size(0)
        max_len = word_seq_len.max().tolist()
        assert (batch_size == len(word_seq_len.data.tolist()))
        # print("input context size:", input_context.size())
        for batch_i in range(batch_size):
            #  当前句子长度
            cur_sentence = ''.join(word_text[batch_i])
            cur_sentence_pos = batch_pos[batch_i]
            cur_sentence_len = len(cur_sentence)
            assert (cur_sentence_len == word_seq_len.data.tolist()[batch_i])
            # iterate through batch num
            local_sentence_vec = torch.Tensor().cuda()
            # print('cur sentence: ', cur_sentence)
            # print('cur sentence pos:', cur_sentence_pos)

            all_candidate_words = self.match_all_candidate_words(cur_sentence, cur_sentence_pos, external_pos)
            # print("all candidate words:", all_candidate_words)
            for char_j, cur_char in enumerate(cur_sentence):
                # iterate through each sentence
                cur_char_embedding = input_context[batch_i][char_j]

                local_candidate_words_vec = torch.Tensor().cuda()
                # print('cur char: %s ' % cur_char)

                for k, v in all_candidate_words.items():
                    start, end = k[0], k[1]
                    if char_j < start or char_j > end:
                        continue

                    if v in self.pos_to_idx:

                        if char_j == start:
                            pos_embed_idx = LongTensor([self.pos_bem_to_idx[v + '_b']])
                        elif char_j == end:
                            pos_embed_idx = LongTensor([self.pos_bem_to_idx[v + '_e']])
                        else:
                            pos_embed_idx = LongTensor([self.pos_bem_to_idx[v + '_m']])

                        candidate_pos_vec = self.pos_embed(LongTensor([pos_embed_idx]))
                        local_candidate_words_vec = torch.cat((local_candidate_words_vec, candidate_pos_vec))

                    elif v in self.tencent_dic:
                        candidate_word_vec = torch.from_numpy(self.tencent_vec[v]).cuda().unsqueeze(0)
                        local_candidate_words_vec = torch.cat((local_candidate_words_vec, candidate_word_vec))

                if local_candidate_words_vec.size(0) == 0:
                    local_candidate_words_vec = zeros(1, self.output_size)

                if self.attention_type == 'dot':
                    local_scores = local_candidate_words_vec.matmul(cur_char_embedding)

                elif self.attention_type == 'general':
                    linear_out = self.linear_W(cur_char_embedding)
                    local_scores = local_candidate_words_vec.matmul(linear_out)

                else:  # concat
                    concat_tensor = torch.cat((local_candidate_words_vec,
                                               cur_char_embedding * local_candidate_words_vec.size(0)))
                    linear_out = self.linear_W(concat_tensor)
                    tanh_out = F.tanh(linear_out)
                    local_scores = self.v_a.matmul(tanh_out)

                # print('local scores: ', local_scores)
                local_softmax_scores = F.softmax(local_scores, 0)
                # print('local softrmax scores: ', local_softmax_scores)
                local_context = local_softmax_scores.matmul(local_candidate_words_vec)
                local_sentence_vec = torch.cat((local_sentence_vec, local_context.unsqueeze(0)))
            #  add padding zero vector
            while local_sentence_vec.size(0) < max_len:
                local_sentence_vec = torch.cat((local_sentence_vec, zeros(1, self.output_size)))

            local_batch_vec = torch.cat((local_batch_vec, local_sentence_vec.unsqueeze(0)))

        return local_batch_vec


def Tensor(*args):
    x = torch.Tensor(*args)
    return x.cuda() if CUDA else x


def LongTensor(*args):
    x = torch.LongTensor(*args)
    return x.cuda() if CUDA else x


def randn(*args):
    x = torch.randn(*args)
    return x.cuda() if CUDA else x


def zeros(*args):
    x = torch.zeros(*args)
    return x.cuda() if CUDA else x
