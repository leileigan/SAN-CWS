# -*- coding: utf-8 -*-
import sys
import numpy as np
from utils.alphabet import Alphabet
from gensim.models import KeyedVectors
import codecs
import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM

NULLKEY = "-null-"
BASE_PATH = '/home/ganleilei/data/TencentDic/'
WORD_VEC_MODEL_PATH = 'Tencent_AILab_ChineseEmbedding_general.txt.kv'
DIC_PATH = BASE_PATH + WORD_VEC_MODEL_PATH


def normalize_word(word):
    new_word = ""
    for char in word:
        if char.isdigit():
            new_word += '0'
        else:
            new_word += char
    return new_word


def load_tencent_dic():
    print("loading tencent dictionary...")
    word2vec = KeyedVectors.load(DIC_PATH, mmap='r')
    dic = set(word2vec.vocab.keys())
    print('finish loading tencent dictionary, dic size: ', len(dic))
    return word2vec, dic


def load_pos_to_idx(filename):
    print('loading pos to idx, path: ', filename)
    pos_to_idx = {}
    for line in codecs.open(filename, 'r', 'utf-8'):
        parts = line.strip().split('###')
        if len(parts) != 4:
            continue
        if parts[3] == 'close':
            continue
        if parts[1].lower() not in pos_to_idx:
            pos_to_idx[parts[1].lower()] = len(pos_to_idx)
    print('pos to idx: ', pos_to_idx)
    return pos_to_idx


def load_external_pos(path):
    res = {}
    print('loading external pos from: ', path)
    for line in codecs.open(path, 'r', 'utf-8'):
        parts = line.strip().split('#')
        if len(parts) != 2:
            continue
        token = parts[0]
        pos = parts[1].lower()
        res[token] = pos
    print('external pos size: ', len(res))
    return res


def load_token_pos_prob(path):
    print('loading token to pos prob')
    res = {}
    for line in codecs.open(path, 'r', 'utf-8'):
        parts = line.strip().split('\t')
        if len(parts) != 3:
            continue
        token_pos = parts[0]
        prob = parts[2]
        res[token_pos] = float(prob) / 2
    print('finish loading token pos prob, len of dic: ', len(res))
    return res


def read_seg_instance(input_file, word_alphabet, biword_alphabet, char_alphabet, label_alphabet, number_normalized, max_sent_length,
                      char_padding_size=-1, char_padding_symbol = '</pad>'):
    in_lines = open(input_file,'r').readlines()
    instence_texts = []
    instence_Ids = []
    words = []
    biwords = []
    chars = []
    labels = []
    word_Ids = []
    biword_Ids = []
    char_Ids = []
    label_Ids = []
    for idx in range(len(in_lines)):
        line = in_lines[idx]
        if len(line) > 2:
            pairs = line.strip().split()
            word = pairs[0]
            if number_normalized:
                word = normalize_word(word)
            label = pairs[-1]
            words.append(word)
            if idx < len(in_lines) -1 and len(in_lines[idx+1]) > 2:
                biword = word + in_lines[idx+1].strip().split()[0]
            else:
                biword = word + NULLKEY
            biwords.append(biword)
            labels.append(label)
            word_Ids.append(word_alphabet.get_index(word))
            biword_Ids.append(biword_alphabet.get_index(biword))
            label_Ids.append(label_alphabet.get_index(label))
            char_list = []
            char_Id = []
            for char in word:
                char_list.append(char)
            if char_padding_size > 0:
                char_number = len(char_list)
                if char_number < char_padding_size:
                    char_list = char_list + [char_padding_symbol]*(char_padding_size-char_number)
                assert(len(char_list) == char_padding_size)
            else:
                ### not padding
                pass
            for char in char_list:
                char_Id.append(char_alphabet.get_index(char))
            chars.append(char_list)
            char_Ids.append(char_Id)
        else:
            if (max_sent_length < 0) or (len(words) < max_sent_length):
                instence_texts.append([words, biwords, chars, labels])
                instence_Ids.append([word_Ids, biword_Ids, char_Ids,label_Ids])
            words = []
            biwords = []
            chars = []
            labels = []
            word_Ids = []
            biword_Ids = []
            char_Ids = []
            label_Ids = []
    print('instance texts: ', instence_texts[0])
    print('instance ids: ', instence_texts[0])
    return instence_texts, instence_Ids


def read_instance(input_file, word_alphabet, biword_alphabet, label_alphabet, number_normalized, max_sent_length):
    in_lines = open(input_file,'r').readlines()
    instance_texts = []
    instance_ids = []
    words = []
    biwords = []
    sentence_pos = {}
    labels = []
    word_Ids = []
    biword_Ids = []
    label_Ids = []
    start_idx, end_idx = 0, 0
    position_idx = 0
    for idx in range(len(in_lines)):
        line = in_lines[idx]
        if len(line.strip()) > 0:
            pairs = line.strip().split('\t')
            word = pairs[0]
            #  add pos
            pos = ''
            if len(pairs[-1].split('-')) == 2:
                pos = pairs[-1].split('-')[-1].lower()

            if number_normalized:
                word = normalize_word(word)

            if pairs[-1][0].lower() == 's':
                start_idx = position_idx
                end_idx = position_idx
                if len(pos) > 0:
                    sentence_pos[(start_idx, end_idx)] = pos
                start_idx, end_idx = 0, 0
            elif pairs[-1][0].lower() == 'b':
                start_idx = position_idx
            elif pairs[-1][0].lower() == 'e':
                end_idx = position_idx
                if len(pos) > 0:
                    sentence_pos[(start_idx, end_idx)] = pos
                start_idx, end_idx = 0, 0

            label = pairs[-1][0] + '-SEG'
            if idx < len(in_lines) -1 and len(in_lines[idx+1]) > 2:
                biword = word + in_lines[idx+1].strip().split()[0]
            else:
                biword = word + NULLKEY
            biwords.append(biword)
            words.append(word)
            labels.append(label)
            word_Ids.append(word_alphabet.get_index(word))
            biword_Ids.append(biword_alphabet.get_index(biword))
            label_Ids.append(label_alphabet.get_index(label))

            position_idx += 1

        else:

            if ((max_sent_length < 0) or (len(words) < max_sent_length)) and (len(words)>0):
                instance_texts.append([words, biwords, labels, sentence_pos])
                instance_ids.append([word_Ids, biword_Ids, label_Ids, sentence_pos])

            words = []
            biwords = []
            sentence_pos = {}
            labels = []
            word_Ids = []
            biword_Ids = []
            label_Ids = []
            start_idx, end_idx = 0, 0
            position_idx = 0

    return instance_texts, instance_ids


def build_pretrain_embedding(embedding_path, word_alphabet, embedd_dim=100, norm=True):    
    embedd_dict = dict()
    if embedding_path != None:
        embedd_dict, embedd_dim = load_pretrain_emb(embedding_path)
    scale = np.sqrt(3.0 / embedd_dim)
    pretrain_emb = np.empty([word_alphabet.size(), embedd_dim])
    perfect_match = 0
    case_match = 0
    not_match = 0
    for word, index in word_alphabet.iteritems():
        if word in embedd_dict:
            if norm:
                pretrain_emb[index,:] = norm2one(embedd_dict[word])
            else:
                pretrain_emb[index,:] = embedd_dict[word]
            perfect_match += 1
        elif word.lower() in embedd_dict:
            if norm:
                pretrain_emb[index,:] = norm2one(embedd_dict[word.lower()])
            else:
                pretrain_emb[index,:] = embedd_dict[word.lower()]
            case_match += 1
        else:
            pretrain_emb[index,:] = np.random.uniform(-scale, scale, [1, embedd_dim])
            not_match += 1
    pretrained_size = len(embedd_dict)
    print("Embedding:\n     pretrain word:%s, prefect match:%s, case_match:%s, oov:%s, oov%%:%s"%(pretrained_size, perfect_match, case_match, not_match, (not_match+0.)/word_alphabet.size()))
    return pretrain_emb, embedd_dim


def norm2one(vec):
    root_sum_square = np.sqrt(np.sum(np.square(vec)))
    return vec/root_sum_square


def load_pretrain_emb(embedding_path):
    embedd_dim = -1
    embedd_dict = dict()
    with open(embedding_path, 'r') as file:
        for line in file:
            line = line.strip()
            if len(line) == 0:
                continue
            tokens = line.split()
            if embedd_dim < 0:
                embedd_dim = len(tokens) - 1
            else:
                assert (embedd_dim + 1 == len(tokens))
            embedd = np.empty([1, embedd_dim])
            embedd[:] = tokens[1:]
            embedd_dict[tokens[0]] = embedd
    return embedd_dict, embedd_dim


if __name__ == '__main__':
    word2vec, dic = load_tencent_dic()
    a = torch.Tensor().cuda()
    b = torch.cat((a, torch.from_numpy(word2vec['太阳']).cuda().unsqueeze(0)))
    print(b.shape)
    b = torch.cat((b, b), 0)
    print(b.shape)
