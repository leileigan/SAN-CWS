# -*- coding: utf-8 -*-
# @Author: Jie
# @Date:   2017-06-15 14:11:08
# @Last Modified by:   Leilei Gan,     Contact: 11921071@zju.edu.cn
# @Last Modified time: 2019-04-19 22:29:08

import time
import sys
import argparse
import random
import copy
import torch 
import gc
import pickle
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from model.optim import ScheduledOptim
import numpy as np
from utils.metric import get_ner_fmeasure
from model.cws import CWS
from utils.data import Data
from utils import functions
import os
import datetime

seed_num = 10
random.seed(seed_num)
torch.manual_seed(seed_num)
np.random.seed(seed_num)
torch.cuda.manual_seed(seed_num)


def data_initialization(data, train_file, dev_file, test_file):
    print('begin building word alphabet set')
    data.build_alphabet(train_file)
    data.build_alphabet(dev_file)
    data.build_alphabet(test_file)
    data.fix_alphabet()
    print('word alphabet size: %d' % data.word_alphabet_size)
    print('bi word alphabet size: %d' % data.biword_alphabet_size)
    return data


def predict_check(pred_variable, gold_variable, mask_variable):
    """
        input:
            pred_variable (batch_size, sent_len): pred tag result, in numpy format
            gold_variable (batch_size, sent_len): gold result variable
            mask_variable (batch_size, sent_len): mask variable
    """
    pred = pred_variable.cpu().data.numpy()
    gold = gold_variable.cpu().data.numpy()
    mask = mask_variable.cpu().data.numpy()
    overlaped = (pred == gold)
    right_token = np.sum(overlaped * mask)
    total_token = mask.sum()
    # print("right: %s, total: %s"%(right_token, total_token))
    return right_token, total_token


def recover_label(pred_variable, gold_variable, mask_variable, label_alphabet, word_recover):
    """
        input:
            pred_variable (batch_size, sent_len): pred tag result
            gold_variable (batch_size, sent_len): gold result variable
            mask_variable (batch_size, sent_len): mask variable
    """

    pred_variable = pred_variable[word_recover]
    gold_variable = gold_variable[word_recover]
    mask_variable = mask_variable[word_recover]
    batch_size = gold_variable.size(0)
    seq_len = gold_variable.size(1)
    mask = mask_variable.cpu().data.numpy()
    pred_tag = pred_variable.cpu().data.numpy()
    gold_tag = gold_variable.cpu().data.numpy()
    batch_size = mask.shape[0]
    pred_label = []
    gold_label = []
    for idx in range(batch_size):
        pred = [label_alphabet.get_instance(pred_tag[idx][idy]) for idy in range(seq_len) if mask[idx][idy] != 0]
        gold = [label_alphabet.get_instance(gold_tag[idx][idy]) for idy in range(seq_len) if mask[idx][idy] != 0]
        # print "p:",pred, pred_tag.tolist()
        # print "g:", gold, gold_tag.tolist()
        assert (len(pred) == len(gold))
        pred_label.append(pred)
        gold_label.append(gold)
    return pred_label, gold_label


def save_data_setting(data, save_file):
    new_data = copy.deepcopy(data)
    ## remove input instances
    new_data.train_texts = []
    new_data.dev_texts = []
    new_data.test_texts = []
    new_data.raw_texts = []

    new_data.train_Ids = []
    new_data.dev_Ids = []
    new_data.test_Ids = []
    new_data.raw_Ids = []
    ## save data settings
    with open(save_file, 'wb') as fp:
        pickle.dump(new_data, fp)
    print("Data setting saved to file: ", save_file)


def load_data_setting(save_file):
    with open(save_file, 'rb') as fp:
        data = pickle.load(fp)
    print("Data setting loaded from file: ", save_file)
    data.show_data_summary()
    return data


def lr_decay(optimizer, epoch, decay_rate, init_lr):
    lr = init_lr * ((1 - decay_rate) ** epoch)
    print(" Learning rate is setted as:", lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer


def evaluate(data, model, name, external_pos={}):
    if name == "train":
        instances = data.train_Ids
        instance_texts = data.train_texts
    elif name == "dev":
        instances = data.dev_Ids
        instance_texts = data.dev_texts
    elif name == 'test':
        instances = data.test_Ids
        instance_texts = data.test_texts
    elif name == 'raw':
        instances = data.raw_Ids
        instance_texts = data.raw_texts
    else:
        print("Error: wrong evaluate name,", name)
    right_token = 0
    whole_token = 0
    pred_results = []
    gold_results = []
    #  set model in eval model
    model.eval()
    batch_size = 64
    start_time = time.time()
    train_num = len(instances)
    total_batch = train_num // batch_size + 1
    for batch_id in range(total_batch):
        start = batch_id * batch_size
        end = (batch_id + 1) * batch_size
        if end > train_num:
            end = train_num
        instance = instances[start:end]
        instance_text = [sent[0] for sent in instance_texts[start:end]]
        if not instance:
            continue
        batch_word, batch_biword, batch_wordlen, batch_wordrecover, batch_label, mask, rearrange_instance_texts, batch_pos = \
            batchify_with_label(instance_text, instance, data.HP_gpu)
        with torch.no_grad():
            tag_seq = model.forward(rearrange_instance_texts, batch_word, batch_biword, batch_wordlen, mask, batch_pos, external_pos)
            pred_label, gold_label = recover_label(tag_seq, batch_label, mask, data.label_alphabet, batch_wordrecover)
            pred_results += pred_label
            gold_results += gold_label
    decode_time = time.time() - start_time
    speed = len(instances) / decode_time
    acc, p, r, f = get_ner_fmeasure(gold_results, pred_results)
    model.train()
    return speed, acc, p, r, f, pred_results


def batchify_with_label(instance_texts, input_ids_list, gpu):
    rerange_instance_text = []
    batch_pos = []
    batch_size = len(input_ids_list)
    words = [sent[0] for sent in input_ids_list]
    biwords = [sent[1] for sent in input_ids_list]
    labels = [sent[2] for sent in input_ids_list]
    pos_list = [sent[3] for sent in input_ids_list]

    word_seq_lengths = torch.LongTensor([len(item) for item in words])
    max_seq_len = word_seq_lengths.max()
    word_seq_tensor = torch.zeros(batch_size, max_seq_len).long()
    biword_seq_tensor = torch.zeros(batch_size, max_seq_len).long()
    label_seq_tensor = torch.zeros(batch_size, max_seq_len).long()
    mask = torch.zeros(batch_size, max_seq_len).byte()
    for idx, (seq, biseq, label, seqlen) in enumerate(zip(words, biwords, labels, word_seq_lengths)):
        word_seq_tensor[idx, :seqlen] = torch.LongTensor(seq)
        biword_seq_tensor[idx, :seqlen] = torch.LongTensor(biseq)
        label_seq_tensor[idx, :seqlen] = torch.LongTensor(label)
        mask[idx, :seqlen] = torch.Tensor([1] * seqlen.numpy().tolist())
    word_seq_lengths, word_perm_idx = word_seq_lengths.sort(0, descending=True)
    word_seq_tensor = word_seq_tensor[word_perm_idx]
    biword_seq_tensor = biword_seq_tensor[word_perm_idx]
    # not reorder label
    label_seq_tensor = label_seq_tensor[word_perm_idx]
    mask = mask[word_perm_idx]
    _, word_seq_recover = word_perm_idx.sort(0, descending=False)
    [rerange_instance_text.append(instance_texts[item]) for item in word_perm_idx.data.tolist()]
    [batch_pos.append(pos_list[item]) for item in word_perm_idx.data.tolist()]

    if gpu:
        word_seq_tensor = word_seq_tensor.cuda()
        biword_seq_tensor = biword_seq_tensor.cuda()
        word_seq_lengths = word_seq_lengths.cuda()
        label_seq_tensor = label_seq_tensor.cuda()
        mask = mask.cuda()
    return word_seq_tensor, biword_seq_tensor, word_seq_lengths, word_seq_recover, label_seq_tensor, mask, rerange_instance_text, batch_pos


def train(model, data, save_model_dir, seg=True):
    print("Training model...")
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    if data.use_sgd:
        optimizer = optim.SGD(parameters, lr=data.HP_lr, momentum=data.HP_momentum)
    elif data.use_adam:
        optimizer = optim.Adam(parameters, lr=1e-3)
    elif data.use_warmup_adam:
        optimizer = ScheduledOptim(
            optim.Adam(parameters, betas=(0.9, 0.98), eps=1e-9),
            d_model=512,
            n_warmup_steps=1000
        )
    elif data.use_adadelta:
        optimizer = optim.Adadelta(parameters, lr=1e-4, rho=0.95, eps=1e-6)
    elif data.use_bert:
        optimizer = optim.Adam(parameters, lr=5e-6, weight_decay=1e-4)  # fine tuning
    else:
        raise ValueError("Unknown optimizer")

    print('optimizer: ', optimizer)
    best_dev = -1
    # start training
    for idx in range(data.HP_iteration):
        epoch_start = time.time()
        temp_start = epoch_start
        print("Epoch: %s/%s" % (idx, data.HP_iteration))
        if data.use_bert is False and data.use_warmup_adam is False and data.use_adam is False:
            optimizer = lr_decay(optimizer, idx, data.HP_lr_decay, data.HP_lr)
        instance_count = 0
        sample_id = 0
        sample_loss = 0
        batch_loss = 0
        total_loss = 0
        right_token = 0
        whole_token = 0
        data_train = [(ids, text) for ids, text in zip(data.train_Ids, data.train_texts)]
        random.shuffle(data_train)
        #  set model in train model
        model.train()
        model.zero_grad()
        batch_size = data.HP_batch_size
        batch_id = 0
        train_num = len(data.train_Ids)
        total_batch = train_num // batch_size + 1
        for batch_id in range(total_batch):
            start = batch_id * batch_size
            end = (batch_id + 1) * batch_size
            if end > train_num:
                end = train_num
            instance = [item[0] for item in data_train[start:end]]
            instance_text = [item[1][0] for item in data_train[start:end]]
            if not instance:
                continue
            batch_word, batch_biword, batch_wordlen, _, batch_label, mask, rearrange_instance_texts, batch_pos \
                = batchify_with_label(instance_text, instance, data.HP_gpu)
            instance_count += 1
            loss, tag_seq = model.neg_log_likelihood_loss(rearrange_instance_texts, batch_word, batch_biword,
                                                          batch_label, mask, batch_wordlen, batch_pos)
            right, whole = predict_check(tag_seq, batch_label, mask)
            right_token += right
            whole_token += whole
            sample_loss += loss.data
            total_loss += loss.data
            batch_loss += loss

            if end % 500 == 0:
                temp_time = time.time()
                temp_cost = temp_time - temp_start
                temp_start = temp_time
                print("     Instance: %s; Time: %.2fs; loss: %.4f; acc: %s/%s=%.4f" % (
                    end, temp_cost, sample_loss, right_token, whole_token, (right_token + 0.) / whole_token))
                sys.stdout.flush()
                sample_loss = 0

            if end % data.HP_batch_size == 0:
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters, data.HP_clip)
                # if end % 500 == 0:
                #     for name, parameter in model.named_parameters():
                #         print('name:', name)
                #         print('parameter grad: ', parameter.grad)

                if data.use_warmup_adam:
                    optimizer.step_and_update_lr()
                else:
                    optimizer.step()

                model.zero_grad()
                batch_loss = 0
        temp_time = time.time()
        temp_cost = temp_time - temp_start
        print("     Instance: %s; Time: %.2fs; loss: %.4f; acc: %s/%s=%.4f" % (
            end, temp_cost, sample_loss, right_token, whole_token, (right_token + 0.) / whole_token))
        epoch_finish = time.time()
        epoch_cost = epoch_finish - epoch_start
        print("Epoch: %s training finished. Time: %.2fs, speed: %.2fst/s,  total loss: %s" % (idx, epoch_cost, train_num / epoch_cost, total_loss))
        # exit(0)
        # continue
        speed, acc, p, r, f, _ = evaluate(data, model, "dev")
        dev_finish = time.time()
        dev_cost = dev_finish - epoch_finish

        if seg:
            current_score = f
            print("Dev: time: %.2fs, speed: %.2fst/s; acc: %.4f, p: %.4f, r: %.4f, f: %.4f" % (dev_cost, speed, acc, p, r, f))
        else:
            current_score = acc
            print("Dev: time: %.2fs speed: %.2fst/s; acc: %.4f" % (dev_cost, speed, acc))

        if current_score > best_dev:
            if seg:
                print("Exceed previous best f score:", best_dev)
            else:
                print("Exceed previous best acc score:", best_dev)
            best_dev = current_score

        model_name = save_model_dir + '.' + str(idx) + ".model"
        torch.save(model.state_dict(), model_name)
        # ## decode test
        speed, acc, p, r, f, _ = evaluate(data, model, "test")
        test_finish = time.time()
        test_cost = test_finish - dev_finish
        if seg:
            print("Test: time: %.2fs, speed: %.2fst/s; acc: %.4f, p: %.4f, r: %.4f, f: %.4f" % (
            test_cost, speed, acc, p, r, f))
        else:
            print("Test: time: %.2fs, speed: %.2fst/s; acc: %.4f" % (test_cost, speed, acc))

        if data.cross_domain or data.cross_test:
            for domain in ['zx', 'fr', 'dl']:
                cross_file = '../SubWordCWS/data/cross-domain/bmes/' + domain + '.bmes'
                dic_file = '../SubWordCWS/data/cross-domain/dic/' + domain + '_dict'
                external_pos_list = [functions.load_external_pos(dic_file), {}]
                for external_pos in external_pos_list:
                    print('evaluate %s: ' % cross_file)
                    print('external pos size: ', len(external_pos))
                    data.generate_instance(cross_file, "raw")
                    start_time = time.time()
                    speed, acc, p, r, f, pre_results = evaluate(data, model, 'raw', external_pos)
                    end_time = time.time()
                    time_cost = end_time - start_time
                    if seg:
                        print("%s: time:%.2fs, speed:%.2fst/s; acc: %.4f, p: %.4f, r: %.4f, f: %.4f" % (
                            'raw', time_cost, speed, acc, p, r, f))
                    else:
                        print("%s: time:%.2fs, speed:%.2fst/s; acc: %.4f" % ('raw', time_cost, speed, acc))

        gc.collect()


def load_model_decode(model_dir, data, name, gpu, seg=True, external_pos={}):
    data.HP_gpu = gpu
    print("Load Model from file: ", model_dir)
    model = CWS(data)

    model.load_state_dict(torch.load(model_dir))

    print("Decode %s data ..." % (name))
    start_time = time.time()
    speed, acc, p, r, f, pred_results = evaluate(data, model, name, external_pos)
    end_time = time.time()
    time_cost = end_time - start_time
    if seg:
        print("%s: time:%.2fs, speed:%.2fst/s; acc: %.4f, p: %.4f, r: %.4f, f: %.4f" % (
        name, time_cost, speed, acc, p, r, f))
    else:
        print("%s: time:%.2fs, speed:%.2fst/s; acc: %.4f" % (name, time_cost, speed, acc))
    return pred_results


if __name__ == '__main__':
    print(datetime.datetime.now())
    parser = argparse.ArgumentParser(description='Investigate Self Attention Network for Chinese Word Segmentation')
    parser.add_argument('--embedding', help='Embedding for words', default='None')
    parser.add_argument('--char_embedding', help='Embedding for chars', default='None')
    parser.add_argument('--bichar_embedding', help='Embedding for bi-chars', default='None')
    parser.add_argument('--status', choices=['train', 'test', 'decode'], help='update algorithm', default='train')
    parser.add_argument('--savemodel', default="data/model/saved_model")
    parser.add_argument('--savedset', help='Dir of saved data setting', default="data/save.dset")
    parser.add_argument('--source', default="../SubWordCWS/data/cross-domain/pd/pku.pos.pre.bmes")
    parser.add_argument('--train', default="data/ctb6.0/origin/train.ctb60.char.bmes")
    parser.add_argument('--dev', default="data/ctb6.0/origin/dev.ctb60.char.bmes")
    parser.add_argument('--test', default="data/ctb6.0/origin/test.ctb60.char.bmes")
    parser.add_argument('--seg', default="True")
    parser.add_argument('--raw')
    parser.add_argument('--loadmodel')
    parser.add_argument('--output')
    parser.add_argument('--use_attention', default="False")
    parser.add_argument('--cross_domain', default="False")
    parser.add_argument('--external_pos', default=None)
    parser.add_argument('--token_replace_prob', default="data/pd.prob")
    parser.add_argument('--pos_to_idx', default="data/pos_count")
    parser.add_argument('--use_cnn', type=bool, default=False)
    parser.add_argument('--use_san', default="False")
    parser.add_argument('--use_adam', default="False")
    parser.add_argument('--cross_test', default="False")
    parser.add_argument('--use_bert', default="False")
    parser.add_argument('--use_crf', default="False")
    parser.add_argument('--use_warmup_adam', default="False")
    parser.add_argument('--use_sgd', default="False")
    parser.add_argument('--use_adadelta', type=bool, default=False)
    parser.add_argument('--use_window', default="True")
    parser.add_argument('--use_tencent_dic', default="True")
    parser.add_argument('--dropout', type=float)

    args = parser.parse_args()

    train_file = args.train
    dev_file = args.dev
    test_file = args.test
    raw_file = args.raw
    model_dir = args.loadmodel
    dset_dir = args.savedset
    output_file = args.output

    # attention
    use_attention = True if args.use_attention.lower() == 'true' else False
    cross_domain = True if args.cross_domain.lower() == 'true' else False
    if cross_domain:
        train_file = args.source

    cross_test = True if args.cross_test.lower() == 'true' else False
    use_san = True if args.use_san.lower() == 'true' else False
    use_adam = True if args.use_adam.lower() == 'true' else False
    use_bert = True if args.use_bert.lower() == 'true' else False
    use_warmup_adam = True if args.use_warmup_adam.lower() == 'true' else False
    use_sgd = True if args.use_sgd.lower() == 'true' else False
    external_pos_path = args.external_pos
    token_replace_pro_path = args.token_replace_prob
    pos_to_idx_path = args.pos_to_idx

    status = args.status.lower()

    if args.seg.lower() == "true":
        seg = True
    else:
        seg = False

    save_model_dir = args.savemodel
    gpu = torch.cuda.is_available()
    char_emb = args.char_embedding
    bichar_emb = args.bichar_embedding

    print("CuDNN:", torch.backends.cudnn.enabled)
    # gpu = False
    print("GPU available:", gpu)
    print("Status:", status)
    print("Seg: ", seg)
    print("Train file:", train_file)
    print("Dev file:", dev_file)
    print("Test file:", test_file)
    print("Raw file:", raw_file)
    print("Char emb:", char_emb)
    print("Bichar emb:", bichar_emb)
    if status == 'train':
        print("Model saved to:", save_model_dir)
    sys.stdout.flush()

    if status == 'train':
        print('train model')
        if model_dir is not None and dset_dir is not None and \
                os.path.isfile(model_dir) and os.path.exists(dset_dir):
            print('load model from: %s' % model_dir)
            data = load_data_setting(dset_dir)
            data_initialization(data, train_file, dev_file, test_file)

            data.generate_instance(train_file, 'train')
            data.generate_instance(dev_file, 'dev')
            data.generate_instance(test_file, 'test')
            data.HP_gpu = gpu
            model = CWS(data)
            model.load_state_dict(torch.load(model_dir))
            train(model, data, save_model_dir, seg)
        else:
            print('new train parameter')
            data = Data()
            data.HP_gpu = gpu
            data.HP_batch_size = 1
            data.use_bigram = True
            data.HP_lr = 1e-2
            data.HP_dropout = 0.4
            data.HP_iteration = 100
            data_initialization(data, train_file, dev_file, test_file)

            data.generate_instance(train_file, 'train')
            data.generate_instance(dev_file, 'dev')
            data.generate_instance(test_file, 'test')
            data.build_word_pretrain_emb(char_emb)
            data.build_biword_pretrain_emb(bichar_emb)
            # data.build_word_vec_100()
            # attention
            data.cross_domain = cross_domain
            data.cross_test = cross_test
            data.use_attention = use_attention
            data.use_san = use_san
            data.use_cnn = args.use_cnn
            data.use_adam = use_adam
            data.use_bert = use_bert
            data.use_warmup_adam = use_warmup_adam
            data.use_adadelta = args.use_adadelta
            data.use_sgd = use_sgd
            data.use_window = True if args.use_window.lower() == 'true' else False
            data.use_crf = True if args.use_crf.lower() == 'true' else False
            data.use_tencent_dic = True if args.use_tencent_dic.lower() == 'true' else False

            if cross_domain:
                data.pos_to_idx = functions.load_pos_to_idx(args.pos_to_idx)
                data.token_replace_prob = functions.load_token_pos_prob(args.token_replace_prob)

            save_data_name = save_model_dir + ".dset"
            save_data_setting(data, save_data_name)
            data.show_data_summary()
            model = CWS(data)
            train(model, data, save_model_dir, seg)

    elif status == 'test':
        if os.path.exists(model_dir) is False:
            print('model does not exit: ', model_dir)
        else:
            data = load_data_setting(dset_dir)

            if cross_domain and external_pos_path is not None:
                data.external_pos = functions.load_external_pos(external_pos_path)

            data.generate_instance(dev_file, 'test')
            load_model_decode(model_dir, data, 'test', gpu, seg, data.external_pos)

    elif status == 'decode':
        if os.path.exists(model_dir) is False:
            print('model does not exit: ', model_dir)
        else:
            data = load_data_setting(dset_dir)
            if cross_domain or external_pos_path is not None:
                data.external_pos = functions.load_external_pos(external_pos_path)
                print('data external pos path: ', external_pos_path)
            data.generate_instance(raw_file, 'raw')
            decode_results = load_model_decode(model_dir, data, 'raw', gpu, seg, data.external_pos)
            data.write_decoded_results(output_file, decode_results, 'raw')
    else:
        print("Invalid argument! Please use valid arguments! (train/test/decode)")
