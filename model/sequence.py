# -*- coding: utf-8 -*-
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.word_attention import Attention
from utils import functions
import model.transformer as transformer
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
from model.cnn import CNNModel

seed_num = 10
torch.manual_seed(seed_num)
np.random.seed(seed_num)
torch.cuda.manual_seed(seed_num)


BERT_TOKEN_PATH = '../data/bert/bert-base-chinese-vocab.txt'
BERT_MODEL_PATH = '../data/bert/'


class Seq(nn.Module):
    def __init__(self, data):
        super(Seq, self).__init__()
        print("build batched vallina bilstm...")
        self.use_bigram = data.use_bigram
        self.use_attention = data.use_attention
        self.use_transformer = data.use_san
        self.gpu = data.HP_gpu
        self.batch_size = data.HP_batch_size
        self.char_hidden_dim = 0
        self.embedding_dim = data.word_emb_dim
        self.hidden_dim = data.HP_hidden_dim
        self.drop = nn.Dropout(data.HP_dropout)
        self.droplstm = nn.Dropout(data.HP_lstmdropout)
        self.char_embeddings = nn.Embedding(data.word_alphabet.size(), self.embedding_dim)
        self.bichar_embeddings = nn.Embedding(data.biword_alphabet.size(), data.biword_emb_dim)
        self.bilstm_flag = data.HP_bilstm
        self.word_alphabet = data.word_alphabet
        # self.bilstm_flag = False
        self.lstm_layer = data.HP_lstm_layer
        self.use_bert = data.use_bert
        self.use_crf = data.use_crf
        self.use_cnn = data.use_cnn

        if data.pretrain_word_embedding is not None:
            print('loading pretrain word embedding...')
            self.char_embeddings.weight.data.copy_(torch.from_numpy(data.pretrain_word_embedding))
        else:
            self.char_embeddings.weight.data.copy_(torch.from_numpy(self.random_embedding(data.word_alphabet.size(), self.embedding_dim)))

        if data.pretrain_biword_embedding is not None:
            print('loading pretrain bi word embedding...')
            self.bichar_embeddings.weight.data.copy_(torch.from_numpy(data.pretrain_biword_embedding))
        else:
            self.bichar_embeddings.weight.data.copy_(torch.from_numpy(self.random_embedding(data.biword_alphabet.size(), data.biword_emb_dim)))
        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.

        if self.bilstm_flag:
            hidden_dim = data.HP_hidden_dim // 2
        else:
            hidden_dim = data.HP_hidden_dim
        input_dim = 768 if self.use_bert else self.embedding_dim + self.char_hidden_dim
        if self.use_bigram:
            input_dim += data.biword_emb_dim

        if self.use_attention:
            input_dim += data.tencent_word_embed_dim
            self.char_size = 768 if self.use_bert else 50
            self.attention = Attention(data, input_size=self.char_size, attention_type='general')

        if self.use_transformer:
            self.transformer = transformer.TransformerEncoder(data, context_size=input_dim)
            self.hidden2tag = nn.Linear(512, data.label_alphabet_size)

        elif self.use_cnn:
            self.cnn = CNNModel(input_dim, hidden_dim, 3, 0.2)
            self.hidden2tag = nn.Linear(hidden_dim, data.label_alphabet_size)

        elif self.use_crf:
            self.hidden2tag = nn.Linear(input_dim, data.label_alphabet_size)

        else:
            self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=self.lstm_layer, batch_first=True, bidirectional=self.bilstm_flag)
            # The linear layer that maps from hidden state space to tag space
            self.hidden2tag = nn.Linear(data.HP_hidden_dim, data.label_alphabet_size)

        if self.use_bert:
            self.tokenizer = BertTokenizer.from_pretrained(BERT_TOKEN_PATH)
            self.bert_model = BertModel.from_pretrained(BERT_MODEL_PATH)
            bert_params = list(self.bert_model.named_parameters())
            self.bert_model.eval()
            self.bert_model.cuda()
            for p in bert_params:
                p[1].requires_grad = False

        if self.gpu:
            print('begin copying data to gpu')
            self.drop = self.drop.cuda()
            self.droplstm = self.droplstm.cuda()
            self.char_embeddings = self.char_embeddings.cuda()
            self.bichar_embeddings = self.bichar_embeddings.cuda()
            if self.use_attention:
                self.attention = self.attention.cuda()

            if self.use_transformer:
                self.transformer = self.transformer.cuda()

            elif self.use_cnn:
                self.cnn = self.cnn.cuda()

            elif self.use_crf is False:
                self.lstm = self.lstm.cuda()

            self.hidden2tag = self.hidden2tag.cuda()
            print('finish copy')

    def random_embedding(self, vocab_size, embedding_dim):
        pretrain_emb = np.empty([vocab_size, embedding_dim])
        scale = np.sqrt(3.0 / embedding_dim)
        for index in range(vocab_size):
            pretrain_emb[index, :] = np.random.uniform(-scale, scale, [1, embedding_dim])
        return pretrain_emb

    def load_bert_char_embedding(self, word_text, word_seq_lens):
        max_len = max(word_seq_lens.data.tolist())
        text = [' '.join(['[CLS]'] + item + ['[SEP]'] + (max_len - len(item)) * ['[PAD]']) for item in word_text]
        tokenized_text = [self.tokenizer.tokenize(item) for item in text]
        indexed_token_ids = [self.tokenizer.convert_tokens_to_ids(item) for item in tokenized_text]
        segment_ids = [[0] * len(item) for item in indexed_token_ids]
        tokens_tensor = torch.tensor(indexed_token_ids).cuda()
        segment_tensor = torch.tensor(segment_ids).cuda()
        # with torch.no_grad():
        encoded_layers, _ = self.bert_model(tokens_tensor, segment_tensor)
        word_embs = encoded_layers[-1]  # 1 x seq_len x 768
        return word_embs[:, 1:-1, :]

    def get_lstm_features(self, mask, word_text, word_inputs, biword_inputs, word_seq_lens, batch_pos, external_pos):

        char_embs = self.char_embeddings(word_inputs)
        if self.use_bert:
            char_embs = self.load_bert_char_embedding(word_text, word_seq_lens)

        word_embs = char_embs

        if self.use_bigram:
            bichar_embs = self.bichar_embeddings(biword_inputs)
            word_embs = torch.cat((word_embs, bichar_embs), 2)

        if self.use_crf:
            return self.drop(word_embs)

        if self.use_attention:
            att_context = self.attention.forward(char_embs, word_inputs, word_text, word_seq_lens, batch_pos, external_pos)
            word_embs = torch.cat((word_embs, att_context), 2)

        if self.use_transformer:
            lstm_out = self.transformer.forward(word_embs, word_inputs)

        elif self.use_cnn:
            word_embs = self.drop(word_embs)
            cnn_out = self.cnn.forward(word_embs)
            return cnn_out

        else:
            word_embs = self.drop(word_embs)
            # packed_words = pack_padded_sequence(word_embs, word_seq_lengths.cpu().numpy(), True)
            hidden = None
            lstm_out, hidden = self.lstm.forward(word_embs, hidden)
            lstm_out = self.droplstm(lstm_out)
        return lstm_out

    def get_output_score(self, mask, word_text, word_inputs, biword_inputs, word_seq_lens, batch_pos, external_pos):
        lstm_out = self.get_lstm_features(mask, word_text, word_inputs, biword_inputs, word_seq_lens, batch_pos, external_pos)
        # lstm_out (batch_size, sent_len, hidden_dim)
        outputs = self.hidden2tag(lstm_out)
        return outputs
