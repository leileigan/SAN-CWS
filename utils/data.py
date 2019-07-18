import sys
import numpy as np
from utils.alphabet import Alphabet
from utils.functions import *
# import cPickle as pickle
import pickle
import gensim

START = "</s>"
UNKNOWN = "</unk>"
PADDING = "</pad>"
NULLKEY = "-null-"


class Data:
    def __init__(self):
        self.MAX_SENTENCE_LENGTH = 250
        self.MAX_WORD_LENGTH = -1
        self.number_normalized = False
        self.norm_word_emb = True
        self.norm_biword_emb = True
        self.word_alphabet = Alphabet('word')
        self.biword_alphabet = Alphabet('biword')
        self.pos_alphabet = Alphabet('pos')
        self.label_alphabet = Alphabet('label', True)

        self.tagScheme = "NoSeg"
        self.char_features = "LSTM"

        self.train_texts = []
        self.dev_texts = []
        self.test_texts = []
        self.raw_texts = []

        self.train_Ids = []
        self.dev_Ids = []
        self.test_Ids = []
        self.raw_Ids = []
        self.use_bigram = False
        self.word_emb_dim = 50
        self.biword_emb_dim = 50

        self.pretrain_word_embedding = None
        self.pretrain_biword_embedding = None
        self.label_size = 0
        self.word_alphabet_size = 0
        self.biword_alphabet_size = 0
        self.label_alphabet_size = 0
        #  hyperparameters
        self.HP_iteration = 100
        self.HP_batch_size = 16
        self.HP_char_hidden_dim = 50
        self.HP_hidden_dim = 200
        self.HP_dropout = 0.2
        self.HP_lstmdropout = 0.5
        self.HP_lstm_layer = 1
        self.HP_bilstm = True
        self.HP_gpu = False
        self.HP_lr = 0.015
        self.HP_lr_decay = 0.05
        self.HP_clip = 5.0
        self.HP_momentum = 0

        #  attention
        self.tencent_word_embed_dim = 200
        self.pos_embed_dim = 200
        self.cross_domain = False
        self.cross_test = False
        self.use_san = False
        self.use_cnn = False
        self.use_attention = True
        self.pos_to_idx = {}
        self.external_pos = {}
        self.token_replace_prob = {}
        self.use_adam = False
        self.use_bert = False
        self.use_warmup_adam = False
        self.use_sgd = False
        self.use_adadelta = False
        self.use_window = True
        self.mode = 'train'
        self.use_tencent_dic = False

        # cross domain file
        self.computer_file = ""
        self.finance_file = ""
        self.medicine_file = ""
        self.literature_file = ""

    def show_data_summary(self):
        print("DATA SUMMARY START:")
        print("     Tag          scheme: %s" % (self.tagScheme))
        print("     MAX SENTENCE LENGTH: %s" % (self.MAX_SENTENCE_LENGTH))
        print("     MAX   WORD   LENGTH: %s" % (self.MAX_WORD_LENGTH))
        print("     Number   normalized: %s" % (self.number_normalized))
        print("     Use          bigram: %s" % (self.use_bigram))
        print("     Char  alphabet size: %s" % (self.word_alphabet_size))
        print("     BiChar alphabet size: %s" % (self.biword_alphabet_size))
        print("     Label alphabet size: %s" % (self.label_alphabet_size))
        print("     Char embedding size: %s" % (self.word_emb_dim))
        print("     BiChar embedding size: %s" % (self.biword_emb_dim))
        print("     Norm     char   emb: %s" % (self.norm_word_emb))
        print("     Norm     bichar emb: %s" % (self.norm_biword_emb))
        print("     Train instance number: %s" % (len(self.train_texts)))
        print("     Dev   instance number: %s" % (len(self.dev_texts)))
        print("     Test  instance number: %s" % (len(self.test_texts)))
        print("     Raw   instance number: %s" % (len(self.raw_texts)))
        print("     Hyperpara  iteration: %s" % (self.HP_iteration))
        print("     Hyperpara  batch size: %s" % (self.HP_batch_size))
        print("     Hyperpara          lr: %s" % (self.HP_lr))
        print("     Hyperpara    lr_decay: %s" % (self.HP_lr_decay))
        print("     Hyperpara     HP_clip: %s" % (self.HP_clip))
        print("     Hyperpara    momentum: %s" % (self.HP_momentum))
        print("     Hyperpara  hidden_dim: %s" % (self.HP_hidden_dim))
        print("     Hyperpara     dropout: %s" % (self.HP_dropout))
        print("     Hyperpara  lstm_layer: %s" % (self.HP_lstm_layer))
        print("     Hyperpara      bilstm: %s" % (self.HP_bilstm))
        print("     Hyperpara         GPU: %s" % (self.HP_gpu))
        print("     Cross domain: %s" % self.cross_domain)
        print("     Hyperpara  use window: %s" % self.use_window)
        print("DATA SUMMARY END.")
        sys.stdout.flush()

    def build_alphabet(self, input_file):
        in_lines = open(input_file, 'r').readlines()
        for idx in range(len(in_lines)):
            line = in_lines[idx]
            if len(line) > 2:
                pairs = line.strip().split('\t')
                # word = pairs[0].decode('utf-8')
                word = pairs[0]
                if self.number_normalized:
                    word = normalize_word(word)
                label = pairs[-1][0] + '-SEG'
                self.label_alphabet.add(label)
                self.word_alphabet.add(word)
                if idx < len(in_lines) - 1 and len(in_lines[idx + 1]) > 2:
                    # biword = word + in_lines[idx + 1].strip('\t').split()[0].decode('utf-8')
                    biword = word + in_lines[idx + 1].strip('\t').split()[0]
                else:
                    biword = word + NULLKEY
                self.biword_alphabet.add(biword)

        self.word_alphabet_size = self.word_alphabet.size()
        self.biword_alphabet_size = self.biword_alphabet.size()
        self.label_alphabet_size = self.label_alphabet.size()
        startS = False
        startB = False
        for label, _ in self.label_alphabet.iteritems():
            if "S-" in label.upper():
                startS = True
            elif "B-" in label.upper():
                startB = True
        if startB:
            if startS:
                self.tagScheme = "BMES"
            else:
                self.tagScheme = "BIO"

    def fix_alphabet(self):
        self.word_alphabet.close()
        self.biword_alphabet.close()
        self.label_alphabet.close()

    def build_word_pretrain_emb(self, emb_path):
        print("build word pretrain emb...")
        self.pretrain_word_embedding, self.word_emb_dim = build_pretrain_embedding(emb_path, self.word_alphabet, self.word_emb_dim,
                                                                                   self.norm_word_emb)

    def build_biword_pretrain_emb(self, emb_path):
        print("build biword pretrain emb...")
        self.pretrain_biword_embedding, self.biword_emb_dim = build_pretrain_embedding(emb_path, self.biword_alphabet, self.biword_emb_dim,
                                                                                       self.norm_biword_emb)

    def build_word_vec_100(self):
        self.pretrain_word_embedding, self.pretrain_biword_embedding = self.get_embedding()
        self.word_emb_dim, self.biword_emb_dim = 100, 100

    # get pre-trained embeddings
    def get_embedding(self, size=100):
        fname = 'data/wordvec_' + str(size)
        print("build pretrain word embedding from: ", fname)
        word_init_embedding = np.zeros(shape=[self.word_alphabet.size(), size])
        bi_word_init_embedding = np.zeros(shape=[self.biword_alphabet.size(), size])
        pre_trained = gensim.models.KeyedVectors.load(fname, mmap='r')
        # pre_trained_vocab = set([unicode(w.decode('utf8')) for w in pre_trained.vocab.keys()])
        pre_trained_vocab = set([w for w in pre_trained.vocab.keys()])
        c = 0
        for word, index in self.word_alphabet.iteritems():
            if word in pre_trained_vocab:
                word_init_embedding[index] = pre_trained[word]
            else:
                word_init_embedding[index] = np.random.uniform(-0.5, 0.5, size)
                c += 1

        for word, index in self.biword_alphabet.iteritems():
            bi_word_init_embedding[index] = (word_init_embedding[self.word_alphabet.get_index(word[0])]
                                             + word_init_embedding[self.word_alphabet.get_index(word[1])]) / 2
        # word_init_embedding[word2id[PAD]] = np.zeros(shape=size)
        # bi_word_init_embedding[]
        print('oov character rate %f' % (float(c) / self.word_alphabet.size()))
        return word_init_embedding, bi_word_init_embedding

    def generate_instance(self, input_file, name):
        self.fix_alphabet()
        if name == "train":
            self.train_texts, self.train_Ids = read_instance(input_file, self.word_alphabet, self.biword_alphabet,
                                                             self.label_alphabet, self.number_normalized, self.MAX_SENTENCE_LENGTH)
        elif name == "dev":
            self.dev_texts, self.dev_Ids = read_instance(input_file, self.word_alphabet, self.biword_alphabet,
                                                         self.label_alphabet, self.number_normalized, self.MAX_SENTENCE_LENGTH)
        elif name == "test":
            self.test_texts, self.test_Ids = read_instance(input_file, self.word_alphabet, self.biword_alphabet,
                                                           self.label_alphabet, self.number_normalized, self.MAX_SENTENCE_LENGTH)
        elif name == "raw":
            self.raw_texts, self.raw_Ids = read_instance(input_file, self.word_alphabet, self.biword_alphabet,
                                                         self.label_alphabet, self.number_normalized, self.MAX_SENTENCE_LENGTH)
        else:
            print("Error: you can only generate train/dev/test instance! Illegal input:%s" % (name))

    def write_decoded_results(self, output_file, predict_results, name):
        fout = open(output_file, 'w')
        sent_num = len(predict_results)
        content_list = []
        if name == 'raw':
            content_list = self.raw_texts
        elif name == 'test':
            content_list = self.test_texts
        elif name == 'dev':
            content_list = self.dev_texts
        elif name == 'train':
            content_list = self.train_texts
        else:
            print("Error: illegal name during writing predict result, name should be within train/dev/test/raw !")
        assert (sent_num == len(content_list))
        for idx in range(sent_num):
            sent_length = len(predict_results[idx])
            for idy in range(sent_length):
                ## content_list[idx] is a list with [word, char, label]
                fout.write(content_list[idx][0][idy] + "\t" + predict_results[idx][idy][0] + '\n')

            fout.write('\n')
        fout.close()
        print("Predict %s result has been written into file. %s" % (name, output_file))
