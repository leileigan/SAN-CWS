import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
from model.sequence import Seq
from model.crf import CRF

seed_num = 100
torch.manual_seed(seed_num)
torch.cuda.manual_seed(seed_num)


class CWS(nn.Module):
    def __init__(self, data):
        super(CWS, self).__init__()
        print("build batched vallina lstmcrf...")
        self.gpu = data.HP_gpu
        #  add two more label for downlayer lstm, use original label size for CRF
        label_size = data.label_alphabet_size
        data.label_alphabet_size += 2
        self.lstm = Seq(data)
        self.crf = CRF(label_size, self.gpu)
        print("finished built model: ", self)

    def neg_log_likelihood_loss(self, word_text, word_inputs, biword_inputs, batch_label, mask, word_seq_lens, batch_pos):

        outs = self.lstm.get_output_score(mask, word_text, word_inputs, biword_inputs, word_seq_lens, batch_pos, external_pos={})
        batch_size = word_inputs.size(0)
        seq_len = word_inputs.size(1)
        total_loss = self.crf.neg_log_likelihood_loss(outs, mask, batch_label)
        _, tag_seq = self.crf._viterbi_decode(outs, mask)
        return total_loss, tag_seq

    def forward(self, word_text, word_inputs, biword_inputs, word_seq_lens, mask, batch_pos, external_pos):

        outs = self.lstm.get_output_score(mask, word_text, word_inputs, biword_inputs, word_seq_lens, batch_pos, external_pos)
        batch_size = word_inputs.size(0)
        seq_len = word_inputs.size(1)
        scores, tag_seq = self.crf._viterbi_decode(outs, mask)
        return tag_seq
