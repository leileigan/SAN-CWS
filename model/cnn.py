import torch
import torch.nn as nn

seed_num = 10
torch.manual_seed(seed_num)
torch.cuda.manual_seed(seed_num)


class CNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layer, dropout, gpu=True):
        super(CNNModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layer = num_layer
        self.gpu = gpu

        print('-' * 10 + 'cnn hyper parameters' + '-' * 10)
        print('input dim: ', input_dim)
        print('hidden dim: ', hidden_dim)
        print('num layer: ', num_layer)

        self.cnn_layer0 = nn.Conv1d(self.input_dim, self.hidden_dim, kernel_size=1, padding=0)
        self.cnn_layers = [nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=3, padding=1) for i in range(self.num_layer - 1)]
        self.drop = nn.Dropout(dropout)

        if self.gpu:
            self.cnn_layer0 = self.cnn_layer0.cuda()
            for i in range(self.num_layer - 1):
                self.cnn_layers[i] = self.cnn_layers[i].cuda()

    def forward(self, input_feature):
        batch_size = input_feature.size(0)
        seq_len = input_feature.size(1)

        input_feature = input_feature.transpose(2, 1).contiguous()
        cnn_output = self.cnn_layer0(input_feature)
        cnn_output = self.drop(cnn_output)
        cnn_output = torch.tanh(cnn_output)

        for layer in self.cnn_layers:
            cnn_output = layer(cnn_output)
            cnn_output = self.drop(cnn_output)
            cnn_output = torch.tanh(cnn_output)

        cnn_output = cnn_output.transpose(2, 1).contiguous()
        return cnn_output