from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

################################################################################

class LSTM(nn.Module):

    def __init__(self, seq_length, input_dim, hidden_dim, output_dim, batch_size):
        super(LSTM, self).__init__()
        # Initialization here ...
        self.layer_num = seq_length  # 可以参考图片
        self.batch = batch_size
        self.input_dim = input_dim
        self.h = hidden_dim
        n = input_dim
        m = output_dim
        self.Wgx = nn.Linear(n, self.h, bias=True)
        self.Wgh = nn.Linear(self.h, self.h, bias=False)
        self.Wix = nn.Linear(n, self.h, bias=True)
        self.Wih = nn.Linear(self.h, self.h, bias=False)
        self.Wfx = nn.Linear(n, self.h, bias=True)
        self.Wfh = nn.Linear(self.h, self.h, bias=False)
        self.Wox = nn.Linear(n, self.h, bias=True)
        self.Woh = nn.Linear(self.h, self.h, bias=False)
        self.Wp = nn.Linear(self.h, m, bias=True)

    def forward(self, x):
        # Implementation here ...
        x_list = list()
        for t in range(self.layer_num):
            x_num = torch.zeros([self.batch, self.input_dim])
            for j in range(self.batch):
                x_num[j] = x[j][t]
            x_list.append(x_num)

        ht = torch.zeros([self.batch, self.h])
        ct = torch.zeros([self.batch, self.h])
        for t in range(self.layer_num):
            gt = torch.tanh(self.Wgx(x_list[t]) + self.Wgh(ht))
            it = torch.sigmoid(self.Wix(x_list[t]) + self.Wih(ht))
            ft = torch.sigmoid(self.Wfx(x_list[t]) + self.Wfh(ht))
            ot = torch.sigmoid(self.Wox(x_list[t]) + self.Woh(ht))
            ct = gt * it + ct * ft
            ht = torch.tanh(ct) * ot
        y = self.Wp(ht)
        return y

    # add more methods here if needed