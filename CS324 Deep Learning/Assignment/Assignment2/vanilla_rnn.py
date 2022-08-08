from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F


class VanillaRNN(nn.Module):

    # 1 2 3 2 1   sample 1 2 3 2  label 1
    # seq_length = 4
    # input dim = 1
    # hidden_dim = 128(默认)
    # output_dim = 10 (0-9之间一个数)
    # batch_size = 128(默认)
    def __init__(self, seq_length, input_dim, hidden_dim, output_dim, batch_size):
        super(VanillaRNN, self).__init__()
        # Initialization here ...
        self.layer_num = seq_length  # 可以参考图片
        self.batch = batch_size
        self.input_dim = input_dim
        self.h = hidden_dim
        n = input_dim
        m = output_dim
        self.Wx = nn.Linear(n, self.h, bias=True)
        self.Wh = nn.Linear(self.h, self.h, bias=False)
        self.Wp = nn.Linear(self.h, m, bias=True)

    # 输入的 x 是一串回文数字
    # 如 [2. 3. 3. 0. 4. 4. 0. 3. 3. 2.]
    def forward(self, x):
        # Implementation here ...
        # 将一批输入 x 转换成 batch x 1 的 tensor

        # batch=4 len=10
        # [2. 3. 3. 0. 4. 4. 0. 3. 3. 2.]
        # [2. 3. 3. 0. 4. 4. 0. 3. 3. 2.]
        # [2. 3. 3. 0. 4. 4. 0. 3. 3. 2.]
        # [2. 3. 3. 0. 4. 4. 0. 3. 3. 2.]
        x_list = list()
        for t in range(self.layer_num):
            x_num = torch.zeros([self.batch, self.input_dim])
            for j in range(self.batch):
                x_num[j] = x[j][t]
            x_list.append(x_num)

        ht = torch.zeros([self.batch, self.h])
        for t in range(self.layer_num):
            ht = torch.tanh(self.Wx(x_list[t]) + self.Wh(ht))
        ot = self.Wp(ht)
        #y_hat = F.softmax(ot, dim=1)
        return ot

    # add more methods here if needed
