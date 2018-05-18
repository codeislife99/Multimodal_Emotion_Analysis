import torch.nn as nn
import torch.nn.functional as F


class GatedMemUpdate(nn.Module):
    def __init__(self, size):

        super(GatedMemUpdate, self).__init__()

        self.gate = nn.Linear(size, size)

    def forward(self, t, c):
        """
            :param c: the value to be 'carried'
            :type c: tensor with shape of [batch_size, size]
            :param t: the 'transformed' value
            :type t: tensor with shape of [batch_size, size]
            :return: tensor with shape of [batch_size, size]

            Applies σ(c) ⨀ (t) + (1 - σ(c)) ⨀ (Q(c)) transformation | G and Q is affine transformation,
            σ(x) is affine transformation with sigmoid non-linearity
            and ⨀ is element-wise multiplication
            """

        for layer in range(self.num_layers):
            gate = F.sigmoid(self.gate[layer](c))

            linear = self.linear[layer](c)

            x = gate * t + (1 - gate) * c

        return x