import torch
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


        gate = F.sigmoid(self.gate(c))

        x = gate * t + (1 - gate) * c

        return x

class GatedMemUpdate_B(nn.Module):
    """
    Differs from GatedMemUpdate by allowing both t and c to control
    the gating behaviour. In general the size of the tensor to be gated
    will differ from the size of the tensor controlling the gating.
    Further, the tensor to be routed through when the gate is 'on' is
    in general another external signal u (neither c nor g).
    So, in summary,
    - t is passed when gate is 'off'
    - both t and g control the gate, and
    - u is passed through when gate is 'on'
    """
    def __init__(self, gate_in_size, gate_out_size):

        super(GatedMemUpdate_B, self).__init__()

        self.gate = nn.Linear(gate_in_size, gate_out_size)

    def forward(self, g, c, u):

        concated = torch.cat((g, c))
        gate = F.sigmoid(self.gate(concated))

        x = gate * u + (1 - gate) * c

        return x
