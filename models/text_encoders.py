import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.nn.init import xavier_normal, xavier_uniform, orthogonal
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence


class TextEncoder(nn.Module):
    """
    Bi-LSTM encoder for textual input. Can be incorporated into multi-attention.
    The shared memory modulates the word-level attention,
    as per DAN paper, i.e. the shared memory affects how the word encodings are attended to,
    but now how they were derived from the word embeddings via recurrent encoding.
    """
    def __init__(self, in_size, hid_size, out_size, batch_size, num_layers=1, dropout=0.2, bidirectional=False, batch_first=True):


        super(TextEncoder, self).__init__()
        self.rnn = nn.LSTM(in_size, hid_size, num_layers=num_layers, dropout=dropout,
                           bidirectional=bidirectional, batch_first=batch_first)
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.hid_size = hid_size

    def init_hidden(self):
        num_directions = 2 if self.bidirectional else 1
        return ((torch.zeros(self.num_layers * num_directions, self.batch_size, self.hid_size)),
                (torch.zeros(self.num_layers * num_directions, self.batch_size, self.hid_size)))

    def forward(self, x, seq_lens=None):
        """
        
        param x: tensor of shape (batch_size, max_seq_len, in_size)
        param seq_lens: 
        """
        if not isinstance(x, PackedSequence):
            output, (final_h, final_c) = self.rnn(x)
            # print("not packed!")
            # print(x.size())
        else:
            packed_input = pack_padded_sequence(x, seq_lens, batch_first=True)
            packed_output, (final_h, final_c) = self.rnn(packed_input)
            output, _ = pad_packed_sequence(packed_output, batch_first=True)

        return output, (final_h, final_c)

class TextOnlyModel(nn.Module):
    """
    Text only model. Encodes with TextEncoder and then projects final hidden state through a FC layer
    """
    def __init__(self, in_size, hid_size, out_size, batch_size, num_layers=1, 
                rnn_dropout=0.2, post_dropout=0.2, bidirectional=False, 
                output_scale_factor=1, output_shift=0, self_attention=None):

        super(TextOnlyModel, self).__init__()
        self.rnn_enc = TextEncoder(in_size, hid_size, out_size, batch_size, num_layers=num_layers, dropout=rnn_dropout, bidirectional=bidirectional, batch_first=True)
        self.dropout = nn.Dropout(post_dropout)
        num_directions = 2 if bidirectional else 1
        self.linear_last = nn.Linear(hid_size * num_directions * num_layers, out_size)
        # self.output_scale_factor = Parameter(torch.FloatTensor([output_scale_factor]), requires_grad=False)
        # self.output_shift = Parameter(torch.FloatTensor([output_shift]), requires_grad=False)
        if self_attention == 'typeA':
            self.self_att_layer = SelfAttention_A(hid_size)
        elif self_attenntion == 'typeB':
            self.self_att_layer = SelfAttention_B(hid_size, hid_size * 4)
        else: 
            self.self_att_layer = None

    def forward(self, x, seq_lens=None):
        """

        param x: tensor of shape (batch_size, max_seq_len, in_size)
        """
        _, (final_h, final_c) = self.rnn_enc(x, seq_lens)
        # print(final_h.size())
        if self_attention is not None:
            encoded = self.self_att_layer(final_h)
        else:
            encoded = final_h

        final_h_drop = self.dropout(final_h.squeeze()) # num_dir, batch_size, hid_size
        # print(final_h_drop.size())
        # stack along first dim if bidir (one dim for each direction)
        if final_h_drop.size()[0] == 2:
            final_h_drop = torch.cat((final_h_drop[0], final_h_drop[1]), 0)
            # print(final_h_drop.size())
        y = F.sigmoid(self.linear_last(final_h_drop))
        # y = y*self.output_scale_factor + self.output_shift
        y = torch.clamp(y, 0,3)
        return y


# class TextEncoderWrapper:
#     """
#     Wrapper for TextEncoder to be used when we want an encapsulated way to access all h and c states
#     of the rnn. Pytorch's LSTM class only outputs (i) the hidden states of the *final* layer at each
#     time step t=1:seq_len, and (ii) the (h, c) states for final step t=seq_len
#     """

class TextEncoderExtContext(nn.Module):
    """
    Bi-LSTM encoder for textual input which incorporates an external memory state into the
    encoding dynamics when generating a latent summary state.
    """
    def __init__(self, in_size, hid_size, out_size, num_layers=1, dropout=0.2, bidirectional=False):


        super(TextEncoder, self).__init__()
        self.rnn = nn.LSTM(in_size, hid_size, num_layers=num_layers, dropout=dropout,
                           bidirectional=bidirectional, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.linear_last = nn.Linear(hid_size, out_size)
        self.m2h_proj = nn.Linear(mem_size, hid_size*num_layers)
        self.m2c_proj = nn.Linear(mem_size, hid_size*num_layers)

    def forward(self, x, ext_memory=None):
        """
        param x: the input sequence of word embeddings
        type x: tensor of shape (batch_size, seq_len, in_size)
        param ext_memory: external shared memory which provides a context for the encoder
                          via an initial hidden state
        type ext_memory: tensor of shape (batch_size, mem_size)
        """
        if ext_memory:
            h_init = self.m2h_proj(ext_memory)
            c_init = self.m2c_proj(ext_memory)
            output, final_hiddens = self.rnn(x, (h_init, c_init))
            # TODO: do we need a final FC projection here (preceded by dropout)?
        else:
            output, final_hiddens = self.rnn(x)
            # TODO: do we need a final FC projection here (preceded by dropout)?

        return output, final_hiddens



class SelfAttention_A(nn.Module):
    def __init__(self, hidden_size, batch_first=True):
        super(SelfAttention, self).__init__()

        self.hidden_size = hidden_size
        self.batch_first = batch_first

        self.att_weights = Parameter(torch.Tensor(1, hidden_size),
                                     requires_grad=True)

        init.xavier_uniform(self.att_weights.data)

    def get_mask(self):
        pass

    def forward(self, inputs):

        if isinstance(inputs, PackedSequence):
            # unpack output
            inputs, lengths = pad_packed_sequence(inputs,
                                                  batch_first=self.batch_first)
        if self.batch_first:
            batch_size, max_len = inputs.size()[:2]
        else:
            max_len, batch_size = inputs.size()[:2]

        # apply attention layer
        weights = torch.bmm(inputs,
                            self.att_weights  # (1, hidden_size)
                            .permute(1, 0)  # (hidden_size, 1)
                            .unsqueeze(0)  # (1, hidden_size, 1)
                            .repeat(batch_size, 1, 1)
                            # (batch_size, hidden_size, 1)
                            )

        attentions = F.softmax(F.relu(weights.squeeze()))

        # create mask based on the sentence lengths
        mask = Variable(torch.ones(attentions.size())).cuda()
        for i, l in enumerate(lengths):  # skip the first sentence
            if l < max_len:
                mask[i, l:] = 0

        # apply mask and renormalize attention scores (weights)
        masked = attentions * mask
        _sums = masked.sum(-1).expand_as(attentions)  # sums per row
        attentions = masked.div(_sums)

        # apply attention weights
        weighted = torch.mul(inputs, attentions.unsqueeze(-1).expand_as(inputs))

        # get the final fixed vector representations of the sentences
        representations = weighted.sum(1).squeeze()

        return representations, attentions

class SelfAttention_B(nn.Module):
    def __init__(self, enc_hidden_size, att_hidden_size):
        super(SelfAttention_B, self).__init__()
        # this naive implementation can only handle batch sizes = 1, so
        assert len(enc_hidden_size.size()) == 2 # to reject batch size > 1

        self.enc_hidden_size = enc_hidden_size
        self.att_hidden_size = att_hidden_size
        self.W_linear = nn.Linear(enc_hidden_size, att_hidden_size)
        self.v_linear = nn.Linear(att_hidden_size, 1)

    def forward(self, x):
        """
        param x: (seq_len, enc_hidden_size) Tensor
        """
        x = x.transpose(1,0) # (enc_hidden_size, seq_len)
        a_unnorm = self.v_linear(F.tanh(self.W_linear(x))) # (1, seq_len)

        a = F.softmax(a_unnorm, 1)

        convex_comb = x * a

        return convex_comb

