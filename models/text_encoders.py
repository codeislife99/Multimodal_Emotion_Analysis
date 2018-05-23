import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.nn.init import xavier_normal, xavier_uniform, orthogonal
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence
import torch.nn.utils.weight_norm as weight_norm

from torchmoji_local.lstm import LSTMHardSigmoid
from torchmoji_local.attlayer import Attention
from torchmoji_local.global_variables import NB_TOKENS, NB_EMOJI_CLASSES


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
        elif self_attention == 'typeB':
            self.self_att_layer = SelfAttention_B(hid_size, hid_size * 4, batch_size)
        else: 
            self.self_att_layer = None

    def forward(self, x, seq_lens=None):
        """

        param x: tensor of shape (batch_size, max_seq_len, in_size)
        """
        _, (final_h, final_c) = self.rnn_enc(x, seq_lens)
        # print(final_h.size())
        if self.self_att_layer is not None:
            encoded = self.self_att_layer(final_h)
        else:
            encoded = final_h

        final_h_drop = self.dropout(final_h.squeeze()) # num_dir, batch_size, hid_size
        # print(final_h_drop.size())
        # stack along first dim if bidir (one dim for each direction)
        if final_h_drop.size()[0] == 2:
            final_h_drop = torch.cat((final_h_drop[0], final_h_drop[1]), 0)
            # print(final_h_drop.size())
        # y = F.sigmoid(self.linear_last(final_h_drop))
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
        super(SelfAttention_A, self).__init__()

        self.hidden_size = hidden_size
        self.batch_first = batch_first

        self.att_weights = Parameter(torch.Tensor(1, hidden_size),
                                     requires_grad=True)

        xavier_uniform(self.att_weights.data)

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
    def __init__(self, enc_hidden_size, att_hidden_size, batch_size=1):
        super(SelfAttention_B, self).__init__()
        # this naive implementation can only handle batch sizes = 1, so
        assert batch_size == 1 # to reject batch size > 1

        self.enc_hidden_size = enc_hidden_size
        self.att_hidden_size = att_hidden_size
        self.W_linear = nn.Linear(enc_hidden_size, att_hidden_size)
        self.v_linear = nn.Linear(att_hidden_size, 1)

    def forward(self, x):
        """
        param x: (seq_len, enc_hidden_size) Tensor
        """
        # x = x.transpose(1,0) # (enc_hidden_size, seq_len)
        a_unnorm = self.v_linear(F.tanh(self.W_linear(x))) # (1, seq_len)

        a = F.softmax(a_unnorm, 1)

        convex_comb = x * a

        return convex_comb

class TorchMoji_Emb(nn.Module):
    def __init__(self, nb_classes, nb_tokens, feature_output=False, output_logits=True,
                 embed_dropout_rate=0, final_dropout_rate=0, return_attention=False):
        """
        torchMoji model, adjusted to accept embeddings rather than words.
        IMPORTANT: The model is loaded in evaluation mode by default (self.eval()) <-- REMOVED!
        # Arguments:
            nb_classes: Number of classes in the dataset.
            nb_tokens: Number of tokens in the dataset (i.e. vocabulary size).
            feature_output: If True the model returns the penultimate
                            feature vector rather than Softmax probabilities
                            (defaults to False).
            output_logits:  If True the model returns logits rather than probabilities
                            (defaults to False).
            embed_dropout_rate: Dropout rate for the embedding layer.
            final_dropout_rate: Dropout rate for the final Softmax layer.
            return_attention: If True the model also returns attention weights over the sentence
                              (defaults to False).
        """
        super(TorchMoji_Emb, self).__init__()

        embedding_dim = 300
        hidden_size = 512
        attention_size = 4 * hidden_size + embedding_dim

        self.feature_output = feature_output
        self.embed_dropout_rate = embed_dropout_rate
        self.final_dropout_rate = final_dropout_rate
        self.return_attention = return_attention
        self.hidden_size = hidden_size
        self.output_logits = output_logits
        self.nb_classes = nb_classes

        # self.add_module('embed', nn.Embedding(nb_tokens, embedding_dim)) # REMOVED!
        # dropout2D: embedding channels are dropped out instead of words
        # many exampels in the datasets contain few words that losing one or more words can alter the emotions completely
        # self.add_module('embed_dropout', nn.Dropout2d(embed_dropout_rate)) # REMOVED FOR NOW
        self.add_module('lstm_0', LSTMHardSigmoid(embedding_dim, hidden_size, batch_first=True, bidirectional=True))
        self.add_module('lstm_1', LSTMHardSigmoid(hidden_size*2, hidden_size, batch_first=True, bidirectional=True))
        self.add_module('attention_layer', Attention(attention_size=attention_size, return_attention=return_attention))
        if not feature_output:
            self.add_module('final_dropout', nn.Dropout(final_dropout_rate))
            if output_logits:
                self.add_module('output_layer', nn.Sequential(nn.Linear(attention_size, nb_classes if self.nb_classes > 2 else 1)))
            else:
                self.add_module('output_layer', nn.Sequential(nn.Linear(attention_size, nb_classes if self.nb_classes > 2 else 1),
                                                              nn.Softmax() if self.nb_classes > 2 else nn.Sigmoid()))
        self.init_weights()
        # Put model in evaluation mode by default
        # self.eval() REMOVED

    def init_weights(self):
        """
        Here we reproduce Keras default initialization weights for consistency with Keras version
        """
        ih = (param.data for name, param in self.named_parameters() if 'weight_ih' in name)
        hh = (param.data for name, param in self.named_parameters() if 'weight_hh' in name)
        b = (param.data for name, param in self.named_parameters() if 'bias' in name)
        # nn.init.uniform(self.embed.weight.data, a=-0.5, b=0.5) # REMOVED
        for t in ih:
            nn.init.xavier_uniform(t)
        for t in hh:
            nn.init.orthogonal(t)
        for t in b:
            nn.init.constant(t, 0)
        if not self.feature_output:
            nn.init.xavier_uniform(self.output_layer[0].weight.data)

    def forward(self, input_seqs):
        """ Forward pass.
        # Arguments:
            input_seqs: Can be one of Numpy array, Torch.LongTensor, Torch.Variable, Torch.PackedSequence. BATCH_FIRST!
        # Return:
            Same format as input format (except for PackedSequence returned as Variable).
        """
        # Check if we have Torch.LongTensor inputs or not Torch.Variable (assume Numpy array in this case), take note to return same format
        return_numpy = False
        return_tensor = False
        if isinstance(input_seqs, (torch.LongTensor, torch.cuda.LongTensor)):
            input_seqs = Variable(input_seqs)
            return_tensor = True
        elif not isinstance(input_seqs, Variable):
            input_seqs = Variable(torch.from_numpy(input_seqs.astype('int64')).long())
            return_numpy = True

        # If we don't have a packed inputs, let's pack it
        reorder_output = False
        if not isinstance(input_seqs, PackedSequence):
            ho = self.lstm_0.weight_hh_l0.data.new(2, input_seqs.size()[0], self.hidden_size).zero_()
            co = self.lstm_0.weight_hh_l0.data.new(2, input_seqs.size()[0], self.hidden_size).zero_()

            # Reorder batch by sequence length
            input_lengths = torch.LongTensor([torch.max(input_seqs[i, :][0].data.nonzero()) + 1 for i in range(input_seqs.size()[0])])
            input_lengths, perm_idx = input_lengths.sort(0, descending=True)
            input_seqs = input_seqs[perm_idx][:, :input_lengths.max()]

            # Pack sequence and work on data tensor to reduce embeddings/dropout computations
            # packed_input = pack_padded_sequence(input_seqs, input_lengths.cpu().numpy(), batch_first=True)
            packed_input = pack_padded_sequence(input_seqs, [1], batch_first=True)
            reorder_output = True
        else:
            ho = self.lstm_0.weight_hh_l0.data.data.new(2, input_seqs.size()[0], self.hidden_size).zero_()
            co = self.lstm_0.weight_hh_l0.data.data.new(2, input_seqs.size()[0], self.hidden_size).zero_()
            input_lengths = input_seqs.batch_sizes
            packed_input = input_seqs

        hidden = (Variable(ho, requires_grad=False), Variable(co, requires_grad=False))
        print(hidden[0].size())

        # Embed with an activation function to bound the values of the embeddings
        # x = self.embed(packed_input.data)
        # x = nn.Tanh()(x) # REMOVED FOR NOW
        x = packed_input.data # ADDED

        # pyTorch 2D dropout2d operate on axis 1 which is fine for us
        # x = self.embed_dropout(x) # REMOVED FOR NOW

        # Update packed sequence data for RNN
        print("x", x.size())
        print("batch_sizes", packed_input.batch_sizes)
        # packed_input = PackedSequence(data=x, batch_sizes=packed_input.batch_sizes)

        # skip-connection from embedding to output eases gradient-flow and allows access to lower-level features
        # ordering of the way the merge is done is important for consistency with the pretrained model
        print('packed_input', packed_input.data.size())
        lstm_0_output, _ = self.lstm_0(packed_input, hidden)
        lstm_1_output, _ = self.lstm_1(lstm_0_output, hidden)
        print('here', lstm_1_output.size())

        # Update packed sequence data for attention layer
        packed_input = PackedSequence(data=torch.cat((lstm_1_output.data,
                                                      lstm_0_output.data,
                                                      packed_input.data), dim=1),
                                      batch_sizes=packed_input.batch_sizes)

        input_seqs, _ = pad_packed_sequence(packed_input, batch_first=True)

        x, att_weights = self.attention_layer(input_seqs, input_lengths)

        # output class probabilities or penultimate feature vector
        if not self.feature_output:
            x = self.final_dropout(x)
            outputs = self.output_layer(x)
            outputs = torch.clamp(outputs.squeeze(), 0,3) # ADDED
        else:
            outputs = x

        # Reorder output if needed
        if reorder_output:
            reorered = Variable(outputs.data.new(outputs.size()))
            reorered[perm_idx] = outputs
            outputs = reorered

        # Adapt return format if needed
        if return_tensor:
            outputs = outputs.data
        if return_numpy:
            outputs = outputs.data.numpy()

        if self.return_attention:
            return outputs, att_weights
        else:
            return outputs

class BiLSTM(nn.Module):
    """
    Adapted from https://github.com/yezhejack/bidirectional-LSTM-for-text-classification/blob/master/models/RNN.py
    """
    def __init__(self, output_size, hidden_size=150, num_layer=2, embedding_freeze=False):
        super(BiLSTM,self).__init__()

        # embedding layer
        # vocab_size = embedding_matrix.shape[0]
        # embed_size = embedding_matrix.shape[1]
        embed_size = 300
        self.hidden_size = hidden_size
        self.num_layer = num_layer
        # self.embed = nn.Embedding(vocab_size, embed_size)
        # self.embed.weight = nn.Parameter(torch.from_numpy(embedding_matrix).type(torch.FloatTensor), requires_grad=not embedding_freeze)
        # self.embed_dropout = nn.Dropout(p=0.3)
        self.custom_params = []
        # if embedding_freeze == False:
        #     self.custom_params.append(self.embed.weight)

        # The first LSTM layer
        self.lstm1 = nn.LSTM(embed_size, self.hidden_size, num_layer, dropout=0.3, bidirectional=True, batch_first=True)
        for param in self.lstm1.parameters():
            self.custom_params.append(param)
            if param.data.dim() > 1:
                nn.init.orthogonal(param)
            else:
                nn.init.normal(param)

        self.lstm1_dropout = nn.Dropout(p=0.3)

        # The second LSTM layer
        self.lstm2 = nn.LSTM(2*self.hidden_size, self.hidden_size, num_layer, dropout=0.3, bidirectional=True, batch_first=True)
        for param in self.lstm2.parameters():
            self.custom_params.append(param)
            if param.data.dim() > 1:
                nn.init.orthogonal(param)
            else:
                nn.init.normal(param)
        self.lstm2_dropout = nn.Dropout(p=0.3)
        # Attention
        self.attention = nn.Linear(2*self.hidden_size,1)
        self.attention_dropout = nn.Dropout(p=0.5)

        # Fully-connected layer
        self.fc = weight_norm(nn.Linear(2*self.hidden_size, output_size))
        for param in self.fc.parameters():
            self.custom_params.append(param)
            if param.data.dim() > 1:
                nn.init.orthogonal(param)
            else:
                nn.init.normal(param)

        self.hidden1=self.init_hidden()
        self.hidden2=self.init_hidden()

    def init_hidden(self, batch_size=1):
        if torch.cuda.is_available():
            return (Variable(torch.zeros(self.num_layer*2, batch_size, self.hidden_size)).cuda(),
                    Variable(torch.zeros(self.num_layer*2, batch_size, self.hidden_size)).cuda())
        else:
            return (Variable(torch.zeros(self.num_layer*2, batch_size, self.hidden_size)),
                    Variable(torch.zeros(self.num_layer*2, batch_size, self.hidden_size)))

    def forward(self, sentences):
        # get embedding vectors of input
        # padded_sentences, lengths = torch.nn.utils.rnn.pad_packed_sequence(sentences, padding_value=int(0), batch_first=True)
        # embeds = self.embed(padded_sentences)
        noise = Variable(torch.zeros(embeds.shape).cuda())
        noise.data.normal_(std=0.3)
        # embeds += noise
        # embeds = self.embed_dropout(embeds)
        # add noise
        
        # packed_embeds = torch.nn.utils.rnn.pack_padded_sequence(embeds, lengths, batch_first=True)
        packed_embeds = sentences # not really packed.. 
        
        # First LSTM layer
        # self.hidden = num_layers*num_directions batch_size hidden_size
        packed_out_lstm1, self.hidden1 = self.lstm1(packed_embeds, self.hidden1)
        padded_out_lstm1, lengths = torch.nn.utils.rnn.pad_packed_sequence(packed_out_lstm1, padding_value=int(0))
        padded_out_lstm1 = self.lstm1_dropout(padded_out_lstm1)
        packed_out_lstm1 = torch.nn.utils.rnn.pack_padded_sequence(padded_out_lstm1, lengths)
   
        # Second LSTM layer
        packed_out_lstm2, self.hidden2 = self.lstm2(packed_out_lstm1, self.hidden2)
        padded_out_lstm2, lengths = torch.nn.utils.rnn.pad_packed_sequence(packed_out_lstm2, padding_value=int(0), batch_first=True)
        padded_out_lstm2 = self.lstm2_dropout(padded_out_lstm2)

        # attention
        unnormalize_weight = F.tanh(torch.squeeze(self.attention(padded_out_lstm2), 2))
        unnormalize_weight = F.softmax(unnormalize_weight, dim=1)
        unnormalize_weight = torch.nn.utils.rnn.pack_padded_sequence(unnormalize_weight, lengths, batch_first=True)
        unnormalize_weight, lengths = torch.nn.utils.rnn.pad_packed_sequence(unnormalize_weight, padding_value=0.0, batch_first=True)
        logging.debug("unnormalize_weight size: %s" % (str(unnormalize_weight.size())))
        normalize_weight = torch.nn.functional.normalize(unnormalize_weight, p=1, dim=1)
        normalize_weight = normalize_weight.view(normalize_weight.size(0), 1, -1)
        weighted_sum = torch.squeeze(normalize_weight.bmm(padded_out_lstm2), 1)
        
        # fully connected layer
        output = self.fc(self.attention_dropout(weighted_sum))
        return output
