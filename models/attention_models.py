import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.nn.init import xavier_normal, xavier_uniform, orthogonal
from model_utils import Identity



class MultiAttention(nn.Module):

    def __init__(self, mem_size, out_size, k=2, enc_config=None, style='simple'):
        """
        param mem_size: the size of the shared memory vector
        param out_size: the size of the output (logit) space
        param k: number of memory updates steps (int)
        param enc_config: a dictionary with keys ['visual', 'audio', 'text'] corresponding to the types of
                        encoder used for each modality. 
                        enc_config['visual'] in [('resnet', size), ('resnet_ext_context', size), ('facet', size), None]
                        enc_config['audio'] in [('vocalnet', size), ('vocalnet_ext_context', size), ('covarep', size), None]
                        enc_config['text'] in [('LSTM', size), ('LSTM_ext_context', size), None]
                        where x_ext_context indicates encoder models which are provided a shared
                        memory context, and None indicates that this modality is not included, and 
                        size is the length of the LSTM hidden state.
        param style: the type of memory update mechanism. Choices are 'simple' (= same as DAN) or 'gated' (new, experimental!)
        """

        super(MultiAttention, self).__init__()
        self.mem_size = mem_size
        self.style = style

        if not enc_config:
            self.enc_config = {'visual': ('facet', 512), # ??
                               'audio': ('covarep', 74), # ??
                               'text': (None, None)}
        else:
            self.enc_config = enc_config

        # visual setup
        if enc_config['visual'][0]:
            v_hid_size = enc_config['visual'][1]
            assert v_hid_size == mem_size # as in DAN, but we could be more flexible later if we want..
            self.linears_v = []
            self.linears_v_m = []
            self.linears_v_h = []
            for step in range(k):
                # for now fix len(h_u) = len(m) = mem_size, but in general they are different
                self.linears_v[step] = nn.Linear(v_hid_size, mem_size)
                self.linears_v_m[step] = nn.Linear(mem_size, mem_size) 
                self.linears_v_h[step] = nn.Linear(mem_size, mem_size)

        # audio setup
        if enc_config['audio'][0]:
            a_hid_size = enc_config['audio'][1]
            assert a_hid_size == mem_size # as in DAN, but we could be more flexible later if we want..
            self.linears_a = []
            self.linears_a_m = []
            self.linears_a_h = []
            for step in range(k):
                self.linears_a = nn.Linear(a_hid_size, mem_size)
                self.linears_a_m = nn.Linear(mem_size, mem_size)
                self.linears_a_h = nn.Linear(mem_size, mem_size)

        # textual setup
        if enc_config['text'][0]:
            t_hid_size = enc_config['text'][1]
            assert t_hid_size == mem_size # as in DAN, but we could be more flexible later if we want..
            self.linears_t = []
            self.linears_t_m = []
            self.linears_t_h = []
            for step in range(k):
                self.linears_t = nn.Linear(t_hid_size, mem_size)
                self.linears_t_m = nn.Linear(mem_size, mem_size)
                self.linears_t_h = nn.Linear(mem_size, mem_size)


        # final fc projection to output logit space
        self.fc = nn.Linear(mem_size, out_size)

        # memory update gates
        if style == 'simple': # same memory update as DAN paper
            self.keep_gate_v = self.keep_gate_a = self.keep_gate_t = Identity()
            self.forget_get = Identity()
        elif style == 'gated': # a new keep/forget gated version 
            self.keep_gate_

        # utility vars
        self.mem_addition = torch.ones(mem_size, 1) # see shared memory update in forward

    def forward(self, visual_in, audio_in, text_in=None):

        # ensure views and shapes are correct



        # initialise attention state and memory
        v_0 = None
        a_0 = None
        t_0 = None
        if visual_in:
            v_0 = 
        if audio_in:
            a_0 = 
        if text_in:
            t_0 = 





        # modality-specific attention mechanisms and shared memory update
        m_list = []
        m_old = m_0
        v_old = v_0
        a_old = a_0
        t_old = t_0

        for step in range(k):
            if self.style == 'simple':
                # m_new = m_old
                addition = self.mem_addition.view(m_0.size())
            elif self.style == 'gated':
                # TODO: implement gated update addition
            if visual_in:
                # TODO: implement h_v, att_v and v_new update
                addition *= v_new
                v_old = v_new
            if audio_in:
                # TODO: implement h_a, att_a, and a_new update
                addition *= a_new
                v_old = v_new
            if text_in:
                # TODO: implement h_t, att_t, and t_new update
                addition *= t_new
                v_old = v_new
            m_new = m_old + addition
            m_list[step] = m_new
            m_old = m_new

            outputs = self.fc(m_new)

            return outputs # may also want to optionally return m_list for visualisation 






