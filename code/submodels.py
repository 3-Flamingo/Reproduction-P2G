import math
import torch
import torch.nn as nn
from torch.nn import Parameter, Module


class EventComposition(Module):
    """
    Event Composition layer
        integrate event argument embedding into event embedding

    Inputs:
        inputs: arguments embedding
        inputs.shape = (batch_size, argument_length, embedding_size)

    Outputs:
        outputs: event embedding
        outputs.shape = (batch_size, event_length, hidden_size)
    """

    def __init__(self, inputs_size, outputs_size, dropout):
        super(EventComposition, self).__init__()
        self.outputs_size = outputs_size

        self.w_e = InitLinear(inputs_size * 4, outputs_size, dis_func="normal", func_value=0.02)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs):
        inputs = inputs.view(-1, inputs.size(1) // 4, self.outputs_size)
        outputs = self.dropout(torch.tanh(self.w_e(inputs)))
        return outputs

class Attention(Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        Attention
        self.w_i1 = nn.Linear(hidden_size, hidden_size // 2)
        self.w_i2 = nn.Linear(hidden_size // 2, 1)
        self.w_c1 = nn.Linear(hidden_size, hidden_size // 2)
        self.w_c2 = nn.Linear(hidden_size // 2, 1)

    def forward(self, inputs):  # inputs: [batch, 13, event_dim]
        batch_size = inputs.shape[0]
        context = inputs[:, 0:8, :].repeat(1, 5, 1).view(-1, 8, self.hidden_size)  # [batch*5, 8, event_dim]
        candidate = inputs[:, 8:13, :]  # # [batch, 5, event_dim]
        s_i = torch.relu(self.w_i1(context))
        s_i = torch.relu(self.w_i2(s_i))  # [batch*5, 8, 1]
        s_c = torch.relu(self.w_c1(candidate))
        s_c = torch.relu(self.w_c2(s_c))  # [batch, 5, 1]
        # [batch*5, 8] + [batch*5, 1]
        # print("si: ", s_i.shape, s_i.view(-1, 8).shape)
        # print("sc: ", s_c.shape, s_c.view(-1, 1).shape)
        u = torch.tanh(torch.add(s_i.view(-1, 8), s_c.view(-1, 1)))
        # print("##: ", s_i.shape, s_c.shape, u.shape)
        a = (torch.exp(u) / torch.sum(torch.exp(u), 1).view(-1, 1)).view(-1, 8, 1)
        h_i = torch.sum(torch.mul(context, a), 1)
        h_c = (candidate / 8.0).view(-1, self.hidden_size)
        return h_i, h_c

class SelfAttention(Module):
    """
    Self-Attention Layer

    Inputs:
        inputs: word embedding
        inputs.shape = (batch_size, sequence_length, embedding_size)

    Outputs:
        outputs: word embedding with context information
        outputs.shape = (batch_size, sequence_length, embedding_size)
    """

    def __init__(self, embedding_size, n_heads, dropout):
        super(SelfAttention, self).__init__()
        assert embedding_size % n_heads == 0
        self.embedding_size = embedding_size
        self.n_heads = n_heads
        self.d_k = embedding_size // n_heads

        self.w_qkv = InitLinear(embedding_size, embedding_size * 3, dis_func="normal", func_value=0.02)
        self.w_head = InitLinear(embedding_size, embedding_size, dis_func="normal", func_value=0.02)
        self.dropout = nn.Dropout(dropout)

    def split_heads(self, x, k=False):
        x = x.view(-1, x.size(1), self.n_heads, self.d_k)
        if k:
            return x.permute(0, 2, 3, 1)  # key.shape = (batch_size, n_heads, d_k, sequence_length)
        else:
            return x.permute(0, 2, 1, 3)  # query, value.shape = (batch_size, n_heads, sequence_length, d_k)

    def attention(self, query, key, value, mask=None):
        att = torch.matmul(query, key) / math.sqrt(self.d_k)

        if mask is not None:
            att = att + mask

        att = torch.softmax(att, -1)
        att = self.dropout(att)

        # att = self.sample(att)

        outputs = torch.matmul(att, value)
        return outputs

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(-1, x.size(1), x.size(2) * x.size(3))
        x = self.dropout(self.w_head(x))
        return x

    @staticmethod
    def sample(att):
        att_ = att.view(-1, att.size(-1))
        _, tk = torch.topk(att_, k=26, largest=False)
        for i in range(att_.size(0)):
            att_[i][tk[i]] = 0.0
        att = att_.view(att.size())
        return att

    def forward(self, inputs, mask=None):
        inputs = self.w_qkv(inputs)
        query, key, value = torch.split(inputs, self.embedding_size, 2)
        query = self.split_heads(query)
        key = self.split_heads(key, k=True)
        value = self.split_heads(value)

        att_outputs = self.attention(query, key, value, mask)
        outputs = self.merge_heads(att_outputs)
        return outputs


class InitLinear(Module):
    """
    Initialize Linear layer to be distribution function
    """

    def __init__(self, inputs_size, outputs_size, dis_func, func_value, bias=True):
        super(InitLinear, self).__init__()
        self.outputs_size = outputs_size

        self.weight = Parameter(torch.empty(inputs_size, outputs_size), requires_grad=True)
        if bias:
            self.bias = Parameter(torch.empty(outputs_size), requires_grad=True)
        else:
            self.bias = None

        self.reset_parameters(dis_func, func_value)

    def reset_parameters(self, dis_func, func_value):
        if dis_func == "uniform":
            nn.init.uniform_(self.weight, -func_value, func_value)
            if self.bias is not None:
                nn.init.uniform_(self.bias, -func_value, func_value)

        if dis_func == "normal":
            nn.init.normal_(self.weight, std=func_value)
            if self.bias is not None:
                nn.init.normal_(self.bias, std=func_value)

    def forward(self, inputs):
        output_size = inputs.size()[:-1] + (self.outputs_size,)
        if self.bias is not None:
            outputs = torch.addmm(self.bias, inputs.reshape(-1, inputs.size(-1)), self.weight)
        else:
            outputs = torch.mm(inputs.view(-1, inputs.size(-1)), self.weight)
        outputs = outputs.view(*output_size)
        return outputs