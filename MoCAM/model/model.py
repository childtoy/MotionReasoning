import torch
import torch.nn as nn
import numpy as np
from model.TCN import TemporalConvNet
from model.PURE1D import Pure1dNet
from model.PURE3D import Pure3dNet
import torch.nn.functional as F
from model.MLP import MlpNet



class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)
        self.softmax = nn.Softmax(dim=1)
    def forward(self, inputs):
        y1 = self.tcn(inputs)
        output = self.linear(y1[:,:,-1])
        return F.log_softmax(output)



class PURE1D(nn.Module):
    def __init__(self, input_size, output_size, kernel_size, dropout):
        super(PURE1D, self).__init__()
        self.net = Pure1dNet(input_size, output_size, kernel_size=kernel_size, dropout=dropout)
    def forward(self, inputs):
        output = self.net(inputs)

        return F.log_softmax(output)



class PURE3D(nn.Module):
    def __init__(self, input_size, n_channels, output_size, kernel_size, dropout):
        super(PURE3D, self).__init__()
        self.net = Pure3dNet(input_size, n_channels, output_size, kernel_size=kernel_size, dropout=dropout)
    def forward(self, inputs):
        output = self.net(inputs)

        return F.log_softmax(output)


class MLP(nn.Module):
    def __init__(self, input_size, output_size, dropout):
        super(MLP, self).__init__()
        self.net = MlpNet(input_size, output_size, dropout=dropout)
        # self.softmax = nn.Softmax(dim=1)
    def forward(self, inputs):
        output = self.net(inputs)
        return F.log_softmax(output)
    
        # return self.softmax(output)        

class MoT(nn.Module):
    """
    Encoder Network
    """
    def __init__(self, embed_dim, hidden, char_nums, char_maxlen, attn_layers, attn_heads):
        """
        :param embedding_size: dimension of embedding
        :param num_hidden: dimension of hidden
        """
        super(MoT, self).__init__()
        self.hidden = hidden
        self.embed_dim = embed_dim
        self.char_nums = char_nums
        self.char_maxlen = char_maxlen
        self.attn_layers = attn_layers
        self.attn_heads = attn_heads

        self.alpha = nn.Parameter(torch.ones(1))
        self.embed = nn.Embedding(self.char_nums, embed_dim, padding_idx=0)
        self.pos_emb = nn.Embedding.from_pretrained(get_sinusoid_encoding_table(self.char_maxlen, hidden, padding_idx=0),
                                                    freeze=True)
        self.pos_dropout = nn.Dropout(p=hp.pos_dropout_rate)
        self.encoder_prenet = EncoderPrenet(embed_dim, hidden)
        self.layers = clones(Attention(hidden, self.attn_heads), self.attn_layers)
        self.ffns = clones(FFN(hidden), self.attn_layers)

        # self.init_model()
        # self.step = nn.Parameter(torch.zeros(1).long(), requires_grad=False)

    def forward(self, x, pos):

        if self.training:
            c_mask = x.ne(0).type(torch.float)
            mask = x.eq(0).unsqueeze(1).repeat(1, x.size(1), 1)

        else:
            c_mask, mask = None, None

        x = self.embed(x)           # B*T*d
        x = self.encoder_prenet(x)  

        # Get positional embedding, apply alpha and add
        pos = self.pos_emb(pos)
        x = pos * self.alpha + x

        # Positional dropout
        x = self.pos_dropout(x)

        # Attention encoder-encoder
        attns = list()
        for layer, ffn in zip(self.layers, self.ffns):
            x, attn = layer(x, x, mask=mask, query_mask=c_mask)
            x = ffn(x)
            attns.append(attn)

        return x, attns
