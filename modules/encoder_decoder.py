from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import json

from .att_model import pack_wrapper, AttModel
device = torch.device('cuda:0')

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn
def subsequent_mask(size):
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


class Transformer(nn.Module):
    def __init__(self, g_encoder,node_encoder,encoder, decoder, src_embed, tgt_embed):
        super(Transformer, self).__init__()
        self.g_encoder = g_encoder
        self.node_encoder = node_encoder
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.norm = LayerNorm(512)

    def get_node_embedding(self, embedding, matrix):
        embedding = embedding.unsqueeze(dim=0)
        matrix = matrix.unsqueeze(dim=0)
        embeddings = self.g_encoder(embedding, matrix)

        return embeddings

    def forward(self, txt_mean,image_features,embeddings, tgt, tgt_mask):
        return self.decode(txt_mean,image_features,embeddings,  tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, txt_mean,hidden_states,embeddings,  tgt, tgt_mask):
        b, l, _ = tgt_mask.size()
        add_mask = (torch.zeros([b, 1, l]) > 0).to(tgt_mask.device)
        tgt_mask = torch.cat([add_mask, tgt_mask], dim=1)
        add_mask = (torch.ones([b, l + 1, 1]) > 0).to(tgt_mask.device)
        tgt_mask = torch.cat([add_mask, tgt_mask], dim=2)

        tgt_embed = self.tgt_embed(tgt)
        tgt_embed = torch.cat([txt_mean.unsqueeze(dim=1), tgt_embed], dim=1)
        tgt_embed = self.norm(tgt_embed)

        return self.decoder(tgt_embed, hidden_states,  tgt_mask, embeddings)

    def _decode(self,  txt_mean,hidden_states,embeddings,  tgt, tgt_mask):
        b, l, _ = tgt_mask.size()
        add_mask = (torch.zeros([b, 1, l]) > 0).to(tgt_mask.device)
        tgt_mask = torch.cat([add_mask, tgt_mask], dim=1)
        add_mask = (torch.ones([b, l + 1, 1]) > 0).to(tgt_mask.device)
        tgt_mask = torch.cat([add_mask, tgt_mask], dim=2)

        tgt_embed = self.tgt_embed(tgt)
        tgt_embed = torch.cat([txt_mean.unsqueeze(dim=1), tgt_embed], dim=1)
        tgt_embed = self.norm(tgt_embed)

        return self.decoder(tgt_embed, hidden_states,  tgt_mask, embeddings)


class TextEncoder(nn.Module):
    def __init__(self, layer, N,seqembed):
        super(TextEncoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.d_model)
        self.seqembed = seqembed

    def forward(self,x,mask):
        x = self.seqembed(x)
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class G_Encoder(nn.Module):
    def __init__(self, layer,node_embed, N):
        super(G_Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.d_model)
        self.node_embed = node_embed


    def forward(self, x, mask=None):
        x = self.node_embed(x)
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class NodeEncoder(nn.Module):
    def __init__(self, layer, N):
        super(NodeEncoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.d_model)

    def forward(self, x, m, mask=None):
        for layer in self.layers:
            x = layer(x, m, mask)
        return self.norm(x)

class NodeEncoderLayer(nn.Module):
    def __init__(self, d_model, self_attn, feed_forward, dropout):
        super(NodeEncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(d_model, dropout), 2)
        self.d_model = d_model

    def forward(self, x, m, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, m, m, mask))
        return self.sublayer[1](x, self.feed_forward)



class SublayerConnection(nn.Module):
    def __init__(self, d_model, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.d_model)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(d_model, dropout), 2)
        self.d_model = d_model

    def forward(self, x,mask=None):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x,mask))
        return self.sublayer[1](x, self.feed_forward)

class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.d_model)

    def forward(self, x, hidden_states,  tgt_mask, embeddings):
        for layer in self.layers:
            x = layer(x, hidden_states, tgt_mask, embeddings)
        return self.norm(x)

class DecoderLayer(nn.Module):
    def __init__(self, d_model, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.d_model = d_model
        self.self_attn = self_attn
        self.src_attn = clones(src_attn,2)
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(d_model, dropout), 4)
    def forward(self, x, hidden_states,  tgt_mask, embeddings):
        m = hidden_states
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[2](x, lambda x: self.src_attn[1](x, embeddings, embeddings))
        x = self.sublayer[1](x, lambda x: self.src_attn[0](x, m, m))
        x = self.sublayer[3](x, self.feed_forward)

        return x

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        #self.ff = PositionwiseFeedForward(d_model,d_model*4)
        self.d_model = d_model

    def forward(self, x):
        #return self.ff(self.lut(x) * math.sqrt(self.d_model))
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class EncoderDecoder(AttModel):

    def make_text_encoder(self):
        c = copy.deepcopy
        attn = MultiHeadedAttention(self.num_heads, self.d_model)
        ff = PositionwiseFeedForward(self.d_model, self.d_ff, self.dropout)
        position = PositionalEncoding(self.d_model, self.dropout)
        text_encoder = TextEncoder(EncoderLayer(self.d_model, c(attn), c(ff), self.dropout), self.num_layers, nn.Sequential(Embeddings(self.d_model, self.vocab_size+1), c(position)))
        for p in text_encoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        return text_encoder

    def make_model(self, tgt_vocab):
        c = copy.deepcopy
        attn = MultiHeadedAttention(self.num_heads, self.d_model)
        ff = PositionwiseFeedForward(self.d_model, self.d_ff, self.dropout)
        position = PositionalEncoding(self.d_model, self.dropout)
        g_encoder = G_Encoder(EncoderLayer(self.d_model, c(attn), c(ff), self.dropout),c(position), 2)

        model = Transformer(
            g_encoder,
            NodeEncoder(NodeEncoderLayer(self.d_model, c(attn), c(ff), self.dropout), self.num_layers),
            Encoder(EncoderLayer(self.d_model, c(attn), c(ff), self.dropout), self.num_layers),
            Decoder(
                DecoderLayer(self.d_model, c(attn), c(attn), c(ff), self.dropout),
                self.num_layers),
            lambda x: x,
            nn.Sequential(self.embedding, c(position)))
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        return model

    def __init__(self, args, tokenizer):
        super(EncoderDecoder, self).__init__(args, tokenizer)
        self.args = args
        self.num_layers = args.num_layers
        self.d_model = args.d_model
        self.d_ff = args.d_ff
        self.num_heads = args.num_heads
        self.dropout = args.dropout

        tgt_vocab = self.vocab_size + 1
        self.embedding = Embeddings(self.d_model, tgt_vocab)

        self.norm1 = LayerNorm(self.d_model)
        self.ff = PositionwiseFeedForward(self.d_model, self.d_ff, self.dropout)
        self.do = nn.Dropout(self.dropout)

        self.norm2 = LayerNorm(self.d_model)
        self.ff2 = PositionwiseFeedForward(self.d_model, self.d_ff, self.dropout)
        self.do2 = nn.Dropout(self.dropout)

        self.text_sigma = nn.Parameter(torch.ones(512))
        self.image_sigma = nn.Parameter(torch.ones(512))
        nn.init.constant_(self.text_sigma, 1)
        nn.init.constant_(self.image_sigma, 1)

        self.model = self.make_model(tgt_vocab)
        self.text_encoder = self.make_text_encoder()
        self.logit = nn.Linear(args.d_model, tgt_vocab)
        self.logit1 = nn.Linear(args.d_model, tgt_vocab)

    def init_hidden(self, bsz):
        return []

    def _prepare_feature(self, fc_feats, att_feats, att_masks,embeddings,matrix):

        att_feats, seq, att_masks, seq_mask = self._prepare_feature_forward(att_feats, att_masks)
        embeddings = embeddings.repeat(att_feats.size(0), 1, 1)
        memory = self.model.encode(att_feats, att_masks)
        embeddings = self.model.node_encoder(att_feats[:, 1:, :], embeddings, None)

        image_mean = memory[:, 0, :].squeeze(dim=1)
        image_mean = self.Linear1(image_mean)
        image_mean1 = self.sample(image_mean, self.image_sigma)
        img_mean = self.Linear2(image_mean1)

        return embeddings, img_mean, memory[:, 1:, :], att_masks
    def _prepare_feature_forward(self, att_feats, att_masks=None, seq=None):
        att_feats, att_masks = self.clip_att(att_feats, att_masks)
        att_feats = pack_wrapper(self.att_embed, att_feats, att_masks)

        if att_masks is None:
            att_masks = att_feats.new_ones(att_feats.shape[:2], dtype=torch.long)
        att_masks = att_masks.unsqueeze(-2)

        if seq is not None:
            # crop the last one
            seq = seq[:, :-1]
            seq_mask = (seq.data > 0)
            seq_mask[:, 0] += True

            seq_mask = seq_mask.unsqueeze(-2)
            seq_mask = seq_mask & subsequent_mask(seq.size(-1)).to(seq_mask)
        else:
            seq_mask = None

        return att_feats, seq, att_masks, seq_mask

    def prepare_bert_mask(self,seq):
        bert_mask = (seq.data > 0)
        bert_mask[:, 0] += True
        mask = bert_mask
        bert_mask = bert_mask.unsqueeze(-2)
        bert_mask = bert_mask.repeat(1,seq.size(-1),1)
        return bert_mask,mask
    def sample(self,mean,deviation):
        mean1 = mean
        theta = mean1
        b_num, dim_num = theta.size()
        k = 7
        rand = torch.normal(0, 1, (b_num, k, dim_num)).to(device)
        rand = torch.mean(rand, 1)
        mean1 = 0.1 * rand * deviation + theta

        return mean1

    def KL_Divergence(self,img_mean, text_mean, text_sigma, image_sigma):
        n = img_mean.size(0)
        det = 1
        for i in range(512):
            det = det * (text_sigma[i] / (image_sigma[i] + 1e-25))
        residual = text_mean - img_mean
        divergence = 1 / 2 * (torch.sum(
            torch.diagonal((residual / ((0.1 * image_sigma.unsqueeze(dim=0)) ** 2) @ residual.t()), dim1=0, dim2=1)) +
                              +n * (torch.sum((text_sigma / image_sigma) ** 2) - torch.log((det) ** 2)))
        print('divergence loss:', divergence / n)
        return divergence / n


    def Linear1(self,x):
        return self.norm1(x+self.do(self.ff(x)))


    def Linear2(self,x):
        return self.norm2(x+self.do2(self.ff2(x)))


    def _forward(self, fc_feats, att_feats, embeddings,matrix,seq, att_masks=None):

        att_feats, seq, att_masks, seq_mask = self._prepare_feature_forward(att_feats, att_masks, seq)

        bert_mask, mask = self.prepare_bert_mask(seq)
        text_features = self.text_encoder(seq, bert_mask)

        embeddings = self.model.get_node_embedding(embeddings, matrix)
        embeddings = embeddings.repeat(att_feats.size(0), 1, 1)

        image_features = self.model.encode(att_feats, att_masks)
        embeddings = self.model.node_encoder(att_feats[:, 1:, :], embeddings, None)

        image_mean = image_features[:, 0, :].squeeze(dim=1)
        mask = (mask / (torch.sum(mask, dim=-1).unsqueeze(dim=-1) - 1)).unsqueeze(dim=-1)
        text_mean = torch.sum((text_features * mask)[:, 1:, :], dim=1)

        image_mean = self.Linear1(image_mean)
        divergence = self.KL_Divergence(image_mean, text_mean.detach(), self.text_sigma, self.image_sigma)

        text_mean1 = self.sample(text_mean, self.text_sigma)
        txt_mean = self.Linear2(text_mean1)

        out = self.model(txt_mean, image_features[:, 1:, :], embeddings, seq, seq_mask)
        out1 = self.logit1(out[:, 0, :].unsqueeze(dim=1))
        out2 = self.logit(out[:, 1:, :])
        out = torch.cat([out1, out2], dim=1)
        outputs = F.log_softmax(out, dim=-1)

        return outputs,divergence

    def core(self, it, fc_feats_ph, att_feats_ph, memory, state, mask):

        if len(state) == 0:
            ys = it.unsqueeze(1)
        else:
            ys = torch.cat([state[0][0], it.unsqueeze(1)], dim=1)

        embeddings = fc_feats_ph
        img_mean = att_feats_ph
        out = self.model._decode(img_mean,memory, embeddings, ys, subsequent_mask(ys.size(1)).to(memory.device))

        return out[:, -1], [ys.unsqueeze(0)]