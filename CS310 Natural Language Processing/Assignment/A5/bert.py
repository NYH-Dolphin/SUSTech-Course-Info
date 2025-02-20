import math
from typing import List, Dict
import numpy as np
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device('cpu')


n_layers = 1 # number of Encoder of Encoder Layer
n_heads = 12 # number of heads in Multi-Head Attention
d_model = 768 # Embedding Size
d_ff = 768 * 4  # 4*d_model, FeedForward dimension
d_k = d_v = 64  # dimension of K(=Q), V
n_segments = 2 # number of segments, ex) sentence A and sentence B

class Embedding(nn.Module):
    # MAX_LEN: maximum of width of a batch
    def __init__(self, VOCAB_SIZE, MAX_LEN):
        super(Embedding, self).__init__()
        self.tok_embed = nn.Embedding(VOCAB_SIZE, d_model, padding_idx=0)  # token embedding
        self.pos_embed = nn.Embedding(MAX_LEN, d_model)  # position embedding
        self.seg_embed = nn.Embedding(n_segments + 1, d_model, padding_idx=0)  # segment(token type) embedding
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, seg): #  input_ids, segment_ids, masked_pos
        seq_len = x.size(1)
        pos = torch.arange(seq_len, dtype=torch.long).to(device)
        pos = pos.unsqueeze(0).expand_as(x)  # (seq_len,) -> (batch_size, seq_len)
        embedding = self.tok_embed(x) + self.pos_embed(pos) + self.seg_embed(seg)
        return self.norm(embedding)
    


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k) # scores : [batch_size x n_heads x seq_len x seq_len]
        scores.masked_fill_(attn_mask, -1e9) # Fills elements of self tensor with value where mask is one.
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context, attn

class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads)
        self.W_K = nn.Linear(d_model, d_k * n_heads)
        self.W_V = nn.Linear(d_model, d_v * n_heads)
        self.fc = nn.Linear(n_heads * d_v, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, attn_mask):
        # x: [batch_size x seq_len x d_model]
        residual, batch_size = x, x.size(0)
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q_s = self.W_Q(x).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # q_s: [batch_size x n_heads x seq_len x d_k]
        k_s = self.W_K(x).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # k_s: [batch_size x n_heads x seq_len x d_k]
        v_s = self.W_V(x).view(batch_size, -1, n_heads, d_v).transpose(1,2)  # v_s: [batch_size x n_heads x seq_len x d_v]

        # repeat: repeat n_heads times along the 2nd dimension, for n attention heads
        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1) # attn_mask : [batch_size x n_heads x seq_len x seq_len]

        # context: [batch_size x n_heads x seq_len x d_v], attn: [batch_size x n_heads x seq_len x seq_len]
        context, attn = ScaledDotProductAttention()(q_s, k_s, v_s, attn_mask)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, n_heads * d_v) # context: [batch_size x seq_len x (n_heads * d_v)]
        output = self.fc(context) # nn.Linear(input_dim, output_dim)(input)
        return self.norm(output + residual), attn # output: [batch_size x seq_len x d_model]



# Activation function: GELU (Gaussian Error Linear Unit)
def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_ff) -> (batch_size, seq_len, d_model)
        return self.fc2(gelu(self.fc1(x)))
    

class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, enc_inputs, enc_self_attn_mask):
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_self_attn_mask)
        enc_outputs = self.pos_ffn(enc_outputs)
        return enc_outputs, attn
    
# convert the [PAD] to 1 and elsewhere 0. 
def get_pad_attn_mask(seq_q, seq_k):
    batch_size, seq_len = seq_q.size()
    batch_size, len_k = seq_k.size()
    # eq(zero) is PAD token
    # 与 seq_k 大小相同的布尔张量, 在 seq_k 中查找等于零的位置
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # batch_size x 1 x len_k(=seq_len), one is masking
    # expand: 在第二个维度（seq_len）重复掩码 seq_len
    return pad_attn_mask.expand(batch_size, seq_len, len_k)  # batch_size x seq_len x len_k

class BERT(nn.Module):
    def __init__(self, VOCAB_SIZE, MAX_LEN):
        super(BERT, self).__init__()
        self.embedding = Embedding(VOCAB_SIZE,MAX_LEN)
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])

         # for NSP task
        self.fc1 = nn.Linear(d_model, d_model)
        self.activ1 = nn.Tanh()
        self.classifier = nn.Linear(d_model, 2)

        # for MLM task
        self.fc2 = nn.Linear(d_model, d_model) # unembeddings
        self.activ2 = gelu
        self.norm = nn.LayerNorm(d_model)
        embed_weight = self.embedding.tok_embed.weight
        n_vocab, n_dim = embed_weight.size()
        self.decoder = nn.Linear(n_dim, n_vocab, bias=False)
        self.decoder.weight = embed_weight # decoder is shared with embedding layer
        self.decoder_bias = nn.Parameter(torch.zeros(n_vocab))

    def forward(self, input_ids, segment_ids, masked_pos):
        output = self.embedding(input_ids, segment_ids)
        enc_self_attn_mask = get_pad_attn_mask(input_ids, input_ids)
        for layer in self.layers:
            output, enc_self_attn = layer(output, enc_self_attn_mask)
            # output : [batch_size, len, d_model], attn : [batch_size, n_heads, d_model, d_model]

        # Use the representation of [CLS] to produce logits for NSP task
        ### START YOUR CODE ###
        logits_clsf = self.classifier(self.activ1(self.fc1(output[:, 0])))
        ### END YOUR CODE ###

        # Gather the representations of masked tokens to produce logits for MLM task
        ### START YOUR CODE ###
        masked_pos = masked_pos[:, :, None].expand(-1, -1, output.size(-1)) # [batch_size, max_pred, d_model]
        # 按照 masked_pos 的索引从 output 中提取对应的表示
        h_masked = torch.gather(output, 1, masked_pos) # masking position [batch_size, max_pred, d_model]
        logits_lm = self.decoder(self.norm(self.activ2(self.fc2(h_masked)))) + self.decoder_bias
        ### END YOUR CODE ###

        return logits_lm, logits_clsf