import torch
from torch.nn import Module, Linear, MSELoss, ModuleList, Conv1d, Dropout, LayerNorm, parameter, GELU
import torch.nn.functional as F
import math
import numpy as np


def do_attention(query,key,value, mask= None):
    # get scaled attention scores
    attention_scores = torch.bmm(query, key.transpose(1,2))/math.sqrt(query.size(-1))
    if mask is not None:
        attention_scores = attention_scores.masked_fill(mask==0, float(1e-10))
    
    attention_weights = F.softmax(attention_scores,dim=1)
    return torch.bmm(attention_weights, value)    


def get_positional_encoding(seq_len, dim):
    
    positional_embed = torch.zeros((seq_len, dim))
    
    for t in range(seq_len):
        for i in range(dim//2):
            positional_embed[t,2*i] = np.sin(t/10000**(2*i/dim))
            positional_embed[t,2*i+1] = np.cos(t/10000**(2*i/dim))
    return positional_embed
                                             

class AttentionHead(Module):
    
    def __init__(self, embed_dim, head_dim, mask=None) -> None:
        super().__init__()
        self.mask = mask
        self.Wq = Linear(embed_dim, head_dim)
        self.Wk = Linear(embed_dim, head_dim)
        self.Wv = Linear(embed_dim, head_dim)
    
    def forward(self, h):
        q = self.Wq(h)
        k = self.Wk(h)
        v = self.Wv(h)
        outputs = do_attention(q,k,v, self.mask)
        return outputs
        
        
class MultiHeadAttention(Module):
    
    def __init__(self, hidden_size, num_heads, mask=None) -> None:
        super().__init__()
        num_heads = num_heads
        # by convention
        head_dim = hidden_size // num_heads
        self.heads = ModuleList(
            [AttentionHead(hidden_size, head_dim) for _ in range(num_heads)]
        )
        
        self.output = Linear(num_heads*head_dim, hidden_size)
        
    def forward(self, h):
        x = torch.cat([head(h) for head in self.heads], dim = -1)
        return self.output(x) 

    
class FeedForward(Module):
    
    # rule of thumb: hidden size of first layer 4x emebddding dimension
     def __init__(self,hidden_size, inter_size, dropout_prob = 0.3) -> None:
        super().__init__()
        self.conv1 = Linear(hidden_size, inter_size)
        self.conv2 = Linear(inter_size, hidden_size)
        # standard to use gelu
        self.gelu = GELU()
        self.dropout = Dropout(dropout_prob)
    
     def forward(self,x):
        x = self.conv1(x)
        x = self.gelu(x)
        x = self.conv2(x)
        return self.dropout(x)
   
    
    
class TransformerEncoderLayer(Module):
    
    def __init__(self, hidden_size, intermediate_size, output_size, num_heads, dropout_prob=0.3) -> None:
        super().__init__()
        # layer norm is prefered for transformer
        self.layer_norm1 = LayerNorm(hidden_size)
        self.layer_norm2 = LayerNorm(hidden_size)
        self.attention = MultiHeadAttention(hidden_size, num_heads)
        self.ff = FeedForward(hidden_size ,intermediate_size,dropout_prob)
        self.out = Linear(hidden_size, output_size)
        
    
    def forward(self, x):
        hidden = self.layer_norm1(x)
        #skip connection as in resnet
        x = x + self.attention(hidden)
        # skip connection
        x = x + self.ff(self.layer_norm2(x))
        # skip connection
        return self.out(x)
    
            
class Embedding(Module):
    
    def __init__(self, dim, dropout_prob= 0.3) -> None:
        super().__init__()
        
        self.layer_norm = LayerNorm(dim)
        self.dropout = Dropout(dropout_prob)
        self.linear =  Linear(dim, dim)
        
        
    def forward(self, x):
        x = self.layer_norm(x)
        x = self.linear(x)
        x = self.dropout(x)
        x = torch.relu(x)
        return x
        
    
class TransformerEncoder(Module):
    
    def __init__(self, num_hidden, hidden_size, intermediate_size, output_size, 
                         num_heads, seq_len = 0,  dropout_prob=0.3) -> None:
        
        super().__init__()
       # self.seq_len = seq_len
        self.dim = hidden_size
        self.embed = Embedding(self.dim,dropout_prob)

        self.hidden_dim = hidden_size
        #self.time_embedding = Time2Vec(seq_len)
        self.layers = ModuleList(
            [TransformerEncoderLayer(hidden_size, intermediate_size, hidden_size 
                                     ,num_heads, dropout_prob)
                                 for _ in range(num_hidden-1)]
                                 )
        self.layers.append(TransformerEncoderLayer(hidden_size, intermediate_size, output_size
                                 ,num_heads, dropout_prob))
        
    def forward(self, x):

        positional = get_positional_encoding(x.shape[1], self.dim)
        x = self.embed(x) + positional
        for layer in self.layers:
            #print(x.shape)
            x = layer(x)
        return x
    
class TransformerForPrediction(Module):
    
    def __init__(self, encoder: TransformerEncoder, dropout_prob = 0.3) -> None:
        super(TransformerForPrediction, self).__init__()
        self.encoder = encoder
        self.dropout = Dropout(dropout_prob)
        self.lin1 = Linear(encoder.hidden_dim, encoder.hidden_dim)
        self.out = Linear(encoder.hidden_dim, 1)

        
    def forward(self, x):
        
        x = self.encoder(x)
        x = self.dropout(x)
        x = torch.mean(x, dim=1) 
        x = self.lin1(x)
        x = torch.relu(x)
        return self.out(x)

class TransformerForBinaryClassification(Module):
    
    def __init__(self,encoder: TransformerEncoder, dropout_prob = 0.3) -> None:
        super().__init__()
        self.encoder = encoder
        self.dropout = Dropout(dropout_prob)
        self.l1 = Linear(encoder.hidden_dim,encoder.hidden_dim)
        self.gelu = GELU()
        self.out = Linear(encoder.hidden_dim,1)
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.l1(x)
        x = self.gelu(x)
        return torch.sigmoid(self.out(x))
        
