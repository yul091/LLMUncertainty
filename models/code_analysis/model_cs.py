import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from preprocess.utils import count_parameters
from transformers import (
    RobertaPreTrainedModel, 
    RobertaConfig,
    GPT2Config,
    GPT2Model,
    GPT2PreTrainedModel,
    LlamaConfig,
    LlamaModel,
    LlamaPreTrainedModel,
)
from transformers.models.roberta.modeling_roberta import RobertaEmbeddings
from models.utils import count_parameters


def attention(starts, c, W, a, length, embedding_dim):
    x = torch.tanh(torch.bmm(W, c))  # B X H X T
    x = x.permute(0, 2, 1)
    z = torch.bmm(x, a).squeeze(2)  # B X T
    z = F.softmax(z, dim=1)   # B X T
    z = z.unsqueeze(2)  # B X T X 1
    x = x.permute(0, 2, 1)  # B X H X T
    v = torch.zeros(len(x), embedding_dim, device=starts.device)
    for i in range(len(x)):
        v[i] = torch.bmm(
            x[i:i+1, :, :length[i]], z[i:i+1, :length[i], :]
        ).squeeze(2)
    return v


class Code2Vec(nn.Module):
    def __init__(
        self, nodes_dim, paths_dim, embed_dim,
        output_dim, embed_vec, dropout=0.1, padding_index=1
    ):
        super().__init__()
        self.embedding_dim = embed_dim

        if torch.is_tensor(embed_vec):
            self.node_embedding = nn.Embedding(nodes_dim, embed_dim, padding_idx=padding_index, _weight=embed_vec)
            self.node_embedding.weight.requires_grad = False
        else:
            self.node_embedding = nn.Embedding(nodes_dim, embed_dim, padding_idx=padding_index)

        self.path_embedding = nn.Embedding(paths_dim, embed_dim)
        self.W = nn.Parameter(torch.randn(1, embed_dim, 3 * embed_dim))
        self.a = nn.Parameter(torch.randn(1, embed_dim, 1))
        self.out = nn.Linear(embed_dim, output_dim)
        self.drop = nn.Dropout(dropout)
        print('Created {} with {:,} params:\n{}'.format(
            self.__class__.__name__, count_parameters(self), self
        ))
        self.sub_num = [1]

    def forward(self, starts, paths, ends, length):
        """
        starts/paths/ends: Tensor of B X T.
        length: List(B) of each instance length.
        """
        W = self.W.repeat(len(starts), 1, 1) # (B X H X 3H)
        embedded_starts = self.node_embedding(starts)
        embedded_paths = self.path_embedding(paths)
        embedded_ends = self.node_embedding(ends) # (B X T X H)

        c = torch.cat((embedded_starts, embedded_paths, embedded_ends), dim=2)
        c = self.drop(c)
        c = c.permute(0, 2, 1)  # [B X 3H X T]
        x = torch.tanh(torch.bmm(W, c))  # [B X H X T]
        x = x.permute(0, 2, 1)
        a = self.a.repeat(len(starts), 1, 1)  # [B X H X 1]
        z = torch.bmm(x, a).squeeze(2)  # [B X T]
        z = F.softmax(z, dim=1)   # [B X T]
        z = z.unsqueeze(2)  # [B X T X 1]
        x = x.permute(0, 2, 1)  # [B X H X T]

        v = torch.zeros(len(x), self.embedding_dim, device=starts.device)
        for i in range(len(x)):
            v[i] = torch.bmm(
                x[i:i+1, :, :length[i]], z[i:i+1, :length[i], :]
            ).squeeze(2)
        #v = torch.bmm(x, z).squeeze(2)  # [B X H]
        out = self.out(v)  # [B X V]
        return out

    def get_hidden(self, starts, paths, ends, length):
        res = []
        # the model contrains embedding layer, attention layer, fc layer, we pick hidden state from the last two
        W = self.W.repeat(len(starts), 1, 1) # (B X H X 3H)
        embedded_starts = self.node_embedding(starts)
        embedded_paths = self.path_embedding(paths)
        embedded_ends = self.node_embedding(ends) # (B X T X H)

        c = torch.cat((embedded_starts, embedded_paths, embedded_ends), dim=2)
        c = self.drop(c)
        c = c.permute(0, 2, 1)  # [B X 3H X T]
        x = torch.tanh(torch.bmm(W, c))  # [B X H X T]
        x = x.permute(0, 2, 1)
        a = self.a.repeat(len(starts), 1, 1)  # [B X H X 1]
        z = torch.bmm(x, a).squeeze(2)  # [B X T]
        z = F.softmax(z, dim=1)   # [B X T]
        z = z.unsqueeze(2)  # [B X T X 1]
        x = x.permute(0, 2, 1)  # [B X H X T]

        v = torch.zeros(len(x), self.embedding_dim, device=starts.device)
        for i in range(len(x)):
            v[i] = torch.bmm(
                x[i:i+1, :, :length[i]], z[i:i+1, :length[i], :]
            ).squeeze(2)

        res.append(v.detach().cpu())
        return res



class BiLSTM2Vec(nn.Module):
    def __init__(self, nodes_dim, paths_dim, embed_dim,
                 output_dim, embed_vec, dropout=0.1, padding_index=1, num_layers=2):
        super().__init__()
        self.embedding_dim = embed_dim

        if torch.is_tensor(embed_vec):
            self.node_embedding = nn.Embedding(nodes_dim, embed_dim, padding_idx=padding_index, _weight=embed_vec)
            self.node_embedding.weight.requires_grad = False
        else:
            self.node_embedding = nn.Embedding(nodes_dim, embed_dim, padding_idx=padding_index)
        self.path_embedding = nn.Embedding(paths_dim, embed_dim)
        
        self.lstm = nn.LSTM(
            3*embed_dim, 3*embed_dim, num_layers=num_layers, 
            batch_first=True, dropout=dropout, bidirectional=True
        )
        self.activation = nn.Tanh()
        self.linear = nn.Linear(3*embed_dim, output_dim)
        print('Created {} with {:,} params:\n{}'.format(
            self.__class__.__name__, count_parameters(self), self
        ))
        self.sub_num = [1]

    def forward(self, starts, paths, ends, length):
        """
        starts/paths/ends: Tensor of B X T.
        length: List(B) of each instance length.
        """
        embedded_starts = self.node_embedding(starts)
        embedded_paths = self.path_embedding(paths)
        embedded_ends = self.node_embedding(ends) # (B X T X H)
        c = torch.cat((embedded_starts, embedded_paths, embedded_ends), dim=2) # B X T X 3H
        _, hidden = self.lstm(c) # B X T X 2*3H, 2*layers X B X 3H

        h, c = hidden
        last_hidden = h[-1] + c[-1] # B X 3H
        pooled_output = self.activation(last_hidden) # B X 3H
        pred = self.linear(pooled_output) # B X V
        return pred

    def get_hidden(self, starts, paths, ends, length):
        res = []
        embedded_starts = self.node_embedding(starts)
        embedded_paths = self.path_embedding(paths)
        embedded_ends = self.node_embedding(ends) # (B X T X H)
        c = torch.cat((embedded_starts, embedded_paths, embedded_ends), dim=2) # B X T X 3H
        _, hidden = self.lstm(c) # B X T X 2*3H, 2*layers X B X 3H

        h, c = hidden
        last_hidden = h[-1] + c[-1] # B X 3H
        res.append(last_hidden.detach().cpu())
        return res
    
        
class CodeBert2Vec(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]
    
    def __init__(
        self, 
        config: List[RobertaConfig], 
    ):
        super().__init__(config[0])
        config_node, config_path, config_concat = config
        self.embedding_dim = config_node.hidden_size
        self.node_embedding = RobertaEmbeddings(config_node)
        self.path_embedding = RobertaEmbeddings(config_path)

        self.W1 = nn.Parameter(
            torch.randn(1, config_node.hidden_size, config_concat.hidden_size),
            requires_grad=True, 
        )
        self.W2 = nn.Parameter(
            torch.randn(1, config_node.hidden_size, config_concat.hidden_size),
            requires_grad=True, 
        )
        self.a1 = nn.Parameter(
            torch.randn(1, config_node.hidden_size, 1),
            requires_grad=True,
        )
        self.a2 = nn.Parameter(
            torch.randn(1, config_node.hidden_size, 1),
            requires_grad=True,
        )
        self.out = nn.Linear(config_node.hidden_size, config_node.num_labels)
        self.drop = nn.Dropout(config_node.hidden_dropout_prob)
        print('Created {} with {:,} params:\n{}'.format(
            self.__class__.__name__, count_parameters(self), self
        ))
        self.sub_num = [1]
        self.init_weights()         


    def forward(self, starts, paths, ends, length):
        embedded_starts = self.node_embedding(starts) # B X T X H
        embedded_paths = self.path_embedding(paths)
        embedded_ends = self.node_embedding(ends)
        
        W1 = self.W1.repeat(len(starts), 1, 1) # B X H X 3H
        a1 = self.a1.repeat(len(starts), 1, 1)  # B X H X 1
        W2 = self.W2.repeat(len(starts), 1, 1) # B X H X 3H
        a2 = self.a2.repeat(len(starts), 1, 1)  # B X H X 1
        
        c = torch.cat((embedded_starts, embedded_paths, embedded_ends), dim=2)
        c = self.drop(c)
        c = c.permute(0, 2, 1)  # B X 3H X T
        v1 = attention(starts, c, W1, a1, length, self.embedding_dim)
        v2 = attention(starts, c, W2, a2, length, self.embedding_dim)
        v = v1 + v2 # B X 3H
            
        out = self.out(v)  # B X V
        return out


    def get_hidden(self, starts, paths, ends, length):
        res = []
        embedded_starts = self.node_embedding(starts) # B X T X H
        embedded_paths = self.path_embedding(paths)
        embedded_ends = self.node_embedding(ends)
        
        W1 = self.W1.repeat(len(starts), 1, 1) # B X H X 3H
        a1 = self.a1.repeat(len(starts), 1, 1)  # B X H X 1
        W2 = self.W2.repeat(len(starts), 1, 1) # B X H X 3H
        a2 = self.a2.repeat(len(starts), 1, 1)  # B X H X 1
        
        c = torch.cat((embedded_starts, embedded_paths, embedded_ends), dim=2)
        c = self.drop(c)
        c = c.permute(0, 2, 1)  # B X 3H X T
        v1 = attention(starts, c, W1, a1, length, self.embedding_dim)
        v2 = attention(starts, c, W2, a2, length, self.embedding_dim)
        v = v1 + v2 # B X 3H
        
        res.append(v.detach().cpu())
        return res

    

class CodeBerta2Vec(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]
    
    def __init__(
        self, 
        config: List[RobertaConfig], 
    ):
        super().__init__(config[0])
        config_node, config_path, config_concat = config
        self.embedding_dim = config_node.hidden_size
        self.node_embedding = RobertaEmbeddings(config_node)
        self.path_embedding = RobertaEmbeddings(config_path)

        self.W1 = nn.Parameter(
            torch.randn(1, config_node.hidden_size, config_concat.hidden_size),
            requires_grad=True, 
        )
        self.W2 = nn.Parameter(
            torch.randn(1, config_node.hidden_size, config_concat.hidden_size),
            requires_grad=True, 
        )
        self.a1 = nn.Parameter(
            torch.randn(1, config_node.hidden_size, 1),
            requires_grad=True,
        )
        self.a2 = nn.Parameter(
            torch.randn(1, config_node.hidden_size, 1),
            requires_grad=True,
        )
        self.out = nn.Linear(config_node.hidden_size, config_node.num_labels)
        self.drop = nn.Dropout(config_node.hidden_dropout_prob)
        print('Created {} with {:,} params:\n{}'.format(
            self.__class__.__name__, count_parameters(self), self
        ))
        self.sub_num = [1]
        self.init_weights()         


    def forward(self, starts, paths, ends, length):
        embedded_starts = self.node_embedding(starts) # B X T X H
        embedded_paths = self.path_embedding(paths)
        embedded_ends = self.node_embedding(ends)
        
        W1 = self.W1.repeat(len(starts), 1, 1) # B X H X 3H
        a1 = self.a1.repeat(len(starts), 1, 1)  # B X H X 1
        W2 = self.W2.repeat(len(starts), 1, 1) # B X H X 3H
        a2 = self.a2.repeat(len(starts), 1, 1)  # B X H X 1
        
        c = torch.cat((embedded_starts, embedded_paths, embedded_ends), dim=2)
        c = self.drop(c)
        c = c.permute(0, 2, 1)  # B X 3H X T
        v1 = attention(starts, c, W1, a1, length, self.embedding_dim)
        v2 = attention(starts, c, W2, a2, length, self.embedding_dim)
        v = v1 + v2 # B X 3H
            
        out = self.out(v)  # B X V
        return out


    def get_hidden(self, starts, paths, ends, length):
        res = []
        embedded_starts = self.node_embedding(starts) # B X T X H
        embedded_paths = self.path_embedding(paths)
        embedded_ends = self.node_embedding(ends)
        
        W1 = self.W1.repeat(len(starts), 1, 1) # B X H X 3H
        a1 = self.a1.repeat(len(starts), 1, 1)  # B X H X 1
        W2 = self.W2.repeat(len(starts), 1, 1) # B X H X 3H
        a2 = self.a2.repeat(len(starts), 1, 1)  # B X H X 1
        
        c = torch.cat((embedded_starts, embedded_paths, embedded_ends), dim=2)
        c = self.drop(c)
        c = c.permute(0, 2, 1)  # B X 3H X T
        v1 = attention(starts, c, W1, a1, length, self.embedding_dim)
        v2 = attention(starts, c, W2, a2, length, self.embedding_dim)
        v = v1 + v2 # B X 3H
        
        res.append(v.detach().cpu())
        return res

    

class GraphCodeBert2Vec(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]
    
    def __init__(self, config: List[RobertaConfig], dropout: float = 0.1):
        super().__init__(config[0])

        config_node, config_path, config_concat = config
        self.embedding_dim = config_node.hidden_size
        self.node_embedding = RobertaEmbeddings(config_node)
        self.path_embedding = RobertaEmbeddings(config_path)

        self.W1 = nn.Parameter(
            torch.randn(1, config_node.hidden_size, config_concat.hidden_size),
            requires_grad=True, 
        )
        self.W2 = nn.Parameter(
            torch.randn(1, config_node.hidden_size, config_concat.hidden_size),
            requires_grad=True, 
        )
        self.a1 = nn.Parameter(
            torch.randn(1, config_node.hidden_size, 1),
            requires_grad=True,
        )
        self.a2 = nn.Parameter(
            torch.randn(1, config_node.hidden_size, 1),
            requires_grad=True,
        )
        self.out = nn.Linear(2*config_node.hidden_size, config_node.num_labels)
        self.drop = nn.Dropout(config_node.hidden_dropout_prob)
        print('Created {} with {:,} params:\n{}'.format(
            self.__class__.__name__, count_parameters(self), self
        ))
        self.sub_num = [1]
        self.init_weights()
            
    
    def forward(self, starts, paths, ends, length):
        embedded_starts = self.node_embedding(starts) # B X T X H
        embedded_paths = self.path_embedding(paths)
        embedded_ends = self.node_embedding(ends)
        
        W1 = self.W1.repeat(len(starts), 1, 1) # B X H X 3H
        a1 = self.a1.repeat(len(starts), 1, 1)  # B X H X 1
        W2 = self.W2.repeat(len(starts), 1, 1) # B X H X 3H
        a2 = self.a2.repeat(len(starts), 1, 1)  # B X H X 1
        
        c = torch.cat((embedded_starts, embedded_paths, embedded_ends), dim=2)
        c = self.drop(c)
        c = c.permute(0, 2, 1)  # B X 3H X T
        v1 = attention(starts, c, W1, a1, length, self.embedding_dim)
        v2 = attention(starts, c, W2, a2, length, self.embedding_dim)
        v = torch.cat((v1, v2), dim=1) # B X 2H
            
        out = self.out(v)  # B X V
        return out


    def get_hidden(self, starts, paths, ends, length):

        res = []
        embedded_starts = self.node_embedding(starts) # B X T X H
        embedded_paths = self.path_embedding(paths)
        embedded_ends = self.node_embedding(ends)
        
        W1 = self.W1.repeat(len(starts), 1, 1) # B X H X 3H
        a1 = self.a1.repeat(len(starts), 1, 1)  # B X H X 1
        W2 = self.W2.repeat(len(starts), 1, 1) # B X H X 3H
        a2 = self.a2.repeat(len(starts), 1, 1)  # B X H X 1
        
        c = torch.cat((embedded_starts, embedded_paths, embedded_ends), dim=2)
        c = self.drop(c)
        c = c.permute(0, 2, 1)  # B X 3H X T
        v1 = attention(starts, c, W1, a1, length, self.embedding_dim)
        v2 = attention(starts, c, W2, a2, length, self.embedding_dim)
        v = torch.cat((v1, v2), dim=1) # B X 2H
        res.append(v.detach().cpu())
        
        return res

    
    
class CodeGPT2Vec(GPT2PreTrainedModel):
    def __init__(self, config: List[GPT2Config], dropout: float = 0.1):
        super().__init__(config[0])
        config_node, config_path, config_concat = config
        self.embedding_dim = config_node.n_embd
        self.node_model = GPT2Model(config_node)
        self.path_model = GPT2Model(config_path)
        
        self.W1 = nn.Parameter(
            torch.randn(1, config_node.n_embd, config_concat.n_embd),
            requires_grad=True, 
        )
        self.W2 = nn.Parameter(
            torch.randn(1, config_node.n_embd, config_concat.n_embd),
            requires_grad=True, 
        )
        self.a1 = nn.Parameter(
            torch.randn(1, config_node.n_embd, 1),
            requires_grad=True,
        )
        self.a2 = nn.Parameter(
            torch.randn(1, config_node.n_embd, 1),
            requires_grad=True,
        )
        self.out = nn.Linear(2*config_node.n_embd, config_concat.num_labels)
        self.drop = nn.Dropout(config_node.embd_pdrop)
        print('Created {} with {:,} params:\n{}'.format(
            self.__class__.__name__, count_parameters(self), self
        ))
        self.sub_num = [1]
        self.init_weights()

    def forward(self, starts, paths, ends, length):
        embedded_starts = self.node_model(starts)[0] # B X T X H
        embedded_paths = self.path_model(paths)[0]
        embedded_ends = self.node_model(ends)[0]

        W1 = self.W1.repeat(len(starts), 1, 1) # B X H X H
        a1 = self.a1.repeat(len(starts), 1, 1)  # B X H X 1
        W2 = self.W2.repeat(len(starts), 1, 1) # B X H X H
        a2 = self.a2.repeat(len(starts), 1, 1)  # B X H X 1

        c = torch.cat((embedded_starts, embedded_paths, embedded_ends), dim=2)
        c = self.drop(c)
        c = c.permute(0, 2, 1)  # B X 3H X T
        v1 = attention(starts, c, W1, a1, length, self.embedding_dim)
        v2 = attention(starts, c, W2, a2, length, self.embedding_dim)
        v = torch.cat((v1, v2), dim=1) # B X 2H
        out = self.out(v)  # B X V
        return out
    
    def get_hidden(self, starts, paths, ends, length):
        res = []
        embedded_starts = self.node_model(starts)[0] # B X T X H
        embedded_paths = self.path_model(paths)[0]
        embedded_ends = self.node_model(ends)[0]

        W1 = self.W1.repeat(len(starts), 1, 1) # B X H X H
        a1 = self.a1.repeat(len(starts), 1, 1)  # B X H X 1
        W2 = self.W2.repeat(len(starts), 1, 1) # B X H X H
        a2 = self.a2.repeat(len(starts), 1, 1)  # B X H X 1

        c = torch.cat((embedded_starts, embedded_paths, embedded_ends), dim=2)
        c = self.drop(c)
        c = c.permute(0, 2, 1)  # B X 3H X T
        v1 = attention(starts, c, W1, a1, length, self.embedding_dim)
        v2 = attention(starts, c, W2, a2, length, self.embedding_dim)
        v = torch.cat((v1, v2), dim=1) # B X 2H
        res.append(v.detach().cpu())
        return res
    
    
    
class CodeLlama2Vec(LlamaPreTrainedModel):
    def __init__(self, config: List[LlamaConfig], dropout: float = 0.1):
        super().__init__(config[0])
        config_node, config_path, config_concat = config
        self.embedding_dim = config_node.hidden_size
        self.node_model = LlamaModel(config_node)
        self.path_model = LlamaModel(config_path)
        
        self.W1 = nn.Parameter(
            torch.randn(1, config_node.hidden_size, config_concat.hidden_size),
            requires_grad=True, 
        )
        self.W2 = nn.Parameter(
            torch.randn(1, config_node.hidden_size, config_concat.hidden_size),
            requires_grad=True, 
        )
        self.a1 = nn.Parameter(
            torch.randn(1, config_node.hidden_size, 1),
            requires_grad=True,
        )
        self.a2 = nn.Parameter(
            torch.randn(1, config_node.hidden_size, 1),
            requires_grad=True,
        )
        self.out = nn.Linear(2*config_node.hidden_size, config_concat.num_labels)
        self.drop = nn.Dropout(dropout)
        print('Created {} with {:,} params:\n{}'.format(
            self.__class__.__name__, count_parameters(self), self
        ))
        self.sub_num = [1]
        self.init_weights()

    def forward(self, starts, paths, ends, length):
        embedded_starts = self.node_model(starts)[0] # B X T X H
        embedded_paths = self.path_model(paths)[0]
        embedded_ends = self.node_model(ends)[0]

        W1 = self.W1.repeat(len(starts), 1, 1) # B X H X H
        a1 = self.a1.repeat(len(starts), 1, 1)  # B X H X 1
        W2 = self.W2.repeat(len(starts), 1, 1) # B X H X H
        a2 = self.a2.repeat(len(starts), 1, 1)  # B X H X 1

        c = torch.cat((embedded_starts, embedded_paths, embedded_ends), dim=2)
        c = self.drop(c)
        c = c.permute(0, 2, 1)  # B X 3H X T
        v1 = attention(starts, c, W1, a1, length, self.embedding_dim)
        v2 = attention(starts, c, W2, a2, length, self.embedding_dim)
        v = torch.cat((v1, v2), dim=1) # B X 2H
        out = self.out(v)  # B X V
        return out
    
    def get_hidden(self, starts, paths, ends, length):
        res = []
        embedded_starts = self.node_model(starts)[0] # B X T X H
        embedded_paths = self.path_model(paths)[0]
        embedded_ends = self.node_model(ends)[0]

        W1 = self.W1.repeat(len(starts), 1, 1) # B X H X H
        a1 = self.a1.repeat(len(starts), 1, 1)  # B X H X 1
        W2 = self.W2.repeat(len(starts), 1, 1) # B X H X H
        a2 = self.a2.repeat(len(starts), 1, 1)  # B X H X 1

        c = torch.cat((embedded_starts, embedded_paths, embedded_ends), dim=2)
        c = self.drop(c)
        c = c.permute(0, 2, 1)  # B X 3H X T
        v1 = attention(starts, c, W1, a1, length, self.embedding_dim)
        v2 = attention(starts, c, W2, a2, length, self.embedding_dim)
        v = torch.cat((v1, v2), dim=1) # B X 2H
        res.append(v.detach().cpu())
        return res