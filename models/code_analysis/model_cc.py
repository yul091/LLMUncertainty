import torch
import torch.nn as nn
from transformers import (
    RobertaModel, 
    RobertaPreTrainedModel,
    GPT2Model,
    GPT2PreTrainedModel,
    LlamaModel, 
    LlamaPreTrainedModel,
)
from models.utils import count_parameters


class Code2vecForClassification(nn.Module):
    def __init__(self, vocab_size, token_vec, hidden_size=120):
        super(Code2vecForClassification, self).__init__()
        if torch.is_tensor(token_vec):
            self.encoder = nn.Embedding(vocab_size, hidden_size, padding_idx=0, _weight=token_vec)
            self.encoder.weight.requires_grad = False
        else:
            self.encoder = nn.Embedding(vocab_size, hidden_size, padding_idx=0)

        self.linear = nn.Linear(hidden_size, vocab_size)
        self.drop = nn.Dropout(0.1)
        print('Created {} with {:,} params:\n{}'.format(
            self.__class__.__name__, count_parameters(self), self
        ))
        self.sub_num = [1]

    def forward(self, x): # B X T
        vec = self.encoder(x) # B X T X H
        vec = torch.mean(vec, dim=1) # B X H
        vec = self.drop(torch.tanh(vec)) # B X H
        pred = self.linear(vec) # B X V
        return pred

    def get_hidden(self, x):
        res = []
        vec = self.encoder(x)
        vec = torch.mean(vec, dim=1)
        vec = self.drop(torch.tanh(vec))
        res.append(vec.detach().cpu())
        return res


class BiLSTMForClassification(nn.Module):
    def __init__(self, vocab_size, token_vec, hidden_size=120, num_layers=2, dropout=0.1):
        super().__init__()
        if torch.is_tensor(token_vec):
            self.encoder = nn.Embedding(vocab_size, hidden_size, padding_idx=0, _weight=token_vec)
            self.encoder.weight.requires_grad = False
        else:
            self.encoder = nn.Embedding(vocab_size, hidden_size, padding_idx=0)

        self.lstm = nn.LSTM(
            hidden_size, hidden_size, num_layers=num_layers, 
            batch_first=True, dropout=dropout, bidirectional=True
        )
        self.activation = nn.Tanh()
        self.linear = nn.Linear(hidden_size, vocab_size)
        print('Created {} with {:,} params:\n{}'.format(
            self.__class__.__name__, count_parameters(self), self
        ))
        self.sub_num = [1]
        
    def forward(self, x):
        embed = self.encoder(x) # B X T X H
        _, hidden = self.lstm(embed) # 2*num_layers X B X H
        h, c = hidden
        last_hidden = h[-1] + c[-1] # B X H
        pooled_output = self.activation(last_hidden) # B X H
        pred = self.linear(pooled_output) # B X V
        
        return pred

    def get_hidden(self, x):
        res = []
        embed = self.encoder(x) # B X T X H
        _, hidden = self.lstm(embed) # 2*num_layers X B X H
        h, c = hidden
        last_hidden = h[-1] + c[-1] # B X H
        pooled_output = self.activation(last_hidden) # B X H
        res.append(pooled_output.detach().cpu())
        
        return res


class CodeBertForClassification(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]
    
    def __init__(self, config, dropout=0.1):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.roberta = RobertaModel(config)
        
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.sub_num = [1]
        self.init_weights()
        print('Created {} with {:,} params:\n{}'.format(
            self.__class__.__name__, count_parameters(self), self
        ))
            
    def forward(self, input_ids):
        outputs = self.roberta(input_ids)
        pooled_output = outputs['pooler_output'] # B X H
        pooled_output = self.dropout(pooled_output) # B X H
        pred = self.classifier(pooled_output) # B X V
        return pred

    def get_hidden(self, x):
        res = []
        outputs = self.roberta(x)
        pooled_output = outputs['pooler_output'] # B X H
        res.append(pooled_output.detach().cpu())
        # pooled_output = self.dropout(pooled_output) # B X H
        return res
    
    
class GraphCodeBertForClassification(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]
    
    def __init__(self, config, dropout=0.1):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.roberta = RobertaModel(config)
        
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.sub_num = [1]
        self.init_weights()
        print('Created {} with {:,} params:\n{}'.format(
            self.__class__.__name__, count_parameters(self), self
        ))
            
    def forward(self, input_ids):
        outputs = self.roberta(input_ids)
        pooled_output = outputs['pooler_output'] # B X H
        pooled_output = self.dropout(pooled_output) # B X H
        pred = self.classifier(pooled_output) # B X V
        return pred

    def get_hidden(self, x):
        res = []
        outputs = self.roberta(x)
        pooled_output = outputs['pooler_output'] # B X H
        res.append(pooled_output.detach().cpu())
        # pooled_output = self.dropout(pooled_output) # B X H
        return res
    
    
class CodeBertaForClassification(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]
    
    def __init__(self, config, dropout=0.1):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.roberta = RobertaModel(config)
        
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.sub_num = [1]
        self.init_weights()
        print('Created {} with {:,} params:\n{}'.format(
            self.__class__.__name__, count_parameters(self), self
        ))
            
    def forward(self, input_ids):
        outputs = self.roberta(input_ids)
        pooled_output = outputs['pooler_output'] # B X H
        pooled_output = self.dropout(pooled_output) # B X H
        pred = self.classifier(pooled_output) # B X V
        return pred

    def get_hidden(self, x):
        res = []
        outputs = self.roberta(x)
        pooled_output = outputs['pooler_output'] # B X H
        res.append(pooled_output.detach().cpu())
        # pooled_output = self.dropout(pooled_output) # B X H
        return res
    
    
class CodeGPTForClassification(GPT2PreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"h\.\d+\.attn\.masked_bias", r"lm_head.weight"]
    
    def __init__(self, config, dropout: float = 0.1):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.gpt2 = GPT2Model(config)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(config.n_embd, self.num_labels, bias=False)
        self.sub_num = [1]
        print('Created {} with {:,} params:\n{}'.format(
            self.__class__.__name__, count_parameters(self), self
        ))
        
        # Initialize weights and apply final processing
        self.init_weights()
    
    def forward(self, input_ids: torch.LongTensor):
        outputs = self.gpt2(input_ids)
        hidden_states = outputs['last_hidden_state'] # B X T X H
        hidden_states = self.dropout(hidden_states) # B X T X H
        logits = self.classifier(hidden_states) # B X T X V
        batch_size, sequence_length = input_ids.shape[:2]
        
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                sequence_lengths = (torch.ne(input_ids, self.config.pad_token_id).sum(-1) - 1).to(logits.device)
            else:
                sequence_lengths = -1

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]
        return pooled_logits

    def get_hidden(self, x: torch.LongTensor):
        res = []
        outputs = self.gpt2(x)
        hidden_states = outputs['last_hidden_state'] # B X T X H
        res.append(hidden_states[:, -1, :].detach().cpu())
        # hidden_states = self.dropout(hidden_states)
        return res
    
    
    
class CodeLlamaForClassification(LlamaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = LlamaModel(config)
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)
        self.sub_num = [1, 3, 5, 7, 9]
        print('Created {} with {:,} params:\n{}'.format(
            self.__class__.__name__, count_parameters(self), self
        ))
        # Initialize weights and apply final processing
        self.post_init()
        
    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor,
    ):
        batch_size = input_ids.shape[0]
        # if self.config.pad_token_id is None and batch_size != 1:
        #     raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            sequence_lengths = (torch.eq(input_ids, self.config.pad_token_id).long().argmax(-1) - 1).to(
                input_ids.device
            )
        
        transformer_outputs = self.model(input_ids)
        hidden_states = transformer_outputs[0] # B X T X H
        logits = self.score(hidden_states) # B X T X V

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths] # B X V
        # print("output shape: ", pooled_logits.shape)
        return pooled_logits
    
    
    def get_hidden(self, input_ids: torch.LongTensor,):
        res = []
        batch_size = input_ids.shape[0]
        # if self.config.pad_token_id is None and batch_size != 1:
        #     raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            sequence_lengths = (torch.eq(input_ids, self.config.pad_token_id).long().argmax(-1) - 1).to(
                input_ids.device
            )
        
        # Get the 1, 3, 5, 7, 9-th layer hidden states
        transformer_outputs = self.model(input_ids, output_hidden_states=True)
        hidden_states = transformer_outputs['hidden_states'] # sub_models X B X T X H
        for i in range(len(self.sub_num)):
            hidden_state = hidden_states[self.sub_num[i]+1] # B X T X H
            hidden_state = hidden_state[torch.arange(batch_size, device=hidden_state.device), sequence_lengths] # B X H
            res.append(hidden_state.detach().cpu())
        
        return res