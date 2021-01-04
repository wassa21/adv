import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
import torch.optim as optim
import pdb
from transformers import *


class OLD_GenderPredictor_BERT_SUM(nn.Module):
    def __init__(self, input_size, hidden_size, vocab_size, dropout, output_size=2):
        super(OLD_GenderPredictor_BERT_SUM, self).__init__()
        self.input_size   = input_size
        self.hidden_size  = hidden_size
        self.output_size  = output_size
        self.bert         = BertModel.from_pretrained('bert-base-cased')
        self.outlin       = nn.Linear(self.input_size,self.output_size)
        self.dropout      =nn.Dropout(p=dropout)
        self.loss_func    = nn.CrossEntropyLoss()
        

    def forward(self,input_ids=None, attention_mask=None, token_type_ids=None, labels=None):
        # input_ids[0].shape number of sections in first document in batch x max len
        i = 0
        a,pooled_output = self.bert(input_ids[i],
                                    attention_mask=attention_mask[i],
                                    token_type_ids=token_type_ids[i])

        out = torch.sum(pooled_output,dim=0,keepdim=True)
        out = self.outlin(self.dropout(out))
        lss = self.loss_func(out,labels)
        return (out,lss)

    
class OLD_GenderPredictor_BERT_SUM_MLP(nn.Module):
    def __init__(self, input_size, hidden_size, vocab_size, dropout, output_size=2, bert_model_name='bert-base-cased'):
        super(OLD_GenderPredictor_BERT_SUM_MLP, self).__init__()
        self.input_size   = input_size
        self.hidden_size  = hidden_size
        self.output_size  = output_size
        self.bert         = BertModel.from_pretrained(bert_model_name)
        self.outlin       = nn.Linear(300,self.output_size)
        self.outlin1       = nn.Linear(self.input_size,300)
        self.activation   = nn.ReLU()
        self.dropout      =nn.Dropout(p=dropout)
        self.loss_func    = nn.CrossEntropyLoss()
        

    def forward(self,input_ids=None, attention_mask=None, token_type_ids=None, labels=None):
        # input_ids[0].shape number of sections in first document in batch x max len
        i = 0
        a,pooled_output = self.bert(input_ids[i],
                                    attention_mask=attention_mask[i],
                                    token_type_ids=token_type_ids[i])
        out = torch.sum(pooled_output,dim=0,keepdim=True)
        out = self.dropout(self.activation(self.outlin1(out)))
        out = self.outlin(out)
        lss = self.loss_func(out,labels)
        return (out,lss)


class OLD_GenderPredictor_BERT_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, vocab_size,
                 dropout, layer_num=1, output_size=2):
        super(OLD_GenderPredictor_BERT_LSTM, self).__init__()
        self.input_size   = input_size
        self.hidden_size  = hidden_size
        self.output_size  = output_size
        self.bert         = BertModel.from_pretrained('bert-base-cased')
        self.rnn          = nn.LSTM(self.input_size, self.hidden_size,
                                    num_layers=layer_num,dropout=dropout)
        self.outlin       = nn.Linear(self.hidden_size,self.output_size)
        self.dropout      =nn.Dropout(p=dropout)
        self.loss_func    = nn.CrossEntropyLoss()
            
    def forward(self,input_ids=None, attention_mask=None, token_type_ids=None, labels=None):
        # input_ids[0].shape number of sections in first document in batch x max len
        i = 0
        a,pooled_output = self.bert(input_ids[i],
                                    attention_mask=attention_mask[i],
                                    token_type_ids=token_type_ids[i])
        lstm_input = torch.reshape(pooled_output,
                                   (pooled_output.shape[0],1,pooled_output.shape[1]))
        output,(h_n,c_n) = self.rnn(lstm_input)
        out = self.outlin(self.dropout(h_n[-1]))
        lss = self.loss_func(out,labels)
        return (out,lss)



from reverse_gradient_layer import ReverseLayerF

class GenderPredictor_BERT_SUM_MLP(nn.Module):
    def __init__(self, input_size, hidden_size, vocab_size, dropout, output_size=2,bert_model_name='bert-base-cased'):
        super(GenderPredictor_BERT_SUM_MLP, self).__init__()
        self.input_size   = input_size
        self.hidden_size  = hidden_size
        self.output_size  = output_size
        self.bert_model_name = bert_model_name
        self.bert = AutoModel.from_pretrained(bert_model_name)
        self.bert          = torch.nn.DataParallel(self.bert)
        # For gender classification
        self.outlin        = nn.Linear(300,self.output_size)
        self.outlin1       = nn.Linear(self.input_size,300)
        self.activation    = nn.ReLU()
        self.dropout       = nn.Dropout(p=dropout)
        self.loss_func     = nn.CrossEntropyLoss()
        # For topic classification class-1: Science or Techn. class-0: Others
        self.tp_outlin     = nn.Linear(300,self.output_size)
        self.tp_outlin1    = nn.Linear(self.input_size,300)
        self.tp_activation  = nn.ReLU()
        self.tp_dropout       = nn.Dropout(p=dropout)        
        self.tp_loss_func  = nn.CrossEntropyLoss()

    def forward(self,input_ids=None, attention_mask=None, token_type_ids=None, labels=None, topic_labels=None, alpha=None):
        # input_ids[0].shape number of sections in first document in batch x max len
        a,pooled_output = self.bert(input_ids,
                                    attention_mask=attention_mask,
                                    token_type_ids=token_type_ids)
        encoding = torch.sum(pooled_output,dim=0,keepdim=True)
        reverse_encoding = ReverseLayerF.apply(encoding,alpha)

        out_gender = self.dropout(self.activation(self.outlin1(reverse_encoding)))
        out_gender = self.outlin(out_gender)
        
        # For topic classification
        out_tp = self.tp_dropout(self.tp_activation(self.tp_outlin1(encoding)))
        out_tp = self.tp_outlin(out_tp)
        # Losses 
        lss = self.loss_func(out_gender,labels)
        lss_tp = self.tp_loss_func(out_tp,topic_labels)
        lss_overall = lss_tp + lss
        return (out_gender,out_tp,lss_overall)
    
