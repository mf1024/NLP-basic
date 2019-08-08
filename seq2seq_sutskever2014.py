# implementation based on "Sequence to Sequence Learning with Neural Networks" 2014 paper

import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from fra_eng_dataset import FraEngDataset, fra_eng_dataset_collate
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

def process_fra_eng_sentences(data_path):
   return

RNN_LAYERS = 4
RNN_HIDDEN_SIZE = 128
IN_EMBEDDING_SIZE = 128
OUT_EMBEDDING_SIZE = 128
BATCH_SIZE = 16
MAX_OUTP_TIMESTEPS = 20


class RNN_encoder_model(nn.Module):
   def __init__(self, in_dict_size):
      super().__init__()
      
      self.in_dict_size = in_dict_size

      self.embedding = nn.Linear(
         in_dict_size, 
         IN_EMBEDDING_SIZE)
      
      self.hidden = None 
      self.cell = None

      self.rnn = nn.LSTM(
         input_size = IN_EMBEDDING_SIZE,
         hidden_size = RNN_HIDDEN_SIZE,
         num_layers = RNN_LAYERS
      )
       
   def init_hidden_and_cell(self):
       self.hidden = torch.randn(RNN_LAYERS, BATCH_SIZE, RNN_HIDDEN_SIZE)
       self.cell = torch.rand(RNN_LAYERS, BATCH_SIZE, RNN_HIDDEN_SIZE)
   
   def get_hidden_and_cell(self):
      return self.hidden, self.cell
   
   def forward(self, x):
      padded_sent_one_hot, sent_lens = x
      padded_sent_emb = self.embedding.forward(padded_sent_one_hot)
      packed = pack_padded_sequence(padded_sent_emb, sent_lens)
      packed, (self.hidden, self.cell) = self.rnn.forward(packed, (self.hidden,self.cell))
      padded, sent_lens = pad_packed_sequence(packed)

class RNN_decoder_model(nn.Module):
   def __init__(self, out_dict_size):
      super().__init__()
      
      self.in_embedding = nn.Linear(
         in_features=out_dict_size,
         out_features=IN_EMBEDDING_SIZE
      )
      
      self.rnn = nn.LSTM(
         input_size = IN_EMBEDDING_SIZE,
         hidden_size = RNN_HIDDEN_SIZE,
         num_layers = RNN_LAYERS
      )
      
      self.rnn_to_embedding = nn.Linear(
         in_features = RNN_HIDDEN_SIZE,
         out_features = OUT_EMBEDDING_SIZE
      )
      
      self.embedding_to_logit = nn.Linear(
         in_features = OUT_EMBEDDING_SIZE, 
         out_features = out_dict_size
      )
   
      self.softmax = nn.Softmax(dim=2)
       
   def init_hidden_and_cell(self, hidden, cell):
      self.hidden = hidden
      self.cell = cell
   
   def forward(self, out_eos_code, out_dict_size, max_sentence_len):
      
      batch_size = self.hidden.shape[1]
      prev_outp = (torch.ones(1, batch_size, 1) * out_eos_code).long()
      
      all_outp_prob = []

      for timestep in range(max_sentence_len):
         
         prev_outp_one_hot = torch.zeros(prev_outp.shape[0], prev_outp.shape[1], out_dict_size)
         prev_outp_one_hot = prev_outp_one_hot.scatter_(2,prev_outp.data,1)
         
         prev_outp_in_emb = self.in_embedding(prev_outp_one_hot)
         
         cur_outp_hid, (self.hidden, self.cell) = self.rnn.forward(prev_outp_in_emb, (self.hidden, self.cell))
         cur_outp_emb = self.rnn_to_embedding.forward(cur_outp_hid)
         cur_outp_logits = self.embedding_to_logit(cur_outp_emb)
         cur_outp_prob = self.softmax(cur_outp_logits)
         all_outp_prob.append(cur_outp_prob)
         
         prev_outp = torch.argmax(cur_outp_prob.detach().to('cpu'), dim=2, keepdim=True)
          
      all_outp_prob_tensor = torch.cat(all_outp_prob, dim=0)
       
      return all_outp_prob_tensor
   
dataset = FraEngDataset()
sentences_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, collate_fn=fra_eng_dataset_collate)

rnn_encoder = RNN_encoder_model(dataset.get_eng_dict_size())
rnn_decoder = RNN_decoder_model(dataset.get_fra_dict_size())


params = list(rnn_encoder.parameters()) + list(rnn_decoder.parameters())
optimizer = torch.optim.Adam(params, lr = 1e-4)

for idx, sentences in enumerate(sentences_loader):

   rnn_encoder.init_hidden_and_cell()
   
   eng_sentences = sentences['eng_sentences']
   eng_lens = sentences['eng_lens']
   fra_sentences = sentences['fra_sentences']
   fra_lens = sentences['fra_lens']

   padded_eng = pad_sequence(eng_sentences, padding_value=0)
   padded_fra = pad_sequence(fra_sentences, padding_value=0)

   padded_eng_one_hot = torch.zeros(padded_eng.shape[0], padded_eng.shape[1], dataset.get_eng_dict_size())
   padded_eng_one_hot = padded_eng_one_hot.scatter_(2,padded_eng.data,1)
   
   rnn_encoder.forward((padded_eng_one_hot, eng_lens))
   hidden, cell = rnn_encoder.get_hidden_and_cell()
   
   rnn_decoder.init_hidden_and_cell(hidden,cell)
   
   max_sentence_len = padded_fra.shape[0]
   y_pred = rnn_decoder.forward(dataset.get_fra_eos_code(), dataset.get_fra_dict_size(), max_sentence_len)


   padded_fra_one_hot = torch.zeros(padded_fra.shape[0], padded_fra.shape[1], dataset.get_fra_dict_size())
   padded_fra_one_hot = padded_fra_one_hot.scatter_(2,padded_fra.data,1)
   
   #Make all padded one-hot vectors to all zeros, which which will make
   #padded components loss 0 and sop wont affect the loss
   padded_fra_one_hot[:,:,0] = torch.zeros(padded_fra_one_hot.shape[0], padded_fra_one_hot.shape[1])
   loss = torch.sum(-torch.log(y_pred + 1e-9) * padded_fra_one_hot)
   
   loss.backward()
   optimizer.step()
   optimizer.zero_grad()
