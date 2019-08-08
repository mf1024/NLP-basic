# implementation based on "Sequence to Sequence Learning with Neural Networks" 2014 paper

import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from fra_eng_dataset import FraEngDataset, fra_eng_dataset_collate

def process_fra_eng_sentences(data_path):
   return

RNN_LAYERS = 4
RNN_HIDDEN_SIZE = 128
IN_EMBEDDING_SIZE = 128
OUT_EMBEDDING_SIZE = 128
BATCH_SIZE = 8

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
   
   def forward(self, x):
      
      x = self.embedding.forward(x)
      x, _ = self.rnn.forward(x, (self.hidden,self.cell))
      
      
      return 100
      
      # Return the output hidden states
      
      return (self.hidden, self.cell)

class RNN_decoder_model(nn.Module):
   def __init__(self, out_dict_size):
      super().__init__()
      
      self.rnn = nn.LSTM(
         input_size = IN_EMBEDDING_SIZE,
         hidden_size = RNN_HIDDEN_SIZE,
         num_layers = RNN_LAYERS
      )
      
      self.rnn_to_embedding = nn.Linear(
         in_features = RNN_HIDDEN_SIZE,
         out_features = OUT_EMBEDDING_SIZE
      )
      
      self.out_embedding = nn.Linear(
         in_features = OUT_EMBEDDING_SIZE, 
         out_features = out_dict_size
      )
   
   def init_hidden_and_cell(self, hidden, cell):
      self.hidden = hidden
      self.cell = cell
   
   def forward(self, x):
      
      #Return the output one-hots
       
      return x
   
dataset = FraEngDataset()
sentences_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, collate_fn=fra_eng_dataset_collate)

rnn_encoder = RNN_encoder_model(dataset.get_eng_dict_size())
rnn_decoder = RNN_decoder_model(dataset.get_fra_dict_size())

for idx, sentences in enumerate(sentences_loader):

   eng_sentence = sentences['eng']
   fra_sentence = sentences['fra']
   
   rnn_encoder.init_hidden_and_cell()
   one_hot = torch.zeros(eng_sentence.shape[0], eng_sentence.shape[1], dataset.get_eng_dict_size())
   one_hot = one_hot.scatter_(2,eng_sentence.data,1)
   
   hidden, cell = rnn_encoder.forward(one_hot)
   
   break
