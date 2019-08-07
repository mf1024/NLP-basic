# implementation based on "Sequence to Sequence Learning with Neural Networks" 2014 paper

import torch.nn as nn
from torch.utils.data import DataLoader
from fra_eng_dataset import FraEngDataset, fra_eng_dataset_collate

def process_fra_eng_sentences(data_path):
   return

class RNN_Model(nn.Module):
   def __init__(self):
      super().__init__()
      
      # Def the encoder/ decoder
      # Def word embedding for encoding part
      # Def word embedding for decoding part
   
   def forward(self, x):
      return x

dataset = FraEngDataset()
sentences_loader = DataLoader(dataset, batch_size=10, shuffle=True, drop_last=True, collate_fn=fra_eng_dataset_collate)

for idx, sentences in enumerate(sentences_loader):
   print(sentences)
   if idx>10:
      break


