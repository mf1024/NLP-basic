# implementation based on "Sequence to Sequence Learning with Neural Networks" 2014 paper

import torch
import torch.nn as nn

#Need Dataloader and Dataset for the sentences

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


data_path = 'fra-eng/fra.txt'
sentences = process_fra_eng_sentences(data_path)


