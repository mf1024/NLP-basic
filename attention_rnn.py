# Implementation based on paper NEURAL MACHINE TRANSLATION BY JOINTLY LEARNING TO ALIGN AND TRANSLATE
# https://arxiv.org/abs/1409.0473

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

RNN_LAYERS = 4
RNN_HIDDEN_SIZE = 1024
IN_EMBEDDING_SIZE = 128
OUT_EMBEDDING_SIZE = 128
BATCH_SIZE = 64
MAX_OUTP_TIMESTEPS = 20
EPOCHS = 50

class EncoderModel(nn.Module):
    def __init__(self, in_dict_size):
        super().__init__()

        # Number of different tokens in the input language
        self.in_dict_size = in_dict_size

        self.hidden = None
        self.cell = None

        # This layer will make an embedding out of one_hot_wector for a word
        self.embedding = nn.Linear(
            in_features = in_dict_size,
            out_features = IN_EMBEDDING_SIZE
        )

        self.rnn = nn.LSTM(
            input_size=IN_EMBEDDING_SIZE,
            hidden_size=RNN_HIDDEN_SIZE,
            num_layers=RNN_LAYERS,
            bidirectional=True  #This defines that the LSTM will bi BI-DIRECTIONAL, obviously :)
        )

    def init_hidden_and_cell(self):
        # Dimension zero is multiplied by two because it's bidirectional LSTM, there are RNN_LAYERS layers for each direction
        self.hidden = torch.zeros( 2 * RNN_LAYERS, BATCH_SIZE, RNN_HIDDEN_SIZE)
        self.cell = torch.zeros( 2 * RNN_LAYERS, BATCH_SIZE, RNN_HIDDEN_SIZE)

    def forward(self, x):

        padded_sent_one_hot, sent_lens = x
        padded_sent_emb = self.embedding.forward(padded_sent_one_hot)
        packed = pack_padded_sequence(padded_sent_emb, sent_lens)
        packed, (self.hidden, self.cell) = self.rnn.forward(packed, (self.hidden,self.cell))
        return packed


class DecoderModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.hidden = None
        self.cell = None

        self.rnn = nn.LSTM(
            input_size= 2*RNN_HIDDEN_SIZE, # Input of decoder will be weighted average of the encoder so it will heep the size - 2*RNN_HIDDEN_SIZE
            hidden_size=RNN_HIDDEN_SIZE,
            num_layers=RNN_LAYERS
        )

        self.alignment = nn.Linear(

        )
        #Define the Align and Translate decoder mechanism here

    def init_hidden_and_cell(self):
        self.hidden = torch.zeros(RNN_LAYERS, BATCH_SIZE, RNN_HIDDEN_SIZE)
        self.cell = torch.zeros(RNN_LAYERS, BATCH_SIZE, RNN_HIDDEN_SIZE)

    def forward(self, packed_encoder_sentences):

        padded, sent_lens = pad_packed_sequence(packed_encoder_sentences)
        max_sentence_length = padded.shape[0]

        for i in range(max_sentence_length):






        return 42


