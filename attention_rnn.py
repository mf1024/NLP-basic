# Implementation based on paper NEURAL MACHINE TRANSLATION BY JOINTLY LEARNING TO ALIGN AND TRANSLATE
# https://arxiv.org/abs/1409.0473

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from fra_eng_dataset import FraEngDataset, fra_eng_dataset_collate
from torch.utils.data import DataLoader

RNN_LAYERS = 4
RNN_HIDDEN_SIZE = 1024
IN_EMBEDDING_SIZE = 128
OUT_EMBEDDING_SIZE = 128
ALIGNMENT_HIDDEN_SIZE = 256
BATCH_SIZE = 64
MAXMAX_SENTENCE_LEN = 50
EPOCHS = 50

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'

class EncoderModel(nn.Module):
    def __init__(self, in_dict_size):
        super().__init__()

        # Number of different tokens in the input language
        self.in_dict_size = in_dict_size

        self.hidden = None
        self.cell = None

        # This layer will make an embedding out of one_hot_vector for a word
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
    def __init__(self, out_dict_size):
        super().__init__()

        self.hidden = None
        self.cell = None

        self.rnn = nn.LSTM(
            input_size= 2*RNN_HIDDEN_SIZE, # Input of decoder will be weighted average of the encoder so it will heep the size - 2*RNN_HIDDEN_SIZE
            hidden_size=RNN_HIDDEN_SIZE,
            num_layers=RNN_LAYERS
        )

        #Alignment(or attention) feed-forward network
        #We have the output from the last decoder timestep S[i-1], and we
        #need to get the alignment of every decoder timestep
        self.alignment = nn.Sequential(
            nn.Linear(
                in_features=3 * RNN_HIDDEN_SIZE,
                out_features=ALIGNMENT_HIDDEN_SIZE
            ),
            nn.Linear(
                in_features=ALIGNMENT_HIDDEN_SIZE,
                out_features=ALIGNMENT_HIDDEN_SIZE
            ),
            nn.Linear(
                in_features=ALIGNMENT_HIDDEN_SIZE,
                out_features=ALIGNMENT_HIDDEN_SIZE
            ),
            nn.Linear(
                in_features=ALIGNMENT_HIDDEN_SIZE,
                out_features=1
            )
        )

        self.a_softmax = nn.Softmax(dim=1) #Check this again


    def init_hidden_and_cell(self):
        self.hidden = torch.zeros(RNN_LAYERS, BATCH_SIZE, RNN_HIDDEN_SIZE)
        self.cell = torch.zeros(RNN_LAYERS, BATCH_SIZE, RNN_HIDDEN_SIZE)

    def forward(self, packed_encoder_sentences):

        padded, sent_lens = pad_packed_sequence(packed_encoder_sentences)
        max_sentence_length = padded.shape[0]

        S = torch.zeros(1, BATCH_SIZE, RNN_HIDDEN_SIZE).to(device)

        for i in range(MAXMAX_SENTENCE_LEN):

            #[BATCH_SIZE, 3*RNN_HIDDEN_SIZE]
            a_list = []
            for j in range(max_sentence_length):
                state_concat = torch.cat([S.squeeze(dim=0), padded[j]], dim=1)
                a_i = self.alignment.forward(state_concat)
                a_list.append(a_i)


            a_tensor = torch.cat(a_list, dim=1)
            alignment = self.a_softmax(a_tensor)

            alignment = alignment.unsqueeze(dim=2)
            alignment = alignment.permute(1,0,2)
            context = torch.sum(alignment * padded, dim=0)
            S, (self.hidden, self.cell) = self.rnn(context)
            S = S.detach()

            #TODO: make sure that we dont pay attention to padded entries.

        return 42

dataset = FraEngDataset()
sentences_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, collate_fn=fra_eng_dataset_collate)


rnn_encoder = EncoderModel(dataset.get_fra_dict_size()).to(device)
rnn_decoder = DecoderModel(dataset.get_eng_dict_size()).to(device)

params = list(rnn_encoder.parameters()) + list(rnn_decoder.parameters())
optimizer = torch.optim.Adam(params, lr = 1e-3)

for epoch in range(EPOCHS):

    print(f"Starting epoch {epoch} =====================")

    best_loss = 1e10
    loss_sum = 0

    for idx, sentences in enumerate(sentences_loader):

        rnn_encoder.init_hidden_and_cell()

        in_sentences = sentences['fra_sentences']
        in_lens = sentences['fra_lens']
        out_sentences = sentences['eng_sentences']
        out_lens = sentences['eng_lens']

        padded_in = pad_sequence(in_sentences, padding_value=0).to(device)
        padded_out = pad_sequence(out_sentences, padding_value=0).to(device)

        padded_in_one_hot = torch.zeros(padded_in.shape[0], padded_in.shape[1], dataset.get_fra_dict_size()).to(device)
        padded_in_one_hot = padded_in_one_hot.scatter_(2,padded_in.data,1)

        packed = rnn_encoder.forward((padded_in_one_hot, in_lens))

        rnn_decoder.init_hidden_and_cell()

        max_sentence_len = padded_out.shape[0]
        y_pred = rnn_decoder.forward(packed)

        steps += BATCH_SIZE
        if steps > 5000:
            steps = 0
            print_results(in_sentences, out_sentences, y_pred.to('cpu').detach().data)

        padded_out_one_hot = torch.zeros(padded_out.shape[0], padded_out.shape[1], dataset.get_eng_dict_size()).to(device)
        padded_out_one_hot = padded_out_one_hot.scatter_(2,padded_out.data,1)

        #Make all padded one-hot vectors to all zeros, which which will make
        #padded components loss 0 and sop wont affect the loss
        padded_out_one_hot[:,:,0] = torch.zeros(padded_out_one_hot.shape[0], padded_out_one_hot.shape[1])
        loss = torch.sum(-torch.log(y_pred + 1e-9) * padded_out_one_hot)

        loss_sum += loss.to('cpu').detach().data

        print(loss.to('cpu').detach().data)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    print(f"Epoch {epoch} loss sum is {loss_sum}")
    if best_loss > loss_sum:
        best_loss = loss_sum

        models_path = "models"
        if not os.path.exists(models_path):
            os.mkdir(models_path)

        torch.save(rnn_encoder.state_dict(), os.path.join(models_path, "encoder.pt"))
        torch.save(rnn_decoder.state_dict(), os.path.join(models_path, "decoder.pt"))
