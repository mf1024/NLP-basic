# Implementation based on paper NEURAL MACHINE TRANSLATION BY JOINTLY LEARNING TO ALIGN AND TRANSLATE
# https://arxiv.org/abs/1409.0473

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from fra_eng_dataset import FraEngDataset, fra_eng_dataset_collate
from torch.utils.data import DataLoader
import os

def print_results(in_sentence_list, out_sentence_list, pred_tensor):

    in_token_to_text = dataset.fra_token_to_text
    out_token_to_text = dataset.eng_token_to_text

    for s in range(len(in_sentence_list)):

        in_sent_text = []
        for in_token in in_sentence_list[s].squeeze():
            in_sent_text.append(in_token_to_text[in_token])
        print(f"\nFrench sentence is: {' '.join(in_sent_text)}")

        out_sent_text = []
        for out_token in out_sentence_list[s].squeeze():
            out_sent_text.append(out_token_to_text[out_token])
        print(f"\nEnglish sentence is: {' '.join(out_sent_text)}")

        pred_sent_text = []
        for ts in range(pred_tensor.shape[0]):
            pred_token = torch.argmax(pred_tensor[ts, s,:]).data
            pred_sent_text.append(out_token_to_text[pred_token])
            if pred_token == dataset.get_eng_eos_code():
                break
        print(f"Translated English sentence is: {' '.join(pred_sent_text)}")


RNN_LAYERS = 4
RNN_HIDDEN_SIZE = 1024
IN_EMBEDDING_SIZE = 128
OUT_EMBEDDING_SIZE = 128
OUT_BOTTLENECK_SIZE = 128
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

        self.embedding = nn.Embedding(
            num_embeddings = in_dict_size,
            embedding_dim = IN_EMBEDDING_SIZE
        )

        self.rnn = nn.LSTM(
            input_size=IN_EMBEDDING_SIZE,
            hidden_size=RNN_HIDDEN_SIZE,
            num_layers=RNN_LAYERS,
            bidirectional=True  #This defines that the LSTM will bi BI-DIRECTIONAL, obviously :)
        )

    def init_hidden_and_cell(self):
        # Dimension zero is multiplied by two because it's bidirectional LSTM, there are RNN_LAYERS layers for each direction
        self.hidden = torch.zeros( 2 * RNN_LAYERS, BATCH_SIZE, RNN_HIDDEN_SIZE).to(device)
        self.cell = torch.zeros( 2 * RNN_LAYERS, BATCH_SIZE, RNN_HIDDEN_SIZE).to(device)

    def forward(self, x):

        padded_sent, sent_lens = x
        padded_sent_emb = self.embedding.forward(padded_sent)
        packed = pack_padded_sequence(padded_sent_emb, sent_lens)
        packed, (self.hidden, self.cell) = self.rnn.forward(packed, (self.hidden,self.cell))
        return packed


class DecoderModel(nn.Module):
    def __init__(self, out_dict_size):
        super().__init__()

        self.hidden = None
        self.cell = None

        self.embedding = nn.Embedding(
            num_embeddings = out_dict_size,
            embedding_dim = OUT_EMBEDDING_SIZE
        )

        self.rnn = nn.LSTM(
             # Input of decoder will be weighted average of the encoder outputs concatinated with last timestep output embedding
            input_size= 2*RNN_HIDDEN_SIZE + OUT_EMBEDDING_SIZE,
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

        self.alignment_softmax = nn.Softmax(dim=0) #Check this again

        self.out_bottleneck = nn.Linear(
            in_features = RNN_HIDDEN_SIZE,
            out_features = OUT_BOTTLENECK_SIZE
        )

        self.bottleneck_to_logits = nn.Linear(
            in_features = OUT_BOTTLENECK_SIZE,
            out_features = out_dict_size
        )

        self.softmax = nn.Softmax(dim=2)


    def init_hidden_and_cell(self):
        self.hidden = torch.zeros(RNN_LAYERS, BATCH_SIZE, RNN_HIDDEN_SIZE).to(device)
        self.cell = torch.zeros(RNN_LAYERS, BATCH_SIZE, RNN_HIDDEN_SIZE).to(device)

    def forward(self, packed_encoder_sentences, padded_is_on, out_eos_token, max_out_sentence_len, max_in_sentence_len):

        padded, sent_lens = pad_packed_sequence(packed_encoder_sentences)

        prev_timestep_pred = (torch.ones(1, BATCH_SIZE, 1) * out_eos_token).long().to(device)
        out_rnn = torch.zeros(1, BATCH_SIZE, RNN_HIDDEN_SIZE).to(device)
        prob_list = []

        for i in range(max_out_sentence_len):

            prev_timestep_pred = prev_timestep_pred.squeeze(dim=0).squeeze(dim=1)
            prev_timestep_pred_emb = self.embedding(prev_timestep_pred)

            prev_out_rnn = out_rnn.data

            #[BATCH_SIZE, 3*RNN_HIDDEN_SIZE]
            a_list = []
            for j in range(max_in_sentence_len):
                state_concat = torch.cat([prev_out_rnn[0], padded[j]], dim=1)
                a_i = self.alignment.forward(state_concat)
                a_list.append(a_i)

            a_tensor = torch.cat(a_list, dim=1)
            #Making sure that we don't pay attention to padded elements
            a_tensor  = a_tensor.permute(1,0)
            a_tensor = a_tensor * padded_in_is_on
            alignment = self.alignment_softmax(a_tensor)

            alignment = alignment.unsqueeze(dim=2)
            context = torch.sum(alignment * padded, dim=0)

            rnn_input = torch.cat([context, prev_timestep_pred_emb], dim = 1)
            rnn_input = rnn_input.unsqueeze(dim=0)
            out_rnn, (self.hidden, self.cell) = self.rnn(rnn_input)

            out_bottleneck = self.out_bottleneck(out_rnn)
            out_logits = self.bottleneck_to_logits(out_bottleneck)
            out_prob = self.softmax(out_logits)
            prob_list.append(out_prob)

            prev_timestep_pred = torch.argmax(out_prob.data, dim=2, keepdim=True)

        return torch.cat(prob_list, dim=0)

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
    steps = 0

    for idx, sentences in enumerate(sentences_loader):

        rnn_encoder.init_hidden_and_cell()

        in_sentences = sentences['fra_sentences']
        in_lens = sentences['fra_lens']
        out_sentences = sentences['eng_sentences']
        out_lens = sentences['eng_lens']

        padded_in = pad_sequence(in_sentences, padding_value=0).to(device)
        padded_in = padded_in.squeeze(dim=2)
        padded_out = pad_sequence(out_sentences, padding_value=0).to(device)

        packed = rnn_encoder.forward((padded_in, in_lens))

        rnn_decoder.init_hidden_and_cell()

        max_out_sentence_len = padded_out.shape[0]
        max_in_sentence_len = padded_in.shape[0]
        if max_out_sentence_len > MAXMAX_SENTENCE_LEN:
            max_out_sentence_len = MAXMAX_SENTENCE_LEN

        if max_in_sentence_len > MAXMAX_SENTENCE_LEN:
            max_in_sentence_len = MAXMAX_SENTENCE_LEN

        padded_in_is_on = (padded_in > 0).float()
        y_pred = rnn_decoder.forward(packed, padded_in_is_on, dataset.get_eng_eos_code(), max_out_sentence_len, max_in_sentence_len)

        steps += BATCH_SIZE
        if steps > 5000:
            steps = 0
            print_results(in_sentences, out_sentences, y_pred.to('cpu').detach().data)

        padded_out_one_hot = torch.zeros(padded_out.shape[0], padded_out.shape[1], dataset.get_eng_dict_size()).to(device)
        padded_out_one_hot = padded_out_one_hot.scatter_(2,padded_out.data,1)

        #Make all padded one-hot vectors to all zeros, which which will make
        #padded components loss 0 and sop wont affect the loss
        padded_out_one_hot[:,:,dataset.get_eng_pad_code()] = torch.zeros(padded_out_one_hot.shape[0], padded_out_one_hot.shape[1]).to(device)
        loss = torch.sum(-torch.log(y_pred + 1e-9) * padded_out_one_hot)

        loss_sum += loss.to('cpu').data

        print(loss.data)

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
