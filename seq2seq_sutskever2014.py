# implementation based on "Sequence to Sequence Learning with Neural Networks" 2014 paper

import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from fra_eng_dataset import FraEngDataset, fra_eng_dataset_collate
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import os

RNN_LAYERS = 4
RNN_HIDDEN_SIZE = 1024
IN_EMBEDDING_SIZE = 128
OUT_BOTTLENECK_SIZE = 128
BATCH_SIZE = 64
MAX_OUTP_TIMESTEPS = 20
EPOCHS = 50

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'


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
        self.hidden = torch.zeros(RNN_LAYERS, BATCH_SIZE, RNN_HIDDEN_SIZE).to(device)
        self.cell = torch.zeros(RNN_LAYERS, BATCH_SIZE, RNN_HIDDEN_SIZE).to(device)

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

        self.embedding = nn.Linear(
            in_features=out_dict_size,
            out_features=IN_EMBEDDING_SIZE
        )

        self.rnn = nn.LSTM(
            input_size = IN_EMBEDDING_SIZE,
            hidden_size = RNN_HIDDEN_SIZE,
            num_layers = RNN_LAYERS
        )

        self.rnn_out_bottleneck = nn.Linear(
            in_features = RNN_HIDDEN_SIZE,
            out_features = OUT_BOTTLENECK_SIZE
        )

        self.out_bottleneck_to_logit = nn.Linear(
            in_features = OUT_BOTTLENECK_SIZE,
            out_features = out_dict_size
        )

        self.softmax = nn.Softmax(dim=2)

    def init_hidden_and_cell(self, hidden, cell):
        self.hidden = hidden
        self.cell = cell

    def forward(self, out_eos_code, out_dict_size, max_sentence_len):

        batch_size = self.hidden.shape[1]
        prev_outp = (torch.ones(1, batch_size, 1) * out_eos_code).long()
        prev_outp = prev_outp.to(device)

        all_outp_prob = []

        for timestep in range(max_sentence_len):

            prev_outp_one_hot = torch.zeros(prev_outp.shape[0], prev_outp.shape[1], out_dict_size).to(device)
            prev_outp_one_hot = prev_outp_one_hot.scatter_(2,prev_outp.data,1)

            prev_outp_in_emb = self.embedding(prev_outp_one_hot)

            cur_outp_hid, (self.hidden, self.cell) = self.rnn.forward(prev_outp_in_emb, (self.hidden, self.cell))
            cur_outp_emb = self.rnn_to_embedding.forward(cur_outp_hid)
            cur_outp_logits = self.out_bottleneck_to_logit(cur_outp_emb)
            cur_outp_prob = self.softmax(cur_outp_logits)
            all_outp_prob.append(cur_outp_prob)

            prev_outp = torch.argmax(cur_outp_prob.detach().to(device), dim=2, keepdim=True)

        all_outp_prob_tensor = torch.cat(all_outp_prob, dim=0)

        return all_outp_prob_tensor

dataset = FraEngDataset()
sentences_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, collate_fn=fra_eng_dataset_collate)

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

rnn_encoder = RNN_encoder_model(dataset.get_fra_dict_size()).to(device)
rnn_decoder = RNN_decoder_model(dataset.get_eng_dict_size()).to(device)

trained_encoder_path = None
trained_decoder_path = None

trained_encoder_path = 'models/encoder.pt'
trained_decoder_path = 'models/decoder.pt'

if trained_encoder_path:
    rnn_encoder.load_state_dict(torch.load(trained_encoder_path))
if trained_decoder_path:
    rnn_decoder.load_state_dict(torch.load(trained_decoder_path))


params = list(rnn_encoder.parameters()) + list(rnn_decoder.parameters())
optimizer = torch.optim.Adam(params, lr = 1e-3)

steps = 0

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

        rnn_encoder.forward((padded_in_one_hot, in_lens))
        hidden, cell = rnn_encoder.get_hidden_and_cell()

        rnn_decoder.init_hidden_and_cell(hidden,cell)

        max_sentence_len = padded_out.shape[0]
        y_pred = rnn_decoder.forward(dataset.get_eng_eos_code(), dataset.get_eng_dict_size(), max_sentence_len)

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
