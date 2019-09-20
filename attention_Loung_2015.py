import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from fra_eng_dataset import FraEngDataset, fra_eng_dataset_collate

BATCH_SIZE = 8
EMBEDDING_DIM = 128
RNN_HIDDEN_SIZE = 128
RNN_LAYERS = 4
EPOCHS = 20
MAX_OUTP_SENT_LEN = 20

# I will start without packing and padding to test the concepts first.

class EncoderModel(nn.Module):
    def __init__(self, in_dict_size):
        super().__init__()

        self.hidden = None
        self.cell = None

        self.embedding = nn.Embedding(
            num_embeddings = in_dict_size,
            embedding_dim = EMBEDDING_DIM
        )

        self.rnn = nn.LSTM(
            input_size = EMBEDDING_DIM,
            hidden_size = RNN_HIDDEN_SIZE,
            num_layers = RNN_LAYERS
        )

    def init_hidden_cell(self, batch_size = None):

        if batch_size == None:
            batch_size = BATCH_SIZE

        self.hidden = torch.zeros(RNN_LAYERS, batch_size, RNN_HIDDEN_SIZE)
        self.cell = torch.zeros(RNN_LAYERS, batch_size, RNN_HIDDEN_SIZE)


    def forward(self, x):

        x = self.embedding.forward(x)
        x, (self.hidden, self.cell) = self.rnn.forward(x, (self.hidden, self.cell))

        return x

def dot_score(ht, hs):
    ret = torch.sum(ht * hs, dim=1)
    return ret

class DecoderModel(nn.Module):
    def __init__(self, out_dict_size):
        super().__init__()

        self.score = dot_score
        self.current_batch = None

        self.embedding = nn.Embedding(
            num_embeddings = out_dict_size,
            embedding_dim = EMBEDDING_DIM
        )

        self.rnn = nn.LSTM(
            input_size = EMBEDDING_DIM,
            hidden_size = RNN_HIDDEN_SIZE,
            num_layers = RNN_LAYERS
        )

        self.logits_fc = nn.Sequential(
            nn.Linear(
                in_features = 2*RNN_HIDDEN_SIZE,
                out_features = out_dict_size
            ),
            nn.ReLU()
        )

        self.softmax = nn.Softmax(dim=1)



    def init_hidden_cell(self, hidden = None, cell = None, batch_size = None):

        if batch_size == None:
            batch_size = BATCH_SIZE

        self.current_batch = batch_size

        if not (torch.is_tensor(hidden) and torch.is_tensor(cell)):
            self.hidden = torch.zeros(RNN_LAYERS, batch_size, RNN_HIDDEN_SIZE)
            self.cell = torch.zeros(RNN_LAYERS, batch_size, RNN_HIDDEN_SIZE)
        else:
            self.hidden = hidden
            self.cell = cell


    def forward(self, enc_h, max_sentence_len, start_code):

        prev_dec_outp = torch.ones((self.current_batch, 1)) * start_code
        prev_dec_outp = prev_dec_outp.to(torch.int64)

        for i in range(max_sentence_len):
            print(self.embedding.weight.shape)
            prev_dec_outp_emb = self.embedding(prev_dec_outp)
            h_t, (self.hidden, self.cell) = self.rnn(prev_dec_outp_emb, (self.hidden, self.cell))

            energy = []
            for j in range(enc_h.shape[0]):
                energy.append(self.score(h_t[0], enc_h[j]).unsqueeze(dim=0))

            e_tens = torch.cat(energy, dim=0)
            a = torch.softmax(e_tens, dim=0)
            a = a.unsqueeze(dim=2)

            c = torch.sum(enc_h * a, dim=0)
            h_c_cat = torch.cat([h_t.squeeze(dim=0), c], dim=1)

            logits = self.logits_fc(h_c_cat)

            prob = self.softmax(logits)

            pass
            #concat c and h_t, and predict the next word

def get_one_hot(x, out_dim):

    tens = x.view(-1)
    tens_one_hot = torch.zeros(tens.size() + [out_dim])
    for i in len(tens):
        tens_one_hot[tens[i]] = 1

    tens_one_hot = tens_one_hot.view(x.size() + [out_dim])
    return tens_one_hot


dataset = FraEngDataset()
sentences_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, collate_fn=fra_eng_dataset_collate)

in_dict_size = dataset.get_fra_dict_size()
out_dict_size = dataset.get_eng_dict_size()

encoder_model = EncoderModel(in_dict_size)
decoder_model = DecoderModel(out_dict_size)

params = list(encoder_model.parameters()) + list(decoder_model.parameters())
optimizer = torch.optim.Adam(params, lr = 1e-3)

for epoch in range(EPOCHS):
    for sentences in sentences_loader:

        in_sentences = sentences['fra_sentences']
        in_lens = sentences['fra_lens']
        out_sentences = sentences['eng_sentences']
        out_lens = sentences['eng_lens']

        batch_loss = 0

        # Let's do it sub-optimally first
        for snt_idx in range(len(in_sentences)):

            encoder_model.init_hidden_cell(batch_size=1)

            enc_h = encoder_model.forward(in_sentences[snt_idx])

            decoder_model.init_hidden_cell(hidden=encoder_model.hidden, cell=encoder_model.cell, batch_size=1)
            sent_pred = decoder_model.forward(enc_h, MAX_OUTP_SENT_LEN, dataset.get_eng_start_code())

            out_sentences_one_hot = get_one_hot(out_sentences[snt_idx])
            loss = -torch.sum(torch.log(out_sentences_one_hot * (sent_pred + 1e-10)))
            loss.backward()

            batch_loss += loss.data

        optimizer.step()
        optimizer.zero_grad()


