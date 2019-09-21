import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from fra_eng_dataset import FraEngDataset, fra_eng_dataset_collate

BATCH_SIZE = 32
EMBEDDING_DIM = 512
RNN_HIDDEN_SIZE = 256
RNN_LAYERS = 4
EPOCHS = 20
MAX_OUTP_SENT_LEN = 20

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'


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
            nn.Sigmoid()
        )
        self.softmax = nn.Softmax(dim=1)


    def init_hidden_cell(self, hidden = None, cell = None, batch_size = None):

        if batch_size == None:
            batch_size = BATCH_SIZE

        self.current_batch = batch_size

        if not (torch.is_tensor(hidden) and torch.is_tensor(cell)):
            self.hidden = torch.zeros(RNN_LAYERS, batch_size, RNN_HIDDEN_SIZE).to(device)
            self.cell = torch.zeros(RNN_LAYERS, batch_size, RNN_HIDDEN_SIZE).to(device)
        else:
            self.hidden = hidden
            self.cell = cell


    def forward(self, enc_h, max_sentence_len, start_code):

        prev_dec_outp = torch.ones((self.current_batch, 1)) * start_code
        prev_dec_outp = prev_dec_outp.to(torch.int64).to(device)

        prob_ret_list = []

        for i in range(max_sentence_len):
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

            prob_ret = prob.unsqueeze(dim=0) #Unsqueeze the sequence dimension
            prob_ret_list.append(prob_ret)

            prev_dec_outp = torch.argmax(prob.data, dim=1, keepdim=True)

        prob_ret_tens = torch.cat(prob_ret_list, dim=0)
        return prob_ret_tens


def get_one_hot(x, out_dim):

    tens = x.view(-1)
    tens_one_hot = torch.zeros(list(tens.size()) + [out_dim])
    for i in range(len(tens)):
        tens_one_hot[i,tens[i]] = 1

    tens_one_hot = tens_one_hot.view(list(x.size()) + [out_dim])
    return tens_one_hot.to(device)


dataset = FraEngDataset()
sentences_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, collate_fn=fra_eng_dataset_collate)

in_dict_size = dataset.get_fra_dict_size()
out_dict_size = dataset.get_eng_dict_size()

encoder_model = EncoderModel(in_dict_size).to(device)
decoder_model = DecoderModel(out_dict_size).to(device)

params = list(encoder_model.parameters()) + list(decoder_model.parameters())
optimizer = torch.optim.Adam(params, lr = 1e-3)

for epoch in range(EPOCHS):
    for sentences in sentences_loader:

        in_sentences = sentences['fra_sentences']
        in_lens = sentences['fra_lens']
        out_sentences = sentences['eng_sentences']
        out_lens = sentences['eng_lens']

        batch_loss = 0

        # Let's do it sub-optimally first - one sentence by one sentence
        for snt_idx in range(len(in_sentences)):

            encoder_model.init_hidden_cell(batch_size=1)

            enc_h = encoder_model.forward(in_sentences[snt_idx])

            decoder_model.init_hidden_cell(hidden=encoder_model.hidden, cell=encoder_model.cell, batch_size=1)
            sent_pred = decoder_model.forward(enc_h, out_lens[snt_idx], dataset.get_eng_start_code())

            out_sentences_one_hot = get_one_hot(out_sentences[snt_idx], out_dim=out_dict_size)
            loss = -torch.sum(out_sentences_one_hot * torch.log(sent_pred + 1e-10))
            loss.backward()

            batch_loss += loss.data

        optimizer.step()
        optimizer.zero_grad()

        print(f"batch_loss is {batch_loss}")
        batch_loss = 0
