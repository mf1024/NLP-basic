from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
import pickle
from nltk.tokenize import word_tokenize
import os
import torch

class FraEngDataset(Dataset):
    def __init__(self, data_source_path = 'fra-eng/fra.txt'):
        super().__init__()
        
        data_file_path = "sentences.pkl"
        
        self.sentence_list = []
        self.eng_token_dict = dict()
        self.eng_token_count = 0
        self.fra_token_dict = dict()
        self.fra_token_count = 0
        
        if os.path.exists(data_file_path):
            with open(data_file_path, 'rb') as f:
                self.sentence_list = pickle.load(f)
        
        else:
        
            with open(data_source_path, "r", encoding='utf-8') as f:
                for idx, line in enumerate(f.readlines()):

                    eng_token_sentence = []
                    fra_token_sentence = []
                    
                    snt = line.split('\t') 
                    eng_sentence = snt[0]
                    fra_sentence = snt[1]

                    eng_token_list = word_tokenize(eng_sentence)
                    for token in eng_token_list:
                        if token not in self.eng_token_dict:
                            self.eng_token_count += 1
                            self.eng_token_dict[token] = self.eng_token_count
                        
                        token_idx = self.eng_token_dict[token]
                        eng_token_sentence.append(token_idx)

                    fra_token_list = word_tokenize(fra_sentence)
                    for token in fra_token_list:
                        if token not in self.fra_token_dict:
                            self.fra_token_count += 1
                            self.fra_token_dict[token] = self.fra_token_count

                        token_idx = self.fra_token_dict[token]
                        fra_token_sentence.append(token_idx)
                        
                    self.sentence_list.append(
                        dict(
                            eng = eng_token_sentence,
                            fra = fra_token_sentence
                        ))


            with open(data_file_path, "wb") as f:
                pickle.dump(self.sentence_list, f)
        
        print(len(self.sentence_list))

    def __len__(self):
        return len(self.sentence_list)

    def __getitem__(self, item):
        ret = dict()
        for key in self.sentence_list[item]:
            ret[key] = torch.tensor(self.sentence_list[item][key])
        return ret


def fra_eng_dataset_collate(data):

    eng_sentences = []
    fra_sentences = []
    
    for s in data:
        eng_sentences.append(s['eng'].unsqueeze(dim=1))
        fra_sentences.append(s['fra'].unsqueeze(dim=1))
        
    eng_pad = pad_sequence(eng_sentences, padding_value=-1)
    fra_pad = pad_sequence(fra_sentences, padding_value=-1)
        
    return dict(eng=eng_pad, fra=fra_pad)
