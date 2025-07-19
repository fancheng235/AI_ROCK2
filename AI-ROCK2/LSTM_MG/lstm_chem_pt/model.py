import torch
from torch import nn
from lstm_chem_pt.utils.smiles_tokenizer import SmilesTokenizer
from torch.nn import functional as F
import os


class LSTMmodel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, layer_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim,layer_dim,
                    batch_first=True, dropout=0.5)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        embeds = self.embedding(x.long()).type(torch.float32)
        r_out, (h_n, c_n) = self.lstm(embeds)
        output = self.fc(r_out)
        return output

class LSTMChem(object):
    def __init__(self, config, session='train'):
        assert session in ['train', 'generate', 'finetune'], \
                'one of {train, generate, finetune}'
        st = SmilesTokenizer()
        self.config = config
        self.vocab_size = len(st.table)
        self.session = session
        self.model = None

        if self.session == 'train':
            self.build_model()
        else:
            print(f'Loading model from {self.config.model_path} ...')
            self.model = torch.load(self.config.model_path)

    def build_model(self):
        self.model = LSTMmodel(self.vocab_size, self.config.embedding_dim, 
                            self.config.units,3, self.vocab_size)
        self.config.model_path = os.path.join(self.config.exp_dir,
                                                       'LSTM.pkl')