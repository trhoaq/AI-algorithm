#machine translate
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

import pandas as pd
import os, re, pickle, random, time, math, string
from tqdm import tqdm
from collections import Counter
from nltk.corpus import stopwords
#seed, device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

set_seed(36)


#preprocess

class vocab:
    def __init__(self, vocab=None, inv_vocab=None):
        self.vocab = {"<pad>": 0,
                      "<sos>": 1, 
                      "<eos>": 2, 
                      "<unk>": 3,
                      }
        self.inv_vocab = {j: i for i, j in self.vocab.items()}

    def add_word(self, token):
        for word in token:
            if word not in self.vocab:
                self.vocab[word] = len(self.vocab)
        self.inv_vocab = {j: i for i, j in self.vocab.items()}

    def rev_indexing(self, token):
        idx_sent = []
        for tok in token:
            idx_sent.append(self.vocab.get(tok, self.vocab["<unk>"]))
        return idx_sent
    
    def rev_voc(self, idx):
        sent = []
        for id in idx:
            sent.append(self.inv_vocab.get(id, "<unk>"))
        return sent

def data_prepare(data, vocab, stopword) -> vocab:
    for sent in data:
        sent = str(sent).lower()
        token = re.findall(r"\w+", sent)
        token = [word for word in token if word not in stopword]
        vocab.add_word(token)

root = "./dts/machine_trans/train"
bilingual = os.path.join(root, "song_ngu.csv")
vi = os.path.join(root, "vi.csv")
en = os.path.join(root, "en.csv")
data = pd.read_csv(bilingual)

vi_voc = vocab()
en_voc = vocab()

vi_stopword = [
    "và", "là", "có", "của", "trong", "một", "những", "được", 
    "với", "cho", "khi", "này", "đó", "thì", "ra", "đã", "cũng"
]

en_stopword = list(stopwords.words("english"))

data_prepare(data["english_sentence"], en_voc, en_stopword) 
data_prepare(data["vietnamese_sentence"], vi_voc, vi_stopword)

#dataset
class Dts(Dataset):
    def __init__(self, data, en_voc, vi_voc):
        super().__init__()
        self.data = data
        self.en_voc = en_voc
        self.vi_voc = vi_voc

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Tokenize and numericalize
        vi_tokens = re.findall(r"\w+", str(row["vietnamese_sentence"]).lower())
        en_tokens = re.findall(r"\w+", str(row["english_sentence"]).lower())
        
        vi_idx = self.vi_voc.rev_indexing(vi_tokens)
        en_idx = self.en_voc.rev_indexing(en_tokens)

        # Add <sos> and <eos> tokens
        vi_idx = [self.vi_voc.vocab["<sos>"]] + vi_idx + [self.vi_voc.vocab["<eos>"]]
        en_idx = [self.en_voc.vocab["<sos>"]] + en_idx + [self.en_voc.vocab["<eos>"]]

        return torch.LongTensor(en_idx), torch.LongTensor(vi_idx)

def collate_fn(batch):
    src_batch, trg_batch = [], []
    for src_sample, trg_sample in batch:
        src_batch.append(src_sample)
        trg_batch.append(trg_sample)

    # Pad sequences to the max length in the batch
    src_padded = nn.utils.rnn.pad_sequence(src_batch, batch_first=True, padding_value=en_voc.vocab["<pad>"])
    trg_padded = nn.utils.rnn.pad_sequence(trg_batch, batch_first=True, padding_value=vi_voc.vocab["<pad>"])
    
    return src_padded, trg_padded

#splitting data
train_data, val_data = train_test_split(data, test_size=0.1, random_state=42)
train_dataset = Dts(train_data, en_voc, vi_voc)
val_dataset = Dts(val_data, en_voc, vi_voc)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=32, collate_fn=collate_fn)

#model
class attention(nn.Module):
    def __init__(self, enc_hidden_dim, dec_hidden_dim):
        self.attn_w = nn.Linear((enc_hidden_dim*2)+dec_hidden_dim, dec_hidden_dim)
        self.attn_v = nn.Linear(dec_hidden_dim,1 ,bias=False)

    def forward(self, dec_hidden, enc_outp):
        src_len = enc_outp.shape[0]
        dec_hidden_repp = dec_hidden.unsqueeze(0).repeat(src_len, 1, 1)
        total = torch.cat((dec_hidden_repp, enc_outp), dim=1)
        energy = torch.tanh(self.attn_v(total))
        attention = self.attn_v(energy).squeeze(2)
        return F.softmax(attention.permute(1, 0), dim=1)

class Encoder(nn.Module):
    def __init__(self, input_dim, embedding_dim, enc_hidden_dim,dec_hidden_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, enc_hidden_dim, n_layers, dropout=dropout, bidirectional=True)
        self.fc_h = nn.Liinrear(enc_hidden_dim*2, dec_hidden_dim)
        self.fc_c = nn.Liinrear(enc_hidden_dim*2, dec_hidden_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        emb = self.dropout(self.embedding(x))
        enc_outp, (hidden, cell) = self.lstm(emb)
        hidden = hidden.view(self.lstm.num_layers, 2, -1, self.lstm.hidden_size)
        cell = hidden.view(self.lstm.num_layers, 2, -1, self.lstm.hidden_size)
        hidden_cat = torch.cat(hidden[:, 0,:, :], hidden[:, 1,:, :])
        cell_cat = torch.cat(cell[:,0,:,:]), cell[:, 1,:,:]
        hidden = torch.tanh(self.fc_h(hidden_cat))
        cell = torch.tanh(self.hc_h(cell_cat))
        return enc_outp, hidden, cell

class Decoder(nn.Module):
    def __init__(self, output_dim, embedding_dim, enc_hidden_dim, dec_hidden_dim, n_layers, dropout, attention):
        super().__init__()
        self.output_dim = output_dim
        self.attention = attention
        self.embedding = nn.Embedding(output_dim, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim+(output_dim*2), dec_hidden_dim, n_layers, dropout=dropout)
        self.fc_out = nn.Linear((enc_hidden_dim*2)+dec_hidden_dim+embedding_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, hidden, cell, outp):
        emb = self.dropout(self.embedding(x.unsqueeze(0)))
        attention = hidden[-1, :,:]
        attention = self.attention(attention, outp).unsqeeze(1)
        enc_output_term = torch.permute(1, 0, 2)
        context = torch.bmm(attention, enc_output_term).permute(1, 0,2)
        lstm_input = torch.cat((emb, context), dim=2)
        output, ( h, c) = self.lstm(lstm_input, (hidden, cell))
        pre_outp = torch.cat((output, context, emb), dim=2)
        prediction = self.fc_out(pre_outp.squeeze(0))
        return prediction, hidden, cell, attention.squeeze(1)

class s2s(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, tgt):
        batch_size = tgt.shape[1]
        tgt_len = tgt.shape[0]
        tgt_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(batch_size, tgt_len, tgt_vocab_size).to(self.device)
        outp, (hidden, cell) = self.encoder(src)

        input_tok = tgt[0, :]
        for i in range(1, tgt_len):
            output, hidden, cell, _= self.decoder(input_tok, hidden, cell)
            output[i] = output

        return outputs

# Initialize model
INPUT_DIM = len(en_voc.vocab)
OUTPUT_DIM = len(vi_voc.vocab)
EMBED_DIM = 256
ENC_HIDDEN_DIM = 512
DEC_HIDDEN_DIM = 512
N_LAYERS = 2
DROPOUT = 0.5
LEARNING_RATE = 0.001
N_EPOCHS = 15

encoder = Encoder(INPUT_DIM, EMBED_DIM, ENC_HIDDEN_DIM, DEC_HIDDEN_DIM, N_LAYERS, DROPOUT).to(device)
decoder = Decoder(OUTPUT_DIM, EMBED_DIM, ENC_HIDDEN_DIM, DEC_HIDDEN_DIM, N_LAYERS, DROPOUT).to(device)
model = s2s(encoder, decoder, device).to(device)

def init_weights(m: nn.Module):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Embedding):
        nn.init.normal_(m.weight, mean=0, std=0.01)
        
criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training loop
def train_model(model, dataloader, optimizer, criterion, epochs, device=device):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch}", leave=False)
        for batch_idx, (src, tgt) in progress_bar:
            src, tgt= src.to(device), tgt.to(device) 
            optimizer.zero_grad()
            outp = model(src, tgt)
            loss = criterion(outp.reshape(-1, outp.size(-1)), tgt[:, 1:].reshape(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()
            total_loss += loss.item()
            avg_loss = total_loss / (batch_idx + 1)

            progress_bar.set_postfix({'loss': f'{avg_loss:.4f}'})
        
        print(f'Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}')


# inference
if __name__ =="__main__":
    train_model(model, train_loader, optimizer, criterion, epochs=10)

    model_dir = "./lstm"
    os.makedirs(model_dir, exist_ok=True)

    torch.save({
    'encoder_state_dict': encoder.state_dict(),
    'decoder_state_dict': decoder.state_dict(),
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    }, os.path.join(model_dir, "model_weights.pth"))

    with open(os.path.join(model_dir, "vocab.pkl"), 'wb') as f:
        pickle.dump({
            'en_vocab': en_voc.vocab,
            'vi_vocab': vi_voc.vocab,
            'en_inv_vocab': en_voc.inv_vocab,
            'vi_inv_vocab': vi_voc.inv_vocab
        }, f)

    model_info = { 
        'embedding_dim': 64,
        'hidden_dim': 128,
        'output_dim': 64,
        'max_len': 50,
        'en_vocab_size': len(en_voc.vocab),
        'vi_vocab_size': len(vi_voc.vocab)
    }

    with open(os.path.join(model_dir, "model_info.pkl"), 'wb') as f:
        pickle.dump(model_info, f)

    print("Model đã được lưu thành công!")


