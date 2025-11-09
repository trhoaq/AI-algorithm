import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.nn import Transformer
from tqdm import tqdm
import math, random, os, string, re, pickle
import pandas as pd
from nltk.corpus import stopwords
from collections import Counter
from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def tokenize(data, stopword):
    for sent in data:
        sent = str(sent).lower()
        token = re.sub(r'[^a-zA-Z0-9]',"", sent)
        token = re.findall(r"\w+", token)
        token = [word for word in token if word not in stopword]
    return token

def build_vocab(data, min_freq=2):
    en_counter = Counter()
    vi_counter = Counter()
    for _, row in data.iterrows():
        en_counter.update(row["english_sentence"])
        vi_counter.update(row["vietnamese_sentence"])
    en_vocab = Vocab(en_counter, min_freq)
    vi_vocab = Vocab(vi_counter, min_freq)
    return en_vocab, vi_vocab

class Vocab():
    def __init__(self, counter, min_freq):
        self.vocab = {"<PAD>":0, "<SOS>":1, "<EOS>":2, "<UNK>":3}
        self.inv_vocab = {0:"<PAD>", 1:"<SOS>", 2:"<EOS>", 3:"<UNK>"}
        self.pad_idx = 0
        self.sos_idx = 1
        self.eos_idx = 2
        self.unk_idx = 3

        idx = 4
        for token, cnt in counter.items():
            if cnt > min_freq:
                self.vocab[token] = idx
                self.inv_vocab[idx] = token
                idx += 1

    def __len__(self):
        return len(self.vocab)
    
class DatasetTrans(Dataset):
    def __init__(self,data, en_vocab, vi_vocab, ):
        self.data = data
        self.vi_vocab = vi_vocab
        self.en_vocab = en_vocab

    def __len__(self):
        return len(self.data)
        

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        en_id = [1] + \
            [self.en_vocab.vocab.get(t, 3) for t in row['english_sentence']]+\
            [2]
        vi_id = [1]+\
            [self.vi_vocab.vocab.get(t, 3) for t in row['vietnamese_sentence']]+\
            [2]
        
        return torch.LongTensor(en_id), torch.LongTensor(vi_id)

def collate_fn(batch):
    src_batch, targ_batch = [], []
    for src, targ in batch:
        src_batch.append(src)
        targ_batch.append(targ)

    src_batch = pad_sequence(src_batch, padding_value=0, batch_first=False)
    targ_batch = pad_sequence(targ_batch, padding_value=0, batch_first=False)

    return src_batch, targ_batch

class PositionalEnc(nn.Module):
    def __init__(self, emb_dim, dropout, maxlen=5000):
        super(PositionalEnc, self).__init__()
        den = torch.exp(torch.arange(0, emb_dim, 2)*math.log(10000)/emb_dim)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_emb = torch.zeros((maxlen, emb_dim))
        pos_emb[:, 0::2] = torch.sin(pos*den)
        pos_emb[:, 1::2] = torch.cos(pos*den)

        pos_emb = pos_emb.unsqueeze(-2)
        self.dropout = dropout
        self.register_buffer('pos_embbeding', pos_emb)

    def forward(self, tok_emb):
        return self.dropout(tok_emb, self.pos_emb[:tok_emb.size(0),:])
    
class TokenEmbedding(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super(TokenEmbedding, self).__init__()
        self.emb = nn.Embedding(input_dim, emb_dim)
        self.emb_size = emb_dim

    def forward(self, tokens):
        return self.emb(tokens.long()) * math.sqrt(self.emb_size)
    
class Seq2SeqTransformer(nn.Module):
    def __init__(self, input_dim, ouput_dim, emb_dim, n_head, num_enc_layers, num_dec_layers, ff_dim, dropout):
        super(Seq2SeqTransformer, self).__init__()
        self.transformer = Transformer(d_model=emb_dim,
                                       nhead = n_head,
                                       num_encoder_layers = num_enc_layers,
                                       num_decoder_layers = num_dec_layers,
                                       dim_feedforward = ff_dim,
                                       dropout = dropout,
                                       batch_first = False)
        
        self.gen = nn.Linear(emb_dim, ouput_dim)
        self.scr_embTok = TokenEmbedding(input_dim, emb_dim)
        self.targ_embTok = TokenEmbedding(ouput_dim, emb_dim)
        self.posEnc = PositionalEnc(emb_dim, dropout=dropout)

    def forward(self, src, targ, src_padding_mask, targ_padding_mask, tar_subsequent_mask):
        src_emd = self.posEnc(self.scr_embTok(src))
        targ_emb = self.posEnc(self.targ_embTok(targ))

        outp = self.transformer(src_emd, targ_emb,
                                src_key_padding_mask=src_padding_mask,
                                tgt_key_padding_mask=targ_padding_mask,
                                memory_key_padding_mask=src_padding_mask,
                                tgt_mask=tar_subsequent_mask)
        
        return self.gen(outp)
    
    def encoder(self, src, src_mask):
        return self.transformer.encoder(self.posEnc(self.scr_embTok(src)), src_mask)
    
    def decoder(self, targ, targ_mask):
        return self.transformer.decoder(self.posEnc(self.targ_embTok(targ)), targ_mask)
    
def init_weight(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0, std=0.01)

def gen_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=device))==1).transpose(0,1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask ==1, float(0.0))
    return mask

def create_mask(src, targ, pad_idx=0):
    src_seq_mask = src.shape[0]
    targ_seq_mask = targ.shape[0]
    targ_mask = gen_subsequent_mask(targ_seq_mask)

    src_padding_mask = (src==0).transpose(0,1)
    targ_padding_mask = (targ==0).transpose(0,1)

    return src_padding_mask, targ_padding_mask, targ_mask 

def train_model(model, dataloader, opti, loss, clip, pad_idx=0):
    model.train()
    total_loss = 0
    for src, targ in tqdm(dataloader, desc='training'):
        src = src.to(device)
        targ = targ.to(device)

        targ_in = targ[:-1,:]
        targ_out = targ[1:,:]

        src_padding_mask, targ_padding_mask, targ_mask = create_mask(src, targ)

        opti.zero_grad()
        logits = model(src, targ_in, src_padding_mask, targ_padding_mask, targ_mask)
        outp_dim = logits.shape[-1]
        loss = loss(logits.reshape(-1, outp_dim), targ_out.reshape(-1))
        loss.backward()
        nn.ultils.clip_grad_norm_(model.parameter(), clip)
        opti.step()
        total_loss += loss.item()

    return total_loss/len(dataloader)

if __name__ == "__main__":
    set_seed(36)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    vi_stopword = [
    "và", "là", "có", "của", "trong", "một", "những", "được", 
    "với", "cho", "khi", "này", "đó", "thì", "ra", "đã", "cũng"
    ]

    en_stopword = list(stopwords.words("english"))

    root = "./dts/machine_trans/train"
    bilingual = os.path.join(root, "song_ngu.csv")
    vi = os.path.join(root, "vi.csv")
    en = os.path.join(root, "en.csv")
    data = pd.read_csv(bilingual)

    data = data.dropna().reset_index(drop=True)
    tokenize(data=data["vietnamese_sentence"], stopword=vi_stopword)
    tokenize(data=data["english_sentence"], stopword=en_stopword)

    en_vocab, vi_vocab = build_vocab(data)

    train, val = train_test_split(data, test_size=0.8)
    trainset = DatasetTrans(train, en_vocab, vi_vocab)
    valset = DatasetTrans(val, en_vocab, vi_vocab)
    trainload = DataLoader(trainset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    valload = DataLoader(valset, batch_size=32, collate_fn=collate_fn)

    INP_DIM = len(en_vocab)
    OUTP_DIM = len(vi_vocab)
    EMB_SIZE = 256
    NHEAD = 8
    FFN_HID_DIM = 512
    NUM_ENCODER_LAYERS = 3
    NUM_DECODER_LAYERS = 3
    DROPOUT = 0.1
    LR = 1e-4
    EPOCH = 20
    CLIP = 1
    
    model = Seq2SeqTransformer(INP_DIM, OUTP_DIM, EMB_SIZE, NHEAD, NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, FFN_HID_DIM, DROPOUT).to(device)
    model.apply(init_weight)

    optimizer = optim.Adam(model.parameters(), lr=LR)
    loss = nn.CrossEntropyLoss(ignore_index=0)

    for epoch in range(EPOCH):
        train_loss = train_model(model, trainload, optimizer, loss, CLIP)

        best_loss = 100
        time = 2
        if train_loss<best_loss:
            best_loss = train_loss
            torch.save(model.state_dict(), "./info/model.pt")
        else :
            time -= 1

        if time ==0:
            break
        
    with open("./info/envo.pkl", 'wb') as f:
        pickle.dump(en_vocab, f)
    with open("./info/envo.pkl", 'wb') as f:
        pickle.dump(vi_vocab, f)
            