import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.nn.utils.rnn import pad_sequence
from torch.nn import Transformer

import random
import string
import pandas as pd
from collections import Counter
from tqdm import tqdm
import math
import time
import os
import pickle
import re
from nltk.corpus import stopwords

#set seed
SEED = 1234
random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"--- Sử dụng thiết bị: {device} ---")

#tokenizer
translator = str.maketrans('', '', string.punctuation)

def tokenize_simple(sent):
    stopword = stopwords.words('english')
    sent = str(sent).lower()
    token = re.sub(r'[^a-zA-Z0-9]',"", sent)
    token = re.findall(r"\w+", token)
    token = [word for word in token if word not in stopword]
    return token

#model
class Vocab:
    def __init__(self, counter, min_freq=2):
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.PAD_IDX = 0
        self.SOS_IDX = 1
        self.EOS_IDX = 2
        self.UNK_IDX = 3
        idx = 4
        for token, count in counter.items():
            if count >= min_freq:
                self.stoi[token] = idx
                self.itos[idx] = token
                idx += 1
    def __len__(self):
        return len(self.itos)

class PositionalEncoding(nn.Module):
    def __init__(self, emb_size: int, dropout: float, maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(-torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)
    def forward(self, token_embedding: torch.Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size
    def forward(self, tokens: torch.Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)

class Seq2SeqTransformer(nn.Module):
    def __init__(self, num_encoder_layers: int, num_decoder_layers: int,
                 emb_size: int, nhead: int, vocab_size: int, # Chỉ cần 1 vocab_size
                 dim_feedforward: int = 512, dropout: float = 0.1):
        super(Seq2SeqTransformer, self).__init__()

        # Dùng vocab_size cho cả src và tgt
        self.transformer = Transformer(d_model=emb_size,
                                       nhead=nhead,
                                       num_encoder_layers=num_encoder_layers,
                                       num_decoder_layers=num_decoder_layers,
                                       dim_feedforward=dim_feedforward,
                                       dropout=dropout,
                                       batch_first=False)
        self.generator = nn.Linear(emb_size, vocab_size)
        self.src_tok_emb = TokenEmbedding(vocab_size, emb_size)
        self.tgt_tok_emb = TokenEmbedding(vocab_size, emb_size) # Dùng chung
        self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout)

    def forward(self, src: torch.Tensor, trg: torch.Tensor,
                src_padding_mask: torch.Tensor, tgt_padding_mask: torch.Tensor,
                tgt_subsequent_mask: torch.Tensor):
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg))
        outs = self.transformer(src_emb, tgt_emb,
                                src_key_padding_mask=src_padding_mask,
                                tgt_key_padding_mask=tgt_padding_mask,
                                memory_key_padding_mask=src_padding_mask,
                                tgt_mask=tgt_subsequent_mask)
        return self.generator(outs)

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor):
        return self.transformer.encoder(self.positional_encoding(self.src_tok_emb(src)), src_key_padding_mask=src_mask)

    def decode(self, tgt: torch.Tensor, memory: torch.Tensor, tgt_mask: torch.Tensor, memory_mask: torch.Tensor):
        return self.transformer.decoder(self.positional_encoding(self.tgt_tok_emb(tgt)), memory,
                                        tgt_mask=tgt_mask, memory_key_padding_mask=memory_mask)

#data
def load_summarization_data():
    df = pd.read_csv("hf://datasets/Tommy0201/Kaggle_CNN_Text_Summarization/" + splits["train"])
    df.drop(columns=["id"])
    df['text'] = df['article']
    df['summary'] = df['highlights']
    # Đảm bảo tên cột là 'text' và 'summary'
    df = df[['text', 'summary']].dropna().reset_index(drop=True)
    print("--- Tokenizing dữ liệu (có thể mất thời gian)... ---")
        # Tokenize (Sử dụng hàm đơn giản - nên thay thế)
    df['src_tokens'] = df['text'].apply(tokenize_simple)
    df['tgt_tokens'] = df['summary'].apply(tokenize_simple)

    print(f"--- Tải xong {len(df)} cặp text/summary ---")
    return df

def build_shared_vocab(train_data, min_freq=2):
    shared_counter = Counter()
    for tokens in train_data['src_tokens']:
        shared_counter.update(tokens)
    for tokens in train_data['tgt_tokens']:
        shared_counter.update(tokens)

    vocab = Vocab(shared_counter, min_freq)
    print(f"share vocab: {len(vocab)}")
    return vocab

class SummarizationDataset(data.Dataset):
    def __init__(self, df, vocab, max_len_src=512, max_len_tgt=128):
        self.df = df
        self.vocab = vocab
        self.max_len_src = max_len_src
        self.max_len_tgt = max_len_tgt

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Cắt bớt (Truncate) nếu quá dài
        src_tokens = row['src_tokens'][:self.max_len_src - 2] # Trừ 2 cho SOS/EOS
        tgt_tokens = row['tgt_tokens'][:self.max_len_tgt - 2]

        src_ids = [self.vocab.SOS_IDX] + \
                  [self.vocab.stoi.get(t, self.vocab.UNK_IDX) for t in src_tokens] + \
                  [self.vocab.EOS_IDX]

        tgt_ids = [self.vocab.SOS_IDX] + \
                  [self.vocab.stoi.get(t, self.vocab.UNK_IDX) for t in tgt_tokens] + \
                  [self.vocab.EOS_IDX]

        return torch.tensor(src_ids, dtype=torch.long), torch.tensor(tgt_ids, dtype=torch.long)

def collate_fn(batch):
    src_batch, trg_batch = [], []
    for src_item, trg_item in batch:
        src_batch.append(src_item)
        trg_batch.append(trg_item)
    src_batch = pad_sequence(src_batch, padding_value=0, batch_first=False) # PAD_IDX = 0
    trg_batch = pad_sequence(trg_batch, padding_value=0, batch_first=False)
    return src_batch, trg_batch

#train
def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def create_mask(src, tgt, PAD_IDX):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]
    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_padding_mask = (src == PAD_IDX).transpose(0, 1)
    tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)
    return src_padding_mask, tgt_padding_mask, tgt_mask

def train(model, iterator, optimizer, criterion, clip, PAD_IDX):
    model.train()
    epoch_loss = 0
    for src, trg in tqdm(iterator, desc="Training"):
        src = src.to(device)
        trg = trg.to(device)
        trg_input = trg[:-1, :]
        trg_out = trg[1:, :]
        src_padding_mask, tgt_padding_mask, tgt_mask = create_mask(src, trg_input, PAD_IDX)
        optimizer.zero_grad()
        logits = model(src, trg_input, src_padding_mask, tgt_padding_mask, tgt_mask)
        output_dim = logits.shape[-1]
        loss = criterion(logits.reshape(-1, output_dim), trg_out.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion, PAD_IDX):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for src, trg in tqdm(iterator, desc="Evaluating"):
            src = src.to(device)
            trg = trg.to(device)
            trg_input = trg[:-1, :]
            trg_out = trg[1:, :]
            src_padding_mask, tgt_padding_mask, tgt_mask = create_mask(src, trg_input, PAD_IDX)
            logits = model(src, trg_input, src_padding_mask, tgt_padding_mask, tgt_mask)
            output_dim = logits.shape[-1]
            loss = criterion(logits.reshape(-1, output_dim), trg_out.reshape(-1))
            epoch_loss += loss.item()
    return epoch_loss / len(iterator)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def main():
    MODEL = 'summarizer_model.pt'
    VOCAB = 'summarizer_vocab.pkl'
    ROOT_MODEL_PATH = './model'
    VOCAB_PATH = os.path.join(ROOT_MODEL_PATH, VOCAB)
    MODEL_SAVE_PATH = os.path.join(ROOT_MODEL_PATH, MODEL)

    df_all_data = load_summarization_data()
    if df_all_data is None:
        return

    #train/val
    df_valid = df_all_data.sample(frac=0.1, random_state=SEED)
    df_train = df_all_data.drop(df_valid.index)
    vocab = build_shared_vocab(df_train, min_freq=5)

    PAD_IDX = vocab.PAD_IDX
    BATCH_SIZE = 16
    train_dataset = SummarizationDataset(df_train, vocab, max_len_src=512, max_len_tgt=128)
    valid_dataset = SummarizationDataset(df_valid, vocab, max_len_src=512, max_len_tgt=128)
    train_iterator = data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    valid_iterator = data.DataLoader(valid_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    VOCAB_SIZE = len(vocab)
    EMB_SIZE = 256
    NHEAD = 8
    FFN_HID_DIM = 512
    NUM_ENCODER_LAYERS = 3
    NUM_DECODER_LAYERS = 3
    DROPOUT = 0.1

    model = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE,
                               NHEAD, VOCAB_SIZE,
                               FFN_HID_DIM, DROPOUT).to(device)

    def init_weights(m: nn.Module):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0, std=0.01)
    model.apply(init_weights)
    print(f'parameter: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}')

    optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    #train
    N_EPOCHS = 20
    CLIP = 1
    best_valid_loss = float('inf')
    patience = 3
    patience_counter = 0

    for epoch in range(N_EPOCHS):
        start_time = time.time()
        train_loss = train(model, train_iterator, optimizer, criterion, CLIP, PAD_IDX)
        valid_loss = evaluate(model, valid_iterator, criterion, PAD_IDX)
        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f}')

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"stop at epoch {epoch+1} ---")
            break


    with open(VOCAB_PATH, 'wb') as f:
        pickle.dump(vocab, f)

if __name__ == '__main__':
    main()
