# Install required libraries
# pip install torchtext==0.6.0 # Version compatible with legacy datasets
# pip install torch>=1.6.0
# pip install spacy
# python -m spacy download en
# python -m spacy download de
# pip install nltk

import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator
import spacy
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# Load spaCy tokenizers
spacy_en = spacy.load("en_core_web_sm")
spacy_de = spacy.load("de_core_web_sm")

# Define tokenizer functions
def tokenize_de(text):
    """
    Tokenizes German text from a string into a list of strings
    """
    return [tok.text for tok in spacy_de.tokenizer(text)]

def tokenize_en(text):
    """
    Tokenizes English text from a string into a list of strings
    """
    return [tok.text for tok in spacy_en.tokenizer(text)]

# Define torchtext fields
SRC = Field(tokenize=tokenize_de, init_token='<sos>', eos_token='<eos>', lower=True)
TRG = Field(tokenize=tokenize_en, init_token='<sos>', eos_token='<eos>', lower=True)

# Load the Multi30k dataset
train_data, valid_data, test_data = Multi30k.splits(exts=('.de', '.en'), fields=(SRC, TRG))

# Build the vocabulary
SRC.build_vocab(train_data, min_freq=2)
TRG.build_vocab(train_data, min_freq=2)

# Create iterators
BATCH_SIZE = 64
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size = BATCH_SIZE,
    device = device)
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, dropout):
        super().__init__()
        
        self.hid_dim = hid_dim
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim, hid_dim, bidirectional=True)
        self.fc = nn.Linear(hid_dim * 2, hid_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src):
        # src = [src len, batch size]
        embedded = self.dropout(self.embedding(src))
        # embedded = [src len, batch size, emb dim]
        outputs, hidden = self.rnn(embedded)
        # outputs = [src len, batch size, hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]
        
        # Initial hidden state for the decoder is a linear transformation of the final hidden states of the bi-directional encoder
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)))
        
        return outputs, hidden

class BahdanauAttention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        
        self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias=False)
        
    def forward(self, hidden, encoder_outputs):
        # hidden = [1, batch size, dec hid dim]
        # encoder_outputs = [src len, batch size, enc hid dim * 2]
        
        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]
        
        # Repeat decoder hidden state src_len times
        hidden = hidden.repeat(src_len, 1, 1)
        
        # Swap axes to make concatenation work
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        hidden = hidden.permute(1, 0, 2)
        
        # Calculate scores
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        
        # Calculate attention weights
        attention = self.v(energy).squeeze(2)
        
        # Return softmax'd attention weights
        return nn.functional.softmax(attention, dim=1)

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout, attention):
        super().__init__()
        
        self.output_dim = output_dim
        self.attention = attention
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.GRU((enc_hid_dim * 2) + emb_dim, dec_hid_dim)
        self.fc_out = nn.Linear((enc_hid_dim * 2) + dec_hid_dim + emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden, encoder_outputs):
        # input = [batch size]
        # hidden = [1, batch size, dec hid dim]
        # encoder_outputs = [src len, batch size, enc hid dim * 2]
        
        input = input.unsqueeze(0)
        # input = [1, batch size]
        
        embedded = self.dropout(self.embedding(input))
        # embedded = [1, batch size, emb dim]
        
        # Calculate attention weights
        attn_weights = self.attention(hidden, encoder_outputs)
        # attn_weights = [batch size, src len]
        
        # Reformat encoder outputs
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        # encoder_outputs = [batch size, src len, enc hid dim * 2]
        
        # Create weighted context vector
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).permute(1, 0, 2)
        # context = [1, batch size, enc hid dim * 2]
        
        rnn_input = torch.cat((embedded, context), dim=2)
        # rnn_input = [1, batch size, (enc hid dim * 2) + emb dim]
        
        output, hidden = self.rnn(rnn_input, hidden)
        # output = [1, batch size, dec hid dim]
        # hidden = [1, batch size, dec hid dim]
        
        # Concatenate for final output layer
        output = torch.cat((output.squeeze(0), context.squeeze(0), embedded.squeeze(0)), dim=1)
        prediction = self.fc_out(output)
        
        return prediction, hidden

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
    def forward(self, src, trg, teacher_forcing_ratio = 0.5):
        # src = [src len, batch size]
        # trg = [trg len, batch size]
        
        trg_len, batch_size = trg.shape
        trg_vocab_size = self.decoder.output_dim
        
        # Store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        
        encoder_outputs, hidden = self.encoder(src)
        
        # First input to the decoder is the <sos> token
        input = trg[0, :]
        
        for t in range(1, trg_len):
            output, hidden = self.decoder(input, hidden, encoder_outputs)
            outputs[t] = output
            
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = output.argmax(1) 
            input = trg[t] if teacher_force else top1
            
        return outputs
      INPUT_DIM = len(SRC.vocab)
OUTPUT_DIM = len(TRG.vocab)
ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
ENC_HID_DIM = 512
DEC_HID_DIM = 512
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5

attn = BahdanauAttention(ENC_HID_DIM, DEC_HID_DIM)
enc = Encoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, ENC_DROPOUT)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT, attn)

model = Seq2Seq(enc, dec, device).to(device)

def init_weights(m):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)

model.apply(init_weights)

optimizer = optim.Adam(model.parameters(), lr=0.001)

TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]
criterion = nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX)

def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    for i, batch in enumerate(iterator):
        src = batch.src
        trg = batch.trg
        
        optimizer.zero_grad()
        
        output = model(src, trg)
        
        # trg = [(trg len - 1) * batch size]
        # output = [(trg len - 1) * batch size, output dim]
        output_dim = output.shape[-1]
        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)
        
        loss = criterion(output, trg)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        
        epoch_loss += loss.item()
    return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch.src
            trg = batch.trg
            output = model(src, trg, 0) # No teacher forcing
            
            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)
            
            loss = criterion(output, trg)
            epoch_loss += loss.item()
    return epoch_loss / len(iterator)

def calculate_bleu(model, data, trg_field):
    trg_pad_idx = trg_field.vocab.stoi[trg_field.pad_token]
    trg_eos_idx = trg_field.vocab.stoi[trg_field.eos_token]
    
    trg_vocab = trg_field.vocab
    
    total_bleu = 0
    total_sentences = 0
    
    with torch.no_grad():
        for batch in data:
            src = batch.src
            trg = batch.trg
            
            output_tokens = model.translate(src) # Assuming a translate method exists
            
            for i in range(len(output_tokens)):
                reference = [trg_vocab.itos[token] for token in trg[:, i] if token not in [trg_pad_idx, trg_eos_idx]]
                
                # Check for empty reference to avoid errors
                if not reference:
                    continue
                    
                hypothesis = [trg_vocab.itos[token] for token in output_tokens[i] if token not in [trg_pad_idx, trg_eos_idx]]
                
                total_bleu += sentence_bleu([reference], hypothesis, smoothing_function=SmoothingFunction().method1)
                total_sentences += 1
                
    return total_bleu / total_sentences if total_sentences > 0 else 0

N_EPOCHS = 40
CLIP = 1

for epoch in range(N_EPOCHS):
    train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
    valid_loss = evaluate(model, valid_iterator, criterion)
    print(f'Epoch: {epoch+1:02} | Train Loss: {train_loss:.3f} | Val. Loss: {valid_loss:.3f}')

# Placeholder for BLEU calculation, a full implementation would require a `translate` method in the model
# bleu_score = calculate_bleu(model, test_iterator, TRG)
# print(f'BLEU Score on test set: {bleu_score:.4f}')
