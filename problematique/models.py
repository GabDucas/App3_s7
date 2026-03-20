# GRO722 problématique
# Auteur: Jean-Samuel Lauzon et  Jonathan Vincent
# Hivers 2021

import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

class trajectory2seq(nn.Module):
    def __init__(self, hidden_dim, n_layers, int2symb, symb2int, dict_size, device, max_len):
        super(trajectory2seq, self).__init__()
        # Definition des parametres
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.device = device
        self.symb2int = symb2int
        self.int2symb = int2symb
        self.dict_size = dict_size
        self.max_len = max_len

        # Definition des couches
        # Couches pour rnn
        # TODO
        self.encoder_layer = nn.GRU(
            input_size=2,              # coordonnées (x,y)
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True
        )

        self.embedding = nn.Embedding(
            num_embeddings=dict_size,
            embedding_dim=hidden_dim
        )

        self.decoder_layer = nn.GRU(
            input_size=hidden_dim,
            hidden_size=2*hidden_dim,
            num_layers=n_layers,
            batch_first=True
        )

        # Couches pour attention
        # TODO
        self.att_combine = nn.Linear(4*hidden_dim, hidden_dim)
        self.hidden2query = nn.Linear(2*hidden_dim, 2*hidden_dim)

        # Couche dense pour la sortie
        # TODO
        self.fc = nn.Linear(hidden_dim, dict_size)

    def encoder(self, x):
        out, hidden = self.encoder_layer(x)
        hidden = torch.cat((hidden[0], hidden[1]), dim=1)  # (batch, 2H)
        hidden = hidden.unsqueeze(0) 
        return out, hidden

    def attentionModule(self, query, values):
        #query.shape = (batch_size, 1, hidden_dim)
        #values.shape = (batch_size, seq_len_in, hidden_dim)

        query = self.hidden2query(query)
        
        #values.shape = (batch_size, 1, seq_len_in)
        #attention.shape = (batch_size, 1, seq_len_in)
        attention = torch.bmm(query, values.permute(0,2,1))

        attention_weights = torch.softmax(attention[:,0,:], dim=1)

        attention_weights_repeat = attention_weights[:,:,None].repeat(1,1,2*self.hidden_dim)
        attention_output = torch.sum(attention_weights_repeat * values, dim=1)

        #attention_output  : (batch, hidden_dim)
        #attention_weights : (batch, seq_len_in)

        return attention_output, attention_weights
    
    def decoderWithAttn(self, encoder_outs, hidden, target_seq):
        #encoder_outs = (batch, seq_len_in, hidden_dim)
        #hidden = (num_layers, batch, hidden_dim)

        max_len = target_seq.size(1)
  
        batch_size = hidden.shape[1]

        #vec_in = (batch,1)
        vec_in = torch.full((batch_size,1),
                            self.symb2int['<sos>'],
                            dtype=torch.long).to(self.device)

        vec_out = torch.zeros((batch_size, max_len, self.dict_size)).to(self.device)
        attention_weights = torch.zeros((batch_size, encoder_outs.shape[1], max_len)).to(self.device)

        for i in range(max_len):

            embedded = self.embedding(vec_in)
            buffer, hidden = self.decoder_layer(embedded, hidden)

            attention_out, attention_weight_buff = self.attentionModule(buffer, encoder_outs)
            attention_weights[:,:,i] = attention_weight_buff

            #torch.cat... = (batch, hidden) + (batch, hidden) → (batch, hidden×2)
            #att_combine (batch, hidden×2) → (batch, hidden)
            out = self.att_combine(torch.cat([buffer[:,0,:], attention_out], dim=1))
            out_lin = self.fc(out)

            vec_out[:,i,:] = out_lin
      
            vec_in = target_seq[:, i].unsqueeze(1)  


        return vec_out, hidden, attention_weights

    def forward(self, x, target_seq):
        out, h = self.encoder(x)
        out, hidden, attn = self.decoderWithAttn(out,h, target_seq)
        return out, hidden, attn
    

