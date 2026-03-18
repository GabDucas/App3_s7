# GRO722 Laboratoire 2
# Auteur: Jean-Samuel Lauzon et  Jonathan Vincent
# Hivers 2021

import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

class Seq2seq(nn.Module):
    def __init__(self, n_hidden, n_layers, int2symb, symb2int, dict_size, device, max_len):
        super(Seq2seq, self).__init__()

        # Definition des paramètres
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.device = device
        self.symb2int = symb2int
        self.int2symb = int2symb
        self.dict_size = dict_size
        self.max_len = max_len

        # Définition des couches du rnn
        self.fr_embedding = nn.Embedding(self.dict_size['fr'], n_hidden)
        self.en_embedding = nn.Embedding(self.dict_size['en'], n_hidden)
        self.encoder_layer = nn.GRU(n_hidden, n_hidden, n_layers, batch_first=True)
        self.decoder_layer = nn.GRU(n_hidden, n_hidden, n_layers, batch_first=True)

        # Définition de la couche dense pour la sortie
        self.fc = nn.Linear(n_hidden, self.dict_size['en'])
        self.to(device)
        
    def encoder(self, x):
        # Encodeur

        # ---------------------- Laboratoire 2 - Question 3 - Début de la section à compléter -----------------
        fr_embedded = self.fr_embedding(x)
        out, hidden = self.encoder_layer(fr_embedded)
        
        # ---------------------- Laboratoire 2 - Question 3 - Fin de la section à compléter -----------------

        return out, hidden

    
    def decoder(self, encoder_outs, hidden):
        # Initialisation des variables
        max_len = self.max_len['en'] # Longueur max de la séquence anglaise (avec padding)
        batch_size = hidden.shape[1] # Taille de la batch
        vec_in = torch.zeros((batch_size, 1)).to(self.device).long() # Vecteur d'entrée pour décodage 
        vec_out = torch.zeros((batch_size, max_len, self.dict_size['en'])).to(self.device) # Vecteur de sortie du décodage

        # Boucle pour tous les symboles de sortie
        for i in range(max_len):

            # ---------------------- Laboratoire 2 - Question 3 - Début de la section à compléter -----------------   
            en_embedded = self.en_embedding(vec_in)
            out, hidden = self.decoder_layer(en_embedded, hidden)
            out_lin = self.fc(out)
            vec_in = out_lin.argmax(dim=2)
            vec_out[:,i,:] = out_lin.squeeze()

            # ---------------------- Laboratoire 2 - Question 3 - Début de la section à compléter -----------------

        return vec_out, hidden, None

    def forward(self, x):
        # Passant avant
        out, h = self.encoder(x)
        out, hidden, attn = self.decoder(out,h)
        return out, hidden, attn


class Seq2seq_attn(nn.Module):
    def __init__(self, n_hidden, n_layers, int2symb, symb2int, dict_size, device, max_len):
        super(Seq2seq_attn, self).__init__()

        # Definition des paramètres
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.device = device
        self.symb2int = symb2int
        self.int2symb = int2symb
        self.dict_size = dict_size
        self.max_len = max_len

        self.encoder_layer = nn.GRU(2, n_hidden, n_layers, batch_first=True)
        
        # Embedding décodeur
        self.embedding = nn.Embedding(dict_size, n_hidden)

        # Décodeur
        self.decoder_layer = nn.GRU(n_hidden, n_hidden, n_layers, batch_first=True)

        # Couches attention
        self.att_combine = nn.Linear(2*n_hidden, n_hidden)
        self.hidden2query = nn.Linear(n_hidden, n_hidden)

        # Sortie
        self.fc = nn.Linear(n_hidden, dict_size)
        
    def encoder(self, x):

        out, hidden = self.encoder_layer(x)
        return out, hidden

    def attentionModule(self, query, values):
        query = self.hidden2query(query)

        attention = torch.bmm(query, values.permute(0,2,1))
        attention_weights = torch.softmax(attention[:,0,:], dim=1)

        attention_weights_repeat = attention_weights[:,:,None].repeat(1,1,self.n_hidden)
        attention_output = torch.sum(attention_weights_repeat * values, dim=1)

        return attention_output, attention_weights

    def decoderWithAttn(self, encoder_outs, hidden):
        max_len = self.max_len
        batch_size = hidden.shape[1]

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

            out = self.att_combine(torch.cat([buffer[:,0,:], attention_out], dim=1))
            out_lin = self.fc(out)

            vec_out[:,i,:] = out_lin
            vec_in = torch.argmax(out_lin.unsqueeze(1), dim=2)

        return vec_out, hidden, attention_weights


    def forward(self, x):
        # Passe avant
        out, h = self.encoder(x)
        out, hidden, attn = self.decoderWithAttn(out,h)
        return out, hidden, attn
