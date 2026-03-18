import torch
import numpy as np
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import re
import pickle

class HandwrittenWords(Dataset):
    """Ensemble de donnees de mots ecrits a la main."""

    def __init__(self, filename):
        # Lecture du text
        self.pad_symbol     = pad_symbol = '<pad>'
        self.start_symbol   = start_symbol = '<sos>'
        self.stop_symbol    = stop_symbol = '<eos>'

        self.data = dict()
        self.seq_buffers = dict()
        with open(filename, 'rb') as fp:
            self.data = pickle.load(fp)
        
        print("first word : ", self.data[0][0])
        print("first trajectory : ", self.data[0][1][0][0], ",", self.data[0][1][1][0])
        # data[x] = (word, (x_trajectory, y_trajectory))
        # data[x][0] = word
        # data[x][1] = (x_trajectory, y_trajectory)
        # data[x][1][0] = x_trajectory array
        # data[x][1][1] = y_trajectory array

        # Extraction des symboles
        # TODO
        self.symb2int = dict()
        self.symb2int = {start_symbol:0, stop_symbol:1, pad_symbol:2}
        cpt_symb = 3

        for i in range(len(self.data)):
            word = self.data[i][0]
            for symb in word:
                if symb not in self.symb2int:
                    self.symb2int[symb] = cpt_symb
                    cpt_symb += 1

        self.int2symb = dict()
        self.int2symb = {v:k for k,v in self.symb2int.items()}

        # Ajout du padding aux séquences
        # TODO
        # self.max_len = max([len(i) for i in self.data[:][0]]) + 1 # EOS -- could be hardcoded, 5 max anyway
        self.max_len = 6 # 5 + EOS
        print(self.max_len)

        for i in range(len(self.data)):
            word = self.data[i][0]
            buffer = []
            if word != "<sos>":
                for symb in word:
                    buffer.append(self.symb2int[symb])
            buffer.append(self.symb2int[stop_symbol])
            while len(buffer) < self.max_len:
                buffer.append(self.symb2int[pad_symbol])
            self.seq_buffers[word] = buffer

        # for i in range(len(self.data)):
        #     word = self.data[i][0]
        #     buffer = []
        #     buffer.append(self.symb2int[start_symbol])
        #     if word != "<sos>":
        #         for symb in word:
        #             buffer.append(self.symb2int[symb])
        #     buffer.append(self.symb2int[stop_symbol])
        #     while len(buffer) < self.max_len:
        #         buffer.append(self.symb2int[pad_symbol])
        #     self.seq_buffers[word] = buffer
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # TODO
        return None, None

    def visualisation(self, idx):
        # Visualisation des échantillons
        # TODO (optionel)
        seq = self.data[idx]
        # seq = [self.int2symb[i] for i in seq]
        print('Word : ', seq[0])
        print('First coordinates : ', seq[1][0][0], ',', seq[1][1][0])
        print('Symbols : ', self.seq_buffers[seq[0]])
        print('Int2Symb : ', [self.int2symb[symb] for symb in self.seq_buffers[seq[0]]] )
        print('---')
        

if __name__ == "__main__":
    # Code de test pour aider à compléter le dataset
    a = HandwrittenWords('problematique/data_trainval.p')
    for i in range(10):
        a.visualisation(np.random.randint(0, len(a)))