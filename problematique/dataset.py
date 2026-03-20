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
        self.pad_targ = []
        self.pad_traj = []
        with open(filename, 'rb') as fp:
            self.data = pickle.load(fp)
        
        # ===============================================
        # data[x] = (word, (x_trajectory, y_trajectory))
        # data[x][0] = word
        # data[x][1] = (x_trajectory, y_trajectory)
        # data[x][1][0] = x_trajectory array
        # data[x][1][1] = y_trajectory array
        # ===============================================

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
        self.max_len_traj = max([len(item[1][0]) for item in self.data]) + 1
        self.max_len_word = 6 # 5 + EOS

        for i in range(len(self.data)):
            word, (x, y) = self.data[i]
            x = (x - np.mean(x)) / np.std(x)  # Normalisation
            y = (y - np.mean(y)) / np.std(y)

            # Padding trajectory
            traj_buffer = np.zeros((self.max_len_traj, 2))
            actual_len = len(x)
            traj_buffer[:,0] = x[actual_len-1]
            traj_buffer[:,1] = y[actual_len-1]
            traj_buffer[:actual_len, 0] = x
            traj_buffer[:actual_len, 1] = y
            self.pad_traj.append(torch.tensor(traj_buffer, dtype=torch.float32))

            # Padding target
            targ_buffer = []
            if word != "<sos>":
                for symb in word:
                    targ_buffer.append(self.symb2int[symb])
            targ_buffer.append(self.symb2int[stop_symbol])

            while len(targ_buffer) < self.max_len_word:
                targ_buffer.append(self.symb2int[pad_symbol])
            self.pad_targ.append(torch.tensor(targ_buffer, dtype=torch.long))
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        traj = self.pad_traj[idx]  # Shape: [seq_len, 2]
        target = self.pad_targ[idx]

        # Scaling aléatoire
        scale = torch.empty(1).uniform_(0.95, 1.05).item()
        traj = traj * scale
        
        # Ajout de bruit gaussien
        noise = torch.randn_like(traj) * 0.002
        traj = traj + noise

        return traj, target

    def visualisation(self, idx):
        word_str, (x_coords, y_coords) = self.data[idx]
        target_tensor = self.pad_targ[idx]
        # .tolist() converts the tensor indices [3, 15, 1, 2...] into a standard Python list
        target_indices = target_tensor.tolist()
        symbols = [self.int2symb[symb] for symb in target_indices]

        print(f"--- Sample Index: {idx} ---")
        print(f"Original Label:  {word_str}")
        print(f"Symbols:         {symbols}")
        print(f"Indices:         {target_indices}")
        print(f"Trajectory Len:  {len(x_coords)} points")
        print(f"First Coord:     ({x_coords[0]:.2f}, {y_coords[0]:.2f})")
        print('---')
        

# def indices_to_text(indices_batch, int2symb):
#     """
#     indices_batch: Tensor (batch_size, seq_len)
#     int2symb: dictionnaire int->symbole
#     """
#     words = []
#     for seq in indices_batch:
#         chars = []
#         for idx in seq:
#             symb = int2symb[idx.item()]
#             if symb == '<eos>':
#                 break
#             if symb not in ['<pad>', '<sos>']:
#                 chars.append(symb)
#         words.append(''.join(chars))
#     return words

if __name__ == "__main__":
    # Code de test pour aider à compléter le dataset
    a = HandwrittenWords('problematique/data_trainval.p')
    for i in range(10):
        a.visualisation(np.random.randint(0, len(a)))