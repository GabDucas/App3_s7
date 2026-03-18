# GRO722 problématique
# Auteur: Jean-Samuel Lauzon et  Jonathan Vincent
# Hivers 2021

import torch
from torch import nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from models import *
from dataset import *
from metrics import *

if __name__ == '__main__':

    # ---------------- Paramètres et hyperparamètres ----------------#
    force_cpu = False           # Forcer a utiliser le cpu?
    trainning = True           # Entrainement?
    test = True                # Test?
    learning_curves = True     # Affichage des courbes d'entrainement?
    gen_test_images = True     # Génération images test?
    seed = 1                # Pour répétabilité
    n_workers = 0           # Nombre de threads pour chargement des données (mettre à 0 sur Windows)

    # TODO
    n_epochs = 0

    # ---------------- Fin Paramètres et hyperparamètres ----------------#

    # Initialisation des variables
    if seed is not None:
        torch.manual_seed(seed) 
        np.random.seed(seed)

    # Choix du device
    device = torch.device("cuda" if torch.cuda.is_available() and not force_cpu else "cpu")

    # Instanciation de l'ensemble de données
    # TODO
    dataset = HandwrittenWords(n_samp=4000, samplelen=[6,10])

    
    # Séparation de l'ensemble de données (entraînement et validation)
    # TODO
    dataset_train, dataset_val = torch.utils.data.random_split(dataset, [3000, 1000])

    # Instanciation des dataloaders
    # TODO
    dataload_train = DataLoader(dataset_train, batch_size=100, shuffle=True, num_workers=n_workers)
    dataload_val = DataLoader(dataset_val, batch_size=100, shuffle=False, num_workers=n_workers)

    # Instanciation du model
    # TODO
    model = trajectory2seq(
        hidden_dim=64,
        n_layers=1,
        device=device,
        symb2int=dataset.symb2int,
        int2symb=dataset.int2symb,
        dict_size=dataset.dict_size,
        maxlen=dataset.max_len
    ).to(device)
    # Initialisation des variables
    # TODO

    train_loss_list = []
    val_loss_list = []

    if trainning:

        # Fonction de coût et optimizateur
        # TODO
        criterion = nn.CrossEntropyLoss(ignore_index=dataset.symb2int['<pad>'])
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        for epoch in range(1, n_epochs + 1):
            # Entraînement
            # TODO

            model.train()
            running_loss_train = 0

            for batch_idx, data in enumerate(dataload_train):
                # Formatage des données
                input_seq, target_seq = data
                input_seq = input_seq.to(device).float()
                target_seq = target_seq.to(device).long()

                optimizer.zero_grad() 

                output, hidden, attn = model(input_seq)

                loss = criterion(output.view((-1, model.dict_size)), target_seq.view(-1))
                
                loss.backward() 
                optimizer.step() 

                running_loss_train += loss.item()
            train_loss = running_loss_train / len(dataload_train)
            # Validation
            # TODO

            model.eval()
            running_loss_val = 0
            for batch_idx, data in enumerate(dataload_val):
                with torch.no_grad:
                    input_seq, target_seq = data
                    input_seq = input_seq.to(device).long()
                    target_seq = target_seq.to(device).long()

                    loss = criterion(
                    output.view(-1, dataset.dict_size),
                    target_seq.view(-1)
                )

                    running_loss_val += loss.item()

                    val_loss = running_loss_val / len(dataload_val)
                    
            # Ajouter les loss aux listes
            # TODO
            train_loss_list.append(train_loss)
            val_loss_list.append(val_loss)
            # Enregistrer les poids
            # TODO

            ###################CHATGPT?????????#############################
            if epoch == 1 or val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), "best_model.pt")
            ################################################################

            # Affichage
            if learning_curves:
                plt.figure()
                plt.plot(train_loss_list, label="Train")
                plt.plot(val_loss_list, label="Validation")
                plt.xlabel("Epoch")
                plt.ylabel("Loss")
                plt.legend()
                plt.title("Learning Curves")
                plt.show()

    if test:
        # Évaluation
        # TODO
        
        # Charger les données de tests
        # TODO

        # Affichage de l'attention
        # TODO (si nécessaire)

        # Affichage des résultats de test
        # TODO
        
        # Affichage de la matrice de confusion
        # TODO

        pass