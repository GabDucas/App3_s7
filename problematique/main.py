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
    training = True           # Entrainement?
    test = True                # Test?
    learning_curves = True     # Affichage des courbes d'entrainement?
    gen_test_images = True     # Génération images test?
    seed = 1                # Pour répétabilité
    n_workers = 0           # Nombre de threads pour chargement des données (mettre à 0 sur Windows)
    batch_size = 40
    learning_rate = 0.0001

    # TODO
    n_epochs = 40
    n_samp = 5000

    # ---------------- Fin Paramètres et hyperparamètres ----------------#

    # Initialisation des variables
    if seed is not None:
        torch.manual_seed(seed) 
        np.random.seed(seed)

    # Choix du device
    device = torch.device("cuda" if torch.cuda.is_available() and not force_cpu else "cpu")

    # Instanciation de l'ensemble de données
    # TODO
    dataset = HandwrittenWords('problematique/data_trainval.p')
    
    # Séparation de l'ensemble de données (entraînement et validation)
    # TODO
    dataset_train, dataset_val, dataset_test = torch.utils.data.random_split(dataset, [3500, 1000, 500])

    # Instanciation des dataloaders
    # TODO
    dataload_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=n_workers)
    dataload_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=n_workers)

    # Instanciation du model
    # TODO
    model = trajectory2seq(
        hidden_dim=64,
        n_layers=1,
        device=device,
        symb2int=dataset.symb2int,
        int2symb=dataset.int2symb,
        dict_size=len(dataset.symb2int) ,
        max_len=dataset.max_len_traj
    ).to(device)
    # Initialisation des variables
    # TODO

    train_loss_list = []
    val_loss_list = []
    best_val_loss = float('inf')
    running_loss_val = 0
    
    if training:
        print('in train')
        # Fonction de coût et optimizateur
        # TODO
        criterion = nn.CrossEntropyLoss(ignore_index=dataset.symb2int['<pad>'])
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

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

                output, hidden, attn = model(input_seq,  target_seq)
                #output = (batch_size, seq_length, vocab_size)

                loss = criterion(
                    output.reshape(-1, model.dict_size),
                    target_seq.reshape(-1)
                )
                
                loss.backward() 
                optimizer.step() 

                running_loss_train += loss.item()


                print(
                        'Train - Epoch: {}/{} [{}/{} ({:.0f}%)] Average Loss: {:.6f}'.format(
                            epoch,
                            n_epochs,
                            batch_idx * batch_size,
                            len(dataload_train.dataset),
                            100. * batch_idx * batch_size / len(dataload_train.dataset),
                            running_loss_train / (batch_idx + 1)
                        ),
                        end='\r'
                    )
                
            train_loss = running_loss_train / len(dataload_train)
            # Validation
            # TODO

            model.eval()
            running_loss_val = 0

            with torch.no_grad():
                for batch_idx, data in enumerate(dataload_val):
                    input_seq, target_seq = data
                    input_seq = input_seq.to(device).float()
                    target_seq = target_seq.to(device).long()

                    # Forward
                    output, hidden, attn = model(input_seq, target_seq) 

                    # Loss
                    loss = criterion(output.view(-1, model.dict_size),
                                    target_seq.view(-1))
                    running_loss_val += loss.item()

            val_loss = running_loss_val / len(dataload_val)
            print(f"\nValidation - Epoch {epoch}: Loss={val_loss:.6f}")   

            # Ajouter les loss aux listes
            # TODO
            train_loss_list.append(train_loss)
            val_loss_list.append(val_loss)
            # Enregistrer les poids
            # TODO

            if epoch == 1 or val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), "best_model.pt")

            # Affichage
            if learning_curves:
                plt.clf()
                plt.plot(train_loss_list, label="Train")
                plt.plot(val_loss_list, label="Validation")
                plt.legend()
                plt.pause(0.01)
    plt.show()

    if test:
        total_loss = 0

        criterion = nn.CrossEntropyLoss(ignore_index=dataset.symb2int['<pad>'])

        # Évaluation
        # TODO
        model.load_state_dict(torch.load("best_model.pt", map_location=device))
        model.eval() 
        # Charger les données de tests
        # TODO
        dataload_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=n_workers)


        with torch.no_grad():  # Pas de calcul de gradients
            for batch_idx, data in enumerate(dataload_test):
                # Formatage des données
                input_seq, target_seq = data
                input_seq = input_seq.to(device).float()
                target_seq = target_seq.to(device).long()

                # Forward pass
                output, hidden, attn = model(input_seq, target_seq) 

                # Loss
                loss = criterion(output.view(-1, model.dict_size),
                                 target_seq.view(-1))
                total_loss += loss.item()
                pred_seq = torch.argmax(output, dim=2)

        # Affichage de l'attention
        # TODO (si nécessaire)

        
        # Affichage des résultats de test
         
        for k in range(min(3, input_seq.size(0))):
            points = input_seq[k].cpu().numpy()
            true_tokens = target_seq[k].cpu().numpy()
            pred_tokens = pred_seq[k].cpu().numpy()
            true_tokens = [t for t in true_tokens if t != dataset.symb2int['<pad>']]
            pred_tokens = [t for t in pred_tokens if t != dataset.symb2int['<pad>']]
            true_text = "".join([dataset.int2symb[t] for t in true_tokens])
            pred_text = "".join([dataset.int2symb[t] for t in pred_tokens])
            plt.figure()
            plt.plot(points[:,0], points[:,1])
            plt.title(f"Vrai: {true_text} | Prédit: {pred_text}")
            plt.show()
 
        # TODO
        
        # Affichage de la matrice de confusion
        # TODO

        pass