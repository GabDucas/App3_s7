# GRO722 problématique
# Auteur: Jean-Samuel Lauzon et  Jonathan Vincent
# Hivers 2021

import seaborn as sns
import torch
from torch import nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from models import *
from dataset import *
from metrics import *

def plot_trajectory_attention(points, attention_matrix, predicted_chars, k):
    """
    Affiche l'attention
    """
    N = len(predicted_chars)
    fig, axes = plt.subplots(N, 1, figsize=(6, 1.2 * N))

    if N == 1:
        axes = [axes]
        
    for i in range(N):
        ax = axes[i]
        ax.plot(points[:, 0], points[:, 1], color='#e0e0e0', linewidth=2.5, zorder=1)
        weights = attention_matrix[i]

        if np.max(weights) > 0:
            weights = weights / np.max(weights)

        rgba_colors = np.zeros((len(points), 4))
        rgba_colors[:, 3] = np.clip(weights, 0, 1) # Canal Alpha

        ax.scatter(points[:, 0], points[:, 1], color=rgba_colors, s=20, zorder=2)

        ax.set_ylabel(predicted_chars[i], rotation=0, labelpad=15, fontsize=14, va='center')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal', 'datalim') 
        
    plt.tight_layout()
    plt.savefig(f"attention_output_{k}.png")


if __name__ == '__main__':

    # ---------------- Paramètres et hyperparamètres ----------------#
    force_cpu = False           # Forcer a utiliser le cpu?
    training = True           # Entrainement?
    test = True                # Test?
    learning_curves = True     # Affichage des courbes d'entrainement?
    gen_test_images = True     # Génération images test?
    seed = 1                # Pour répétabilité
    n_workers = 0           # Nombre de threads pour chargement des données (mettre à 0 sur Windows)
    batch_size = 64
    bidirectional = False
    attention = True
    test_lstm = False

    
    n_samp = 5000
    n_epochs = 70
    learning_rate = 0.008
    n_layers = 1

    if bidirectional and not test_lstm:
        hidden_dim = 8//n_layers
    elif not bidirectional and not test_lstm:
        hidden_dim = 28//n_layers
    elif bidirectional and test_lstm:
        hidden_dim = 4//n_layers
    elif not bidirectional and test_lstm:
        hidden_dim = 25//n_layers

    # ---------------- Fin Paramètres et hyperparamètres ----------------#

    # Initialisation des variables
    if seed is not None:
        torch.manual_seed(seed) 
        np.random.seed(seed)

    # Choix du device
    device = torch.device("cuda" if torch.cuda.is_available() and not force_cpu else "cpu")

    # Instanciation de l'ensemble de données
    
    dataset = HandwrittenWords('problematique/data_trainval.p')
    
    # Séparation de l'ensemble de données (entraînement et validation)
    
    dataset_train, dataset_val = torch.utils.data.random_split(dataset, [4250, 750])
    dataset_test = HandwrittenWords('problematique/data_test.p')

    # Instanciation des dataloaders
    
    dataload_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=n_workers)
    dataload_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=n_workers)

    # Instanciation du model
    
    model = trajectory2seq(
        hidden_dim=hidden_dim,
        n_layers=1,
        device=device,
        symb2int=dataset.symb2int,
        int2symb=dataset.int2symb,
        dict_size=len(dataset.symb2int) ,
        max_len=dataset.max_len_traj,
        bidirectional=bidirectional,
        attention=attention,
        lstm=test_lstm
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters:     {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    # Initialisation des variables
    
    
    train_loss_list = []
    val_loss_list = []
    train_dist_list = []
    val_dist_list = []
    dist_epochs = []
    best_val_loss = float('inf')
    running_loss_val = 0
    
    if training:
        print('in train')
        # Fonction de coût et optimizateur
        
        criterion = nn.CrossEntropyLoss(ignore_index=dataset.symb2int['<pad>'])
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

        if learning_curves:
            plt.ion() # Turn on interactive mode for live updates
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        for epoch in range(1, n_epochs + 1):
            # Entraînement
            

            model.train()
            running_loss_train = 0
            running_dist_train = 0

            for batch_idx, data in enumerate(dataload_train):
                # Formatage des données
                input_seq, target_seq = data
                input_seq = input_seq.to(device).float()
                target_seq = target_seq.to(device).long()

                optimizer.zero_grad() 

                teacher_forcing_ratio = max(0.0, 1.0 - epoch * 0.01)

                output, hidden, attn = model(input_seq, target_seq, teacher_forcing_ratio)
                #output = (batch_size, seq_length, vocab_size)

                loss = criterion(
                    output.reshape(-1, model.dict_size),
                    target_seq.reshape(-1)
                )
                
                loss.backward() 
                optimizer.step() 

                running_loss_train += loss.item()

                # Calcul de la distance d'édition pour un échantillon du batch
                if epoch % 5 == 0 or epoch == n_epochs or epoch == 1:
                    pred_seq = torch.argmax(output, dim=2)
                    batch_dist = 0
                    for i in range(input_seq.size(0)):
                        # Convert indices to text, ignoring special tokens
                        t_text = "".join([dataset.int2symb[t.item()] for t in target_seq[i] 
                                        if t.item() not in [dataset.symb2int['<pad>'], dataset.symb2int['<eos>'], dataset.symb2int['<sos>']]])
                        p_text = "".join([dataset.int2symb[p.item()] for p in pred_seq[i] 
                                        if p.item() not in [dataset.symb2int['<pad>'], dataset.symb2int['<eos>'], dataset.symb2int['<sos>']]])
                        batch_dist += edit_distance(t_text, p_text)
                    
                    running_dist_train += (batch_dist / input_seq.size(0))


                print(
                        'Train - Epoch: {}/{} [{}/{} ({:.0f}%)] Average Loss: {:.6f} - Average Edit Distance: {:.6f}'.format(
                            epoch,
                            n_epochs,
                            batch_idx * batch_size,
                            len(dataload_train.dataset),
                            100. * batch_idx * batch_size / len(dataload_train.dataset),
                            running_loss_train / (batch_idx + 1),
                            running_dist_train / (batch_idx + 1)
                        ),
                        end='\r'
                    )
            
            # Calcul des métriques d'entraînement pour l'époque
            train_loss = running_loss_train / len(dataload_train)
            if epoch % 5 == 0 or epoch == n_epochs or epoch == 1:
                train_dist = running_dist_train / len(dataload_train)
                dist_epochs.append(epoch)


            # Validation
            model.eval()
            running_loss_val = 0
            running_dist_val = 0

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

                    # Calcul de la distance d'édition pour un échantillon du batch
                    if epoch % 5 == 0 or epoch == n_epochs or epoch == 1:
                        pred_seq = torch.argmax(output, dim=2)
                        batch_dist = 0
                        for i in range(input_seq.size(0)):
                            # Convert indices to text, ignoring special tokens
                            t_text = "".join([dataset.int2symb[t.item()] for t in target_seq[i] 
                                            if t.item() not in [dataset.symb2int['<pad>'], dataset.symb2int['<eos>'], dataset.symb2int['<sos>']]])
                            p_text = "".join([dataset.int2symb[p.item()] for p in pred_seq[i] 
                                            if p.item() not in [dataset.symb2int['<pad>'], dataset.symb2int['<eos>'], dataset.symb2int['<sos>']]])
                            batch_dist += edit_distance(t_text, p_text)
                        
                        running_dist_val += (batch_dist / input_seq.size(0))

            val_loss = running_loss_val / len(dataload_val)
            if epoch % 5 == 0 or epoch == n_epochs or epoch == 1:
                val_dist = running_dist_val / len(dataload_val)
            else:
                val_dist = 0.0
            print(f"\nValidation - Epoch {epoch}: Loss={val_loss:.6f}, Edit Distance={val_dist:.6f}")   

            # Ajouter les loss aux listes
        
            train_loss_list.append(train_loss)
            val_loss_list.append(val_loss)

            if epoch % 5 == 0 or epoch == n_epochs or epoch == 1:
                train_dist_list.append(train_dist)
                val_dist_list.append(val_dist)

            # Enregistrer les poids
        

            if epoch == 1 or val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), "best_model.pt")

            # Affichage
            if learning_curves:
                ax1.clear()
                ax2.clear()

                # Plot Loss
                ax1.plot(train_loss_list, label="Train Loss", color='blue')
                ax1.plot(val_loss_list, label="Val Loss", color='red')
                ax1.set_title("Evolution of Loss")
                ax1.set_xlabel("Epoch")
                ax1.legend()

                # Plot Edit Distance
                if epoch % 5 == 0 or epoch == n_epochs or epoch == 1: # Only plot if we have at least one data point
                    ax2.plot(dist_epochs, train_dist_list, label="Train Dist", color='blue')
                    ax2.plot(dist_epochs, val_dist_list, label="Val Dist", color='red')
                    ax2.set_title("Edit Distance (Sampled)")
                    ax2.set_xlabel("Epoch")

                plt.tight_layout()
                plt.pause(0.01)
    plt.savefig("learning_curves.png") 

    if test:
        total_loss = 0

        criterion = nn.CrossEntropyLoss(ignore_index=dataset.symb2int['<pad>'])

        # Évaluation
        model.load_state_dict(torch.load("best_model.pt", map_location=device))
        model.eval() 
        # Charger les données de tests
        dataload_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=n_workers)

        cmpt = 0
        edit_distances = []
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
                cmpt += 1
                # Calcul de la distance d'édition pour le batch
                for true_array, pred_array in zip(target_seq, pred_seq):
                    t_chars = [dataset.int2symb[t.item()] for t in true_array if t.item() not in [dataset.symb2int['<pad>'], dataset.symb2int['<eos>']]]
                    p_chars = [dataset.int2symb[p.item()] for p in pred_array if p.item() not in [dataset.symb2int['<pad>'], dataset.symb2int['<eos>']]]

                    t_text = "".join(t_chars)
                    p_text = "".join(p_chars)
                    
                    edit_distances.append(edit_distance(t_text, p_text))
        avg_loss = total_loss / cmpt
        edit_distance = np.mean(edit_distances)
        print(f"\nTest - Average Edit Distance: {edit_distance:.4f} - Loss: {avg_loss:.6f}")
        
        # RESULTATS TEST + ATTENTION
         
        for i in range(min(3, input_seq.size(0))):
            # Sélection aléatoire d'un échantillon du batch
            k = np.random.randint(0, input_seq.size(0))
            points = input_seq[k].cpu().numpy()
            true_tokens = target_seq[k].cpu().numpy()
            pred_tokens = pred_seq[k].cpu().numpy()
            true_tokens = [t for t in true_tokens if t != dataset_test.symb2int['<pad>'] and t != dataset_test.symb2int['<eos>']]
            pred_tokens = [t for t in pred_tokens if t != dataset_test.symb2int['<pad>'] and t != dataset_test.symb2int['<eos>']]
            true_text = "".join([dataset_test.int2symb[t] for t in true_tokens])
            pred_text = "".join([dataset_test.int2symb[t] for t in pred_tokens])
            # Affichage de la trajectoire
            plt.figure()
            plt.plot(points[:,0], points[:,1])
            plt.title(f"Vrai: {true_text} | Prédit: {pred_text}")
            plt.savefig(f"test_results_{i}.png")

            # Préparation de l'échantillon d'attention
            pred_tokens_k = [t for t in pred_seq[k].cpu().numpy() if t != dataset_test.symb2int['<pad>']]
            pred_chars_list = [dataset_test.int2symb[t] for t in pred_tokens_k]
            
            attention_matrix = attn[k].cpu().numpy() 

            if attention_matrix.shape[0] == len(points):
                attention_matrix = attention_matrix.T
            
            attention_matrix = attention_matrix[:len(pred_chars_list), :]
            
            # Affichage
            plot_trajectory_attention(points, attention_matrix, pred_chars_list, i)
 
        # MATRICE DE CONFUSION
        
        true_flat = target_seq.view(-1).cpu().tolist()
        pred_flat = pred_seq.view(-1).cpu().tolist()
        ignore_list = [dataset_test.symb2int['<pad>'], dataset_test.symb2int['<eos>'], dataset_test.symb2int['<sos>']]
        
        num_classes = len(dataset_test.symb2int)
        confusion = confusion_matrix(true_flat, pred_flat, num_classes, ignore=ignore_list)

        valid_idx = [i for i in range(num_classes) if i not in ignore_list]
        
        clean_matrix = confusion[valid_idx][:, valid_idx]
        clean_labels = [dataset_test.int2symb[i] for i in valid_idx]

        # Plot
        plt.figure(figsize=(10, 8))
        sns.heatmap(clean_matrix, annot=True, fmt="d", cmap="Blues", 
                    xticklabels=clean_labels, yticklabels=clean_labels)
        plt.ylabel('Actual Character')
        plt.xlabel('Predicted Character')
        plt.title("Character-Level Confusion Matrix")
        plt.savefig("confusion_matrix.png")
        pass