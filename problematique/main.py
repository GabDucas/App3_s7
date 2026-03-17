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

    
    # Séparation de l'ensemble de données (entraînement et validation)
    # TODO
   

    # Instanciation des dataloaders
    # TODO


    # Instanciation du model
    # TODO


    # Initialisation des variables
    # TODO

    if trainning:

        # Fonction de coût et optimizateur
        # TODO

        for epoch in range(1, n_epochs + 1):
            # Entraînement
            # TODO
            
            # Validation
            # TODO

            # Ajouter les loss aux listes
            # TODO

            # Enregistrer les poids
            # TODO


            # Affichage
            if learning_curves:
                # visualization
                # TODO
                pass

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