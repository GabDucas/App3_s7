# GRO722 problématique
# Auteur: Jean-Samuel Lauzon et  Jonathan Vincent
# Hivers 2021
import numpy as np

def edit_distance(a,b):
    # Calcul de la distance d'édition

    m, n = len(a), len(b)
    D = np.zeros((m+1, n+1), dtype=int)
    
    for coordinates, value in np.ndenumerate(D):
        x, y = coordinates
        if x == 0:
            D[coordinates] = y
        elif y == 0:
            D[coordinates] = x
        else:
            min1 = D[x,y-1] + 1
            min2 = D[x-1,y] + 1
            if a[x-1] == b[y-1]: 
                min3 = D[x-1, y-1]
            else:
                min3 = D[x-1, y-1] + 1
            D[coordinates] = np.min([min1, min2, min3])

    return D[m,n]

def confusion_matrix(true, pred, ignore=[]):

    confusion_matrix = np.zeros((len(set(true)), len(set(pred))), dtype=int)

    for i in range(len(true)):
        if  true[i] not in ignore:
            confusion_matrix[true[i], pred[i]] += 1

       
    return confusion_matrix
