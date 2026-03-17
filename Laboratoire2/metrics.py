import numpy as np
import time
    
def edit_distance(a,b):
    # Calcul de la distance d'édition

    # ---------------------- Laboratoire 2 - Question 1 - Début de la section à compléter ------------------

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
    
    # ---------------------- Laboratoire 2 - Question 1 - Fin de la section à compléter ------------------

if __name__ =="__main__":
    a = list('allo')
    b = list('apollo2')
    c = edit_distance(a,b)

    print('Distance d\'edition entre ',str(a),' et ',str(b), ': ', c)
    