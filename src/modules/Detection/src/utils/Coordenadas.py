import numpy as np

# Me falta comentar la función


def Coordenadas(D, max_arg):
    N_modelos = len(D)
    Indice = np.argmax(max_arg)

    for i in range(N_modelos):
        N_estados = len(D[i])
        for j in range(N_estados):
            if Indice == j:
                Posicion = [i, j]
        Indice = Indice - N_estados

    return Posicion
