# -*- coding: utf-8 -*-
"""
Created on Sun Oct  9 12:14:14 2022

@author: Marc
"""

import sys

sys.path.insert(1, "Detection/src/utils/")

import time

inicio = time.time()
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split, TensorDataset
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(1, "Detection/src/models/")
from .Features_Extraction import Features_Extraction
from ..utils.Main_Algoritmo_Viterbi import Algoritmo_Viterbi
import copy
import random
import time


def Deteccion_sismica(sac_file, models, results_path):
    path_probPrior_train = models["path_probPrior_train"]
    path_modelo = models["path_modelo"]
    path_3estados = models["path_3estados"]
    path_transitions_file = models["path_transitions_file"]
    print("Running deteccion sismica for file: ", sac_file)
    # Rutas de carga de archivos
    # path_probPrior_train = "../../models/Probs_Prior_NorthChile_Train.npy"
    # path_modelo = "../../models/model_MLP_HMM_NorthChile.pt"

    # Rutas para guardar resultados
    name_file_results = sac_file.split("/")[-2]
    # file_viterbi_test = "../../models/results/Detection_" + name_file_results
    file_viterbi_test = results_path + "/Detection_" + name_file_results

    # Se aplica extraccion de caracteristicas al archivo sac
    print("Extrayendo caracteristicas...")
    features = Features_Extraction(sac_file)

    # Se carga las probabilidades a priori
    probPriorTrain = np.load(path_probPrior_train, allow_pickle=True)

    # Cargar modelo
    class MLP(nn.Module):
        def __init__(self, input_dim, output_dim):
            super().__init__()
            self.input_fc = nn.Linear(input_dim, 24)
            self.hidden_fc1 = nn.Linear(24, 20)
            self.hidden_fc2 = nn.Linear(20, 16)
            self.hidden_fc3 = nn.Linear(16, 12)
            self.output_fc = nn.Linear(12, output_dim)

        def forward(self, x):
            # x = [batch size, height, width]
            batch_size = x.shape[0]
            x = x.view(batch_size, -1)
            h_0 = F.relu(self.input_fc(x))
            h_1 = F.relu(self.hidden_fc1(h_0))
            h_2 = F.relu(self.hidden_fc2(h_1))
            h_3 = F.relu(self.hidden_fc3(h_2))
            y_pred = self.output_fc(h_3)
            return y_pred, h_3

    INPUT_DIM = 918  # Features con contexto
    OUTPUT_DIM = 12  # Estados

    model = MLP(INPUT_DIM, OUTPUT_DIM)
    device = torch.device("cpu")
    model.load_state_dict(torch.load(path_modelo))

    def get_predictions(model, iterator, device):

        model.eval()

        images = []
        labels = []
        probs = []

        with torch.no_grad():

            for x, y in iterator:
                x = x.to(device)
                y_pred, _ = model(x.float())
                y_prob = F.softmax(y_pred, dim=-1)
                images.append(x.cpu())
                labels.append(y.cpu())
                probs.append(y_prob.cpu())

        images = torch.cat(images, dim=0)
        labels = torch.cat(labels, dim=0)
        probs = torch.cat(probs, dim=0)

        return images, labels, probs

    def DNN2ProbObs(feat_entrada):
        salida_DNN = []
        for traza in feat_entrada:

            set_conjunto = TensorDataset(
                torch.from_numpy(traza), -1 * torch.ones(len(traza))
            )  # El target lo seteo en -1, da lo mismo
            conjunto_iterator = DataLoader(set_conjunto)
            images, _, probs = get_predictions(model, conjunto_iterator, device)
            salida_DNN.append(probs)

        calculo_Prob = []
        for ProbTraza in salida_DNN:
            calculo_Prob.append(np.log(ProbTraza) - np.log(probPriorTrain))

        Probs_Observations = []
        for traza in calculo_Prob:
            ruido = traza[:, 0:3]
            evento = traza[:, 3:]
            Probs_Observations.append([np.array(ruido), np.array(evento)])
        return Probs_Observations

    # Testear el modelo para obtener las probabilidades de observacion
    print("Testeando modelo...")
    Probs_Observations_test = DNN2ProbObs(features)

    # Algoritmo de viterbi
    print("Aplicando algoritmo de Viterbi...")
    Algoritmo_Viterbi(
        file_viterbi_test, Probs_Observations_test, path_3estados, path_transitions_file
    )
