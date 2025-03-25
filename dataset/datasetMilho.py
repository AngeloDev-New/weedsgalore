import torch
from torch.utils.data import Dataset
import cv2
import os
import numpy as np

class DatasetMilho(Dataset):
    def __init__(self, root_dir='dataset/data', transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.imagens, self.mascaras = self.getPaths(root_dir)  # Obtém os caminhos

    def __len__(self):
        return len(self.mascaras)
    
    def __getitem__(self, index):
        # Obtém os caminhos completos
        imagenPath = self.imagens[index]
        mascaraPath = self.mascaras[index]

        # Carrega a imagem
        imagen = cv2.imread(imagenPath)
        imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)

        # Carrega a máscara
        mascara = cv2.imread(mascaraPath, cv2.IMREAD_GRAYSCALE)
        mascara = cv2.resize(mascara, (600, 600))

        # Converte para tensor (normalizando para [0,1])
        imagen = torch.tensor(imagen, dtype=torch.float32).permute(2, 0, 1) / 255.0
        mascara = torch.tensor(mascara, dtype=torch.float32).unsqueeze(0) / 255.0  # Adiciona canal extra

        # Aplica transformações se houver
        if self.transform:
            imagen = self.transform(imagen)
            mascara = self.transform(mascara)
        imagen = imagen.unsqueeze(0)
        return imagen, mascara
    
    @staticmethod
    def getPaths(root_dir):
        ImagesPaths = [os.path.join(root_dir, path) for path in os.listdir(root_dir) if path.endswith('.png') and 'image' in path]
        MasksPaths = [path.replace('image', 'predict') for path in ImagesPaths]
        return ImagesPaths, MasksPaths
