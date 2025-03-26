import cv2
import torch
import os
from pthPontos.MyNetwork import getModel

class Predict:
    def __init__(self):
        self.prevision = getModel(r'pthPontos\dlv3p_rgb_3.pth')

    def from_path(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Erro: Arquivo não encontrado em {os.path.abspath(path)}")

        image = cv2.imread(path, cv2.IMREAD_COLOR)  # OpenCV lê em BGR
        if image is None:
            raise ValueError(f"Erro ao carregar imagem: {path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Converte para RGB
        image = torch.from_numpy(image).float()  # Converte para tensor
        image = image.permute(2, 0, 1).unsqueeze(0)  # Reordena para [1, 3, H, W]

        semantic = self.prevision(image)  # Passa para o modelo
        return image, semantic
