import torch
from torch.utils.data import Dataset
import cv2
import os
import numpy as np

class DatasetMilho(Dataset):
    def __init__(self, transform=None):
        self.transform = transform
        self.semantic_paths = self.getSemantics()

    def __len__(self):
        return len(self.semantic_paths) * 8  # Correção aqui

    def __getitem__(self, index):
        real_index = index // 8
        transformation = index % 8

        semantic_path = self.semantic_paths[real_index]
        r_path = semantic_path.replace('_', '_r_')
        g_path = semantic_path.replace('_', '_g_')
        b_path = semantic_path.replace('_', '_b_')

        semantic = cv2.imread(semantic_path, cv2.IMREAD_GRAYSCALE)
        r = cv2.imread(r_path, cv2.IMREAD_GRAYSCALE)
        g = cv2.imread(g_path, cv2.IMREAD_GRAYSCALE)
        b = cv2.imread(b_path, cv2.IMREAD_GRAYSCALE)

        image = cv2.merge((b, g, r))

        match transformation:
            case 1:  # Flip Horizontal
                image = cv2.flip(image, 1)
                semantic = cv2.flip(semantic, 1)
            case 2:  # Flip Vertical
                image = cv2.flip(image, 0)
                semantic = cv2.flip(semantic, 0)
            case 3:  # Rotação 90° CW
                image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
                semantic = cv2.rotate(semantic, cv2.ROTATE_90_CLOCKWISE)
            case 4:  # Rotação 180°
                image = cv2.rotate(image, cv2.ROTATE_180)
                semantic = cv2.rotate(semantic, cv2.ROTATE_180)
            case 5:  # Rotação 270° (90° CCW)
                image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)  # Correção aqui
                semantic = cv2.rotate(semantic, cv2.ROTATE_90_COUNTERCLOCKWISE)
            case 6:  # Rotação 90° + Flip Horizontal
                image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
                semantic = cv2.rotate(semantic, cv2.ROTATE_90_CLOCKWISE)
                image = cv2.flip(image, 1)
                semantic = cv2.flip(semantic, 1)
            case 7:  # Rotação 90° + Flip Vertical
                image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
                semantic = cv2.rotate(semantic, cv2.ROTATE_90_CLOCKWISE)
                image = cv2.flip(image, 0)
                semantic = cv2.flip(semantic, 0)
        image = torch.from_numpy(image).float()
        semantic = torch.from_numpy(semantic).float()
        return image, semantic

    @staticmethod
    def getSemantics():
        rootPath = r'dataset\semantics'
        return [os.path.join(rootPath, path) for path in os.listdir(rootPath) if path.endswith('png')]
