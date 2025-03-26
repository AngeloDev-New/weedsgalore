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
        return self.realLen() * 8 

    def getRealItem(self,index):
        return self.__getitem__(index*8)
    
    def realLen(self):
        return len(self.semantic_paths)
    

    def __getitem__(self, index):
        real_index = index // 8
        transformation = index % 8
        semantic_path = self.semantic_paths[real_index]
        semantic = cv2.imread(semantic_path, cv2.IMREAD_GRAYSCALE)

        semantic_path = semantic_path.replace('\\semantics\\','\\images\\')

        r_path = semantic_path.replace('_', '_r_')
        g_path = semantic_path.replace('_', '_g_')
        b_path = semantic_path.replace('_', '_b_')

        
        r = cv2.imread(r_path, cv2.IMREAD_GRAYSCALE)
        g = cv2.imread(g_path, cv2.IMREAD_GRAYSCALE)
        b = cv2.imread(b_path, cv2.IMREAD_GRAYSCALE)

        image = cv2.merge((r, g, b))

        if transformation == 1:  # Flip Horizontal
            image = cv2.flip(image, 1)
            semantic = cv2.flip(semantic, 1)
        elif transformation == 2:  # Flip Vertical
            image = cv2.flip(image, 0)
            semantic = cv2.flip(semantic, 0)
        elif transformation == 3:  # Rotação 90° CW
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            semantic = cv2.rotate(semantic, cv2.ROTATE_90_CLOCKWISE)
        elif transformation == 4:  # Rotação 180°
            image = cv2.rotate(image, cv2.ROTATE_180)
            semantic = cv2.rotate(semantic, cv2.ROTATE_180)
        elif transformation == 5:  # Rotação 270° (90° CCW)
            image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            semantic = cv2.rotate(semantic, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif transformation == 6:  # Rotação 90° + Flip Horizontal
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            semantic = cv2.rotate(semantic, cv2.ROTATE_90_CLOCKWISE)
            image = cv2.flip(image, 1)
            semantic = cv2.flip(semantic, 1)
        elif transformation == 7:  # Rotação 90° + Flip Vertical
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            semantic = cv2.rotate(semantic, cv2.ROTATE_90_CLOCKWISE)
            image = cv2.flip(image, 0)
            semantic = cv2.flip(semantic, 0)
        image = torch.from_numpy(image).float()
        semantic = torch.from_numpy(semantic).float()
        return image, semantic

    @staticmethod
    def getSemantics():
        rootPath = r'dataset\data\semantics'
        return [os.path.join(rootPath, path) for path in os.listdir(rootPath) if path.endswith('png')]
