import cv2
import torch
from pthPontos.MyNetwork import getModel
class Predict:
    def __init__(self):
        self.prevision = getModel(r'pthPontos\dlv3p_rgb_3.pth')
# image, semantic = predict.from_path
    def from_path(self,path):
        image = cv2.imread(path,cv2.IMREAD_COLOR_RGB)
        image = torch.from_numpy(image).float()
        semantic = self.prevision(image)
        return image,semantic