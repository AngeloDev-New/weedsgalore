from weedsgalore.src.nets import deeplabv3plus_resnet50
import torch
import torch.nn as nn
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
class net:
    def __init__(self,path_pth):
        self.Model()
        self.path(path_pth)
        
    def Model(self,num_classes = 3,
            in_channels = 3):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        net = deeplabv3plus_resnet50(num_classes = num_classes)
        net = net.to(device)
        net.backbone.conv1 = nn.Conv2d(in_channels, net.backbone.conv1.out_channels, kernel_size=7, stride=2, padding=3, bias=False, device=device)
        self.model = net
    def path(self,PATH_TO_PTH):
        # PATH_TO_PTH = './modelos/ckpts/dlv3p_rgb_3.pth'
        state_dict = torch.load(PATH_TO_PTH, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.load_state_dict(state_dict)
        # Coloque o modelo em modo de avaliação
        # self.model.eval()

    def inferencia_modelo(self,pathR = './data_weedsgalore/weedsgalore-dataset/2023-06-15/images/2023-06-15_0735_R.png',
                      pathG = './data_weedsgalore/weedsgalore-dataset/2023-06-15/images/2023-06-15_0735_G.png',
                      pathB = './data_weedsgalore/weedsgalore-dataset/2023-06-15/images/2023-06-15_0735_B.png',
                      ):
        # try:
        #     image_R = cv2.imread(pathR, cv2.IMREAD_GRAYSCALE)
        #     image_G = cv2.imread(pathG, cv2.IMREAD_GRAYSCALE)
        #     image_B = cv2.imread(pathB, cv2.IMREAD_GRAYSCALE)
        #     if image_R is None or image_G is None or image_B is None:
        #         raise ValueError("Erro ao carregar uma ou mais imagens.")
        # except:
            
        image_R = cv2.imdecode(pathR, cv2.IMREAD_GRAYSCALE)
        image_G = cv2.imdecode(pathG, cv2.IMREAD_GRAYSCALE)
        image_B = cv2.imdecode(pathB, cv2.IMREAD_GRAYSCALE)
        if image_R is None or image_G is None or image_B is None:
            raise ValueError("Erro ao carregar uma ou mais imagens.")
    # Combinar as imagens R, G, B em um array de 3 canais
        rgb_image = cv2.merge([image_R, image_G, image_B])  # Junta os canais para formar uma imagem RGB

    # Exibir a imagem RGB combinada
        plt.imshow(rgb_image)  
        plt.title('Imagem RGB combinada')
        plt.axis('off')
        plt.show()

    # Transformação para o modelo (sem normalização por enquanto)
        transform = transforms.Compose([
            transforms.ToTensor(),  # Converte para tensor
        ])

    # Pré-processar a imagem
        input_image = transform(rgb_image).unsqueeze(0)  # Adiciona dimensão extra para batch (1, C, H, W)

    # Verificar o dispositivo (GPU ou CPU)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_image = input_image.to(device)

    # Certifique-se de que o modelo está no dispositivo correto
        self.model = self.model.to(device)

    # Colocar o modelo em modo de avaliação
        self.model.eval()

    # Realizar a inferência
        with torch.no_grad():
            output = self.model(input_image)  # Saída do modelo

    # Verificando a saída antes de aplicar argmax
        output_numpy = output.cpu().numpy()
        print("Shape da saída:", output_numpy.shape)  # Deve ser algo como (1, num_classes, H, W)
        print("Valores mínimos e máximos:", output_numpy.min(), output_numpy.max())

    # Se o modelo for de segmentação, obter a máscara final
        output_predictions = output[0]  # Remove a dimensão do batch
        output_predictions = torch.argmax(output_predictions, dim=0)  # Pega a classe mais provável por pixel

    # Converter a saída para imagem
        output_image = output_predictions.cpu().numpy().astype(np.uint8)
    
    # Exibir a imagem de saída (se for uma máscara de segmentação)
        plt.imshow(output_image, cmap="jet")  # Usar cmap para melhor visualização
        plt.colorbar()  # Mostrar escala de cores
        plt.title('Resultado da Inferência')
        plt.axis('off')
        plt.show()
if __name__ == '__main__':
    # inferencia_modelo()
    print('sucesso')