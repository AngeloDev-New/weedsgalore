from weedsgalore.src.nets import deeplabv3plus_resnet50
import torch
import torch.nn as nn
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from io import BytesIO
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
    @staticmethod
    def getLayersFromImage(image_path):
        # Carregar a imagem com OpenCV
        img = cv2.imread(image_path)
    
        # Garantir que a imagem esteja no formato RGB (OpenCV carrega por padrão em BGR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
        # Separar os canais R, G, B
        r_channel = img_rgb[:, :, 0]  # Canal R
        g_channel = img_rgb[:, :, 1]  # Canal G
        b_channel = img_rgb[:, :, 2]  # Canal B

        # Converter os canais para PNG em memória usando imencode
        _, r_encoded = cv2.imencode('.png', r_channel)
        _, g_encoded = cv2.imencode('.png', g_channel)
        _, b_encoded = cv2.imencode('.png', b_channel)

        #   Criar objetos BytesIO a partir dos dados binários
        r_io = BytesIO(r_encoded.tobytes())
        g_io = BytesIO(g_encoded.tobytes())
        b_io = BytesIO(b_encoded.tobytes())

        # Resetar o cursor dos arquivos BytesIO para leitura posterior
        r_io.seek(0)
        g_io.seek(0)
        b_io.seek(0)
        return r_io, g_io, b_io
    def inferencia_modelo_by_RGB(self,RGBImage):
        r_io,g_io,b_io = self.getLayersFromImage(RGBImage)
        self.inferencia_modelo(r_io,g_io,b_io)
    def inferencia_modelo(self,pathR = './data_weedsgalore/weedsgalore-dataset/2023-06-15/images/2023-06-15_0735_R.png',
                      pathG = './data_weedsgalore/weedsgalore-dataset/2023-06-15/images/2023-06-15_0735_G.png',
                      pathB = './data_weedsgalore/weedsgalore-dataset/2023-06-15/images/2023-06-15_0735_B.png',
                      ):
        r_data = np.asarray(bytearray(pathR.read()), dtype=np.uint8)
        g_data = np.asarray(bytearray(pathG.read()), dtype=np.uint8)
        b_data = np.asarray(bytearray(pathB.read()), dtype=np.uint8)
        image_R = cv2.imdecode(r_data, cv2.IMREAD_GRAYSCALE)
        image_G = cv2.imdecode(g_data, cv2.IMREAD_GRAYSCALE)
        image_B = cv2.imdecode(b_data, cv2.IMREAD_GRAYSCALE)
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