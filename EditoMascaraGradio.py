import gradio as gr
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from dataset.datasetMilho import DatasetMilho

def salvar(image, threshold_r=128, threshold_g=128, threshold_b=128, threshold_final=128, save=False):
    # Convertendo a imagem para um formato manipulável
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()  # Convertendo para numpy, movendo para a CPU se necessário
        
        if image.dtype == np.float32 or image.dtype == np.float64:
            image = (image * 255).astype(np.uint8)  # Normalizando para uint8 se necessário
        
        if image.shape[0] == 3:  # Verificando se está no formato (C, H, W)
            image = np.transpose(image, (1, 2, 0))  # Convertendo para (H, W, C)
        
        image = image.astype(np.uint8)  # Garantindo que seja do tipo uint8
        image = Image.fromarray(image)  # Convertendo para PIL.Image

    # Verificando se a imagem já está negativa, caso contrário, ela é tratada como tal
    image_np = np.array(image)  # Convertendo a imagem PIL para um array numpy
    if np.all(image_np <= 255) and np.all(image_np >= 0):  # Verifica se a imagem está em valores válidos
        imagem_negativa_np = 255 - image_np  # Aplicando o negativo apenas se necessário
    else:
        imagem_negativa_np = image_np  # Se já for negativa, usamos ela diretamente

    # Aplicando os limiares individuais para cada canal (R, G, B)
    image_thresholded_r = np.where(imagem_negativa_np[..., 0] < threshold_r, 0, 255).astype(np.uint8)
    image_thresholded_g = np.where(imagem_negativa_np[..., 1] < threshold_g, 0, 255).astype(np.uint8)
    image_thresholded_b = np.where(imagem_negativa_np[..., 2] < threshold_b, 0, 255).astype(np.uint8)

    # Combinando os canais limiarizados
    image_thresholded = np.stack([image_thresholded_r, image_thresholded_g, image_thresholded_b], axis=-1)

    # Convertendo a imagem limiarizada para escala de cinza
    image_gray = np.dot(image_thresholded[..., :3], [0.2989, 0.5870, 0.1140])  # Fórmula de luminância para RGB -> Grayscale
    image_gray = image_gray.astype(np.uint8)

    # Aplicando o limiar final para a imagem em escala de cinza
    image_thresholded_final = np.where(image_gray < threshold_final, 0, 255).astype(np.uint8)

    # Exibindo as imagens com plt
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(imagem_negativa_np)
    axs[0].set_title("Imagem Negativa")
    axs[0].axis('off')

    axs[1].imshow(image_thresholded_final, cmap='gray')
    axs[1].set_title(f"Imagem Limiarizada (Threshold Final {threshold_final})")
    axs[1].axis('off')

    # Retorna a imagem limiarizada final como uma imagem PIL para exibição no Gradio
    return Image.fromarray(image_thresholded_final)


def interface_gradio():
    # Carregar o dataset
    milho_dataset = DatasetMilho()
    image, _ = milho_dataset.getRealItem(1300)  # Pegando a primeira imagem

    # Verificando se a imagem é um tensor e convertendo para numpy
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
        if image.shape[0] == 3:  # Verificando formato (C, H, W)
            image = np.transpose(image, (1, 2, 0))  # Convertendo para (H, W, C)
        image = image.astype(np.uint8)  # Garantindo que seja do tipo uint8
        image = Image.fromarray(image)  # Convertendo para PIL.Image
    
    # Definir os sliders de ajuste para os limiares
    sliders = [
        gr.Slider(minimum=0, maximum=255, step=1, label="Threshold R", value=128),
        gr.Slider(minimum=0, maximum=255, step=1, label="Threshold G", value=128),
        gr.Slider(minimum=0, maximum=255, step=1, label="Threshold B", value=128),
        gr.Slider(minimum=0, maximum=255, step=1, label="Threshold Final", value=128)
    ]

    # Interface com os sliders e a função de salvar
    gr.Interface(fn=salvar, inputs=[gr.Image(type="pil", value=image)] + sliders, outputs=gr.Image(type="pil"), live=True).launch()

# Rodar a interface Gradio
if __name__ == '__main__':
    interface_gradio()
