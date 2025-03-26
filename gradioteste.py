import gradio as gr
import os
import numpy as np
from scripts.dataManipulations import Predict
import cv2

# Diretório das imagens
image_dir = r'imagens'
paths_images = [os.path.join(image_dir, paths) for paths in os.listdir(image_dir) if paths.endswith('png')]

# Função para processar a imagem e a máscara
def process_image(image, mask):
    result_image = np.copy(image)  # Copiar a imagem para sobrepor a máscara
    result_image[mask == 1] = [0, 255, 0]  # Cor verde para "milho"
    result_image[mask == 2] = [0, 0, 255]  # Cor azul para "daninha"
    return result_image

# Função para ir para a próxima imagem
def next_image(current_index):
    current_index = (current_index + 1) % len(paths_images)
    image, semantic = predict.from_path(paths_images[current_index])
    return image, semantic, current_index

# Inicializar o objeto de previsão
predict = Predict()

# Função para carregar a imagem e a máscara
def load_image_and_mask(index):
    image, semantic = predict.from_path(paths_images[index])
    return image, semantic, index

# Mapear valores da máscara para uma paleta de cores
def create_colored_mask(mask):
    colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    colored_mask[mask == 1] = [0, 255, 0]  # Verde para "milho"
    colored_mask[mask == 2] = [0, 0, 255]  # Azul para "daninha"
    return colored_mask

# Criar a interface Gradio
def update_mask(image, mask, edit_mask):
    # Atualiza a máscara com as alterações feitas
    return process_image(image, edit_mask)

# Função para salvar as mudanças
def save_mask(mask, name):
    mask = np.array(mask)  # Certifique-se de que a máscara é uma matriz NumPy
    mask_path = f'./semantics/{name}.png'
    cv2.imwrite(mask_path, mask)  # Salva a máscara no caminho especificado

# Criar interface com Blocks
with gr.Blocks() as demo:
    # Definir os componentes
    image_input = gr.Image(sources=["upload"], type="numpy", label="Imagem Original", scale=600)
    mask_input = gr.Paint(width=600, height=600, label="Máscara Semântica (Fundo, Milho, Daninha)")
    next_button = gr.Button("Próxima Imagem")
    save_button = gr.Button("Salvar Máscara")
    
    # Caixa de texto para o nome do arquivo da máscara
    mask_name_input = gr.Textbox(label="Nome da Máscara", placeholder="Digite o nome da máscara")

    # Estado para armazenar o índice
    current_index = gr.State(0)

    # Inicializa a interface com a imagem e a máscara
    def init_interface():
        image, mask, current_index_val = load_image_and_mask(current_index)
        colored_mask = create_colored_mask(mask)  # Aplica a paleta de cores
        return image, colored_mask, current_index_val
    
    # Inicializa os dados quando o Gradio é carregado
    demo.load(init_interface, outputs=[image_input, mask_input, current_index])
    
    # Configurar os cliques dos botões
    next_button.click(fn=next_image, inputs=[current_index], outputs=[image_input, mask_input, current_index])
    save_button.click(fn=save_mask, inputs=[mask_input, image_input], outputs=[])

# Rodar a interface
demo.launch()
