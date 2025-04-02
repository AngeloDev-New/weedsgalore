from data.nao_temporal.datasetMilho import DatasetMilho
import matplotlib.pyplot as plt
milho_dataset = DatasetMilho()
image,semantic = milho_dataset.getRealItem(0)
print(image.shape)
print(semantic.shape)
plt.imshow(image.numpy() / 255.0)
# for index in range(milho_dataset.realLen()):
#     image,semantic = milho_dataset.getRealItem(index)
#     print(image.shape)
#     print(semantic.shape)