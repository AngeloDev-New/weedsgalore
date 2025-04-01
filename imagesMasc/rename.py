import os
root = 'imagesMasc'
for image_path_name,new_path_name in \
    map(lambda imageName:\
        (os.path.join(root,imageName),
         os.path.join(root,imageName.replace('predict','image'))),os.listdir(root)):
    command = f'move {image_path_name} {new_path_name}'
    os.system(command)