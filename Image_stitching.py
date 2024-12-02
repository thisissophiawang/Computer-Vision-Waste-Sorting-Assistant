import os
import random
from PIL import Image


def create_collage(folder_path, output_path, size=(200, 200)):
    # 获取文件夹中的所有图像文件 get all image files in the folder
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

    # 随机选择四张图像 random select 4 images
    selected_images = random.sample(image_files, 4)

    # 创建一个新的空白图像，大小为 (2 * width, 2 * height) /create a new blank image with size (2 * width, 2 * height)
    collage_width = 2 * size[0]
    collage_height = 2 * size[1]
    collage = Image.new('RGB', (collage_width, collage_height))

    # 加载并缩放图像 load and resize images
    for index, image_file in enumerate(selected_images):
        img_path = os.path.join(folder_path, image_file)
        img = Image.open(img_path)
        img = img.resize(size, Image.ANTIALIAS)  # 缩放图像 resize image

        if index == 0:
            collage.paste(img, (0, 0))  # 左上角 left top
        elif index == 1:
            collage.paste(img, (size[0], 0))  # 右上角 right top
        elif index == 2:
            collage.paste(img, (0, size[1]))  # 左下角 left bottom
        elif index == 3:
            collage.paste(img, (size[0], size[1]))  # 右下角 right bottom

    # 保存拼接后的图像 save the collage image
    collage.save(output_path)
    print(f"拼接后的图像已保存到: {output_path}")


# example
folder_path = r'E:\项目\垃圾分类\3943f-main\yolo_rubbish\images\train'  # 
output_path = 'output_collage5.jpg'  
create_collage(folder_path, output_path)