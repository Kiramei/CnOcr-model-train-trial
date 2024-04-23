from hashlib import md5
import os
import random
import time
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def generate_dataset(otf_file, label_file, img_dir, index_dir):
    """
    Generates a dataset of images and corresponding TSV files for training and testing.

    Args:
        otf_file (str): The path to the font file.
        label_file (str): The path to the label file.
        img_dir (str): The directory to store the images.
        index_dir (str): The directory to store the TSV files.

    Returns:
        None

    Raises:
        FileNotFoundError: If the font file or label file is not found.
        OSError: If there is an error creating the directories.

    Examples:
        generate_dataset('./MainFont.otf', './label_jp.txt', './train', './test', './index')
    """
    # 读取字体文件
    font = ImageFont.truetype(otf_file, size=48)
    
    # 读取标签文件
    with open(label_file, 'r', encoding='utf-8') as f:
        labels = f.read().strip()
    labels = labels.replace('\n', '')
    labels = labels.replace(' ', '')
    
    # 创建训练集和测试集的目录
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(index_dir, exist_ok=True)
    
    
    train_labels = [*labels]
    test_labels = []
    
    jp_chars = train_labels[109:273]
    
    for _ in range(500):
        num_chars = random.randint(2, 40)
        train_labels.append(''.join(random.sample(jp_chars, num_chars)))
    
    for _ in range(2000):
        jp_num_chars = random.randint(6, 40)
        oth_num_chars = random.randint(2, 20)
        jps = ''.join(random.sample(labels, jp_num_chars))
        rand = ''.join(random.sample(jp_chars, oth_num_chars))
        pls= jps+rand
        pls = np.array(list(pls))
        random.shuffle(pls)
        train_labels.append(''.join(pls))
        
    for _ in range(500):
        jp_num_chars = random.randint(6, 40)
        oth_num_chars = random.randint(2, 20)
        jps = ''.join(random.sample(labels, jp_num_chars))
        rand = ''.join(random.sample(jp_chars, oth_num_chars))
        pls= jps+rand
        pls = np.array(list(pls))
        random.shuffle(pls)
        test_labels.append(''.join(pls))
    
    
    # 生成训练集图像和tsv文件
    train_tsv_path = os.path.join(index_dir, 'train.tsv')
    generate_images_and_tsv(font, train_labels, img_dir, train_tsv_path)
    
    # 生成测试集图像和tsv文件
    test_tsv_path = os.path.join(index_dir, 'dev.tsv')
    generate_images_and_tsv(font, test_labels, img_dir, test_tsv_path)

def generate_images_and_tsv(font, labels, output_dir, tsv_file):
    """
    Writes image names and labels to a TSV file.

    Args:
        tsv_file (str): The path to the TSV file.
        labels (List[str]): The list of labels.
        output_dir (str): The directory to store the images.
        font (ImageFont): The font used to generate the images.

    Returns:
        None

    Examples:
        with open('data.tsv', 'w', encoding='utf-8') as f:
            generate_images_and_tsv(font, labels, output_dir, 'data.tsv')
    """
    with open(tsv_file, 'w', encoding='utf-8') as f:
        for i, label in enumerate(labels):
            # hashing
            enc = lambda x: md5(x.encode('utf-8')).hexdigest()
            image_name = f'{enc(str(int(time.time()))+str(i))}.png'
            image_path = os.path.join(output_dir, image_name)
            generate_image(font, label, image_path)
            f.write(f'{image_name}\t{label}\n')

def generate_image(font, label, output_path):
    background_color = (255, 255, 255)  # 白色背景
    text_color = (0, 0, 0)  # 黑色字体
    
    # image_size = get_text_bbox(font, label)
    image_size = (48*len(label), 50)
    image = Image.new('RGB', image_size, background_color)
    draw = ImageDraw.Draw(image)
    position = (0,0)
    
    draw.text(position, label, font=font, fill=text_color)
    image.save(output_path)

def get_text_bbox(font, label):
    image = Image.new('RGB', (1, 1))
    draw = ImageDraw.Draw(image)
    
    bbox = draw.textbbox((0, 0), label, font=font)
    
    image_size = (bbox[2] - bbox[0], bbox[3] - bbox[1])
    return image_size

# 设置参数
otf_file = './MainFont.otf'
label_file = './label_jp.txt'
img_dir = './data'
index_dir = './index'
num_train_images = 1000
num_test_images = 200

# 生成数据集
generate_dataset(otf_file, label_file, img_dir, index_dir)