import json
import os
from shutil import copyfile

def coco_to_yolo(coco_json, image_dir, label_dir):
    with open(coco_json, 'r') as f:
        coco_data = json.load(f)

    category_dict = {category['id']: category['name'] for category in coco_data['categories']}

    for annotation in coco_data['annotations']:
        image_id = annotation['image_id']
        category_id = annotation['category_id']
        bbox = annotation['bbox']  # [x, y, width, height]
        image_name = [img['file_name'] for img in coco_data['images'] if img['id'] == image_id][0]
        image_path = os.path.join(image_dir, image_name)

        # 将标注写入相应的 YOLO 格式文件
        label_file = os.path.join(label_dir, image_name.split('.')[0] + '.txt')
        with open(label_file, 'a') as label:
            x_center = (bbox[0] + bbox[2] / 2) / 640  # Assuming 640x640 image size
            y_center = (bbox[1] + bbox[3] / 2) / 640
            width = bbox[2] / 640
            height = bbox[3] / 640
            label.write(f"{category_id} {x_center} {y_center} {width} {height}\n")

        # 复制图像文件
        if not os.path.exists(os.path.join(label_dir, image_name)):
            copyfile(os.path.join(image_dir, image_name), os.path.join(label_dir, image_name))

# 设置路径
coco_json = 'path/to/coco/annotations/instances_train2017.json'
image_dir = 'path/to/coco/train2017'
label_dir = 'path/to/your/yolo_dataset/labels/train'

# 调用转换函数
coco_to_yolo(coco_json, image_dir, label_dir)