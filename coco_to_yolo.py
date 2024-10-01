import os
import json

from tqdm import tqdm


def coco_to_yolo(coco_annotation_file, output_dir):
    # 创建 labels 输出文件夹
    labels_output_dir = os.path.join(output_dir, 'labels') # 修改
    os.makedirs(labels_output_dir, exist_ok=True)

    # 读取 COCO 标注文件
    with open(coco_annotation_file, 'r') as f:
        coco_data = json.load(f)

    # 获取图像信息和类别信息
    images = {image['id']: image for image in coco_data['images']}
    categories = {category['id']: category['name'] for category in coco_data['categories']}

    # 用于归一化标注框坐标的函数
    def convert_bbox(size, box):
        dw = 1. / size[0]
        dh = 1. / size[1]
        x = (box[0] + box[2]) / 2.0  # x_center
        y = (box[1] + box[3]) / 2.0  # y_center
        w = box[2] - box[0]  # width
        h = box[3] - box[1]  # height
        return (x * dw, y * dh, w * dw, h * dh)

    # 遍历标注文件中的每个目标
    for annotation in tqdm(coco_data['annotations']):
        image_id = annotation['image_id']
        category_id = annotation['category_id']
        bbox = annotation['bbox']  # COCO 格式的 bbox: [x_min, y_min, width, height]

        # 获取图像的文件名和尺寸
        image_info = images[image_id]
        image_file_name = image_info['file_name']
        image_width = image_info['width']
        image_height = image_info['height']

        # 转换 bbox 坐标为 YOLO 格式: [class_id, x_center, y_center, width, height]
        yolo_bbox = convert_bbox((image_width, image_height), bbox)

        # 创建 YOLO 格式的标签文件 (.txt)
        label_file_name = os.path.splitext(image_file_name)[0] + '.txt'
        label_file_path = os.path.join(labels_output_dir, label_file_name)

        # 写入标签文件
        with open(label_file_path, 'a') as label_file:
            label_file.write(f"{category_id - 1} " + " ".join(map(str, yolo_bbox)) + '\n')


if __name__ == '__main__':
    # 配置 COCO 数据集路径
    coco_annotation_file = '/path/to/coco/annotations/instances_train2017.json'  # COCO 标注文件
    output_dir = '/path/to/output/dataset'  # YOLO 格式的输出路径

    coco_to_yolo(coco_annotation_file, output_dir)