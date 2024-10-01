import os
import json

from tqdm import tqdm

# 映射表
coco_to_yolo_index = {
    1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9,
    11: 10, 13: 11, 14: 12, 15: 13, 16: 14, 17: 15, 18: 16, 19: 17, 20: 18, 21: 19,
    22: 20, 23: 21, 24: 22, 25: 23, 27: 24, 28: 25, 31: 26, 32: 27, 33: 28, 34: 29,
    35: 30, 36: 31, 37: 32, 38: 33, 39: 34, 40: 35, 41: 36, 42: 37, 43: 38, 44: 39,
    46: 40, 47: 41, 48: 42, 49: 43, 50: 44, 51: 45, 52: 46, 53: 47, 54: 48, 55: 49,
    56: 50, 57: 51, 58: 52, 59: 53, 60: 54, 61: 55, 62: 56, 63: 57, 64: 58, 65: 59,
    67: 60, 70: 61, 72: 62, 73: 63, 74: 64, 75: 65, 76: 66, 77: 67, 78: 68, 79: 69,
    80: 70, 81: 71, 82: 72, 84: 73, 85: 74, 86: 75, 87: 76, 88: 77, 89: 78, 90: 79
}

def coco_to_yolo(coco_annotation_file, output_dir):
    # 创建 labels 输出文件夹
    labels_output_dir = os.path.join(output_dir, 'val')  # 修改
    os.makedirs(labels_output_dir, exist_ok=True)

    # 读取 COCO 标注文件
    with open(coco_annotation_file, 'r') as f:
        coco_data = json.load(f)

    # 获取图像信息
    images = {image['id']: image for image in coco_data['images']}

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

        # 只处理存在映射的类别，跳过无效的类别
        if category_id in coco_to_yolo_index:
            # 获取 YOLO 格式的类别索引
            yolo_category_id = coco_to_yolo_index[category_id]

            # 转换 bbox 坐标为 YOLO 格式: [class_id, x_center, y_center, width, height]
            yolo_bbox = convert_bbox((image_width, image_height), bbox)

            # 创建 YOLO 格式的标签文件 (.txt)
            label_file_name = os.path.splitext(image_file_name)[0] + '.txt'
            label_file_path = os.path.join(labels_output_dir, label_file_name)

            # 写入标签文件
            with open(label_file_path, 'a') as label_file:
                label_file.write(f"{yolo_category_id} " + " ".join(map(str, yolo_bbox)) + '\n')


if __name__ == '__main__':
    # 配置 COCO 数据集路径
    coco_annotation_file = r'D:\aWorkHome\project\datasets\yolov8_cocotest\labels\instances_val2017.json'  # COCO 标注文件
    output_dir = r'D:\aWorkHome\project\datasets\yolov8_cocotest\labels'  # YOLO 格式的输出路径

    coco_to_yolo(coco_annotation_file, output_dir)