import os

def process_files_in_folder(folder_path):
    # 获取文件夹中的所有文件
    file_list = os.listdir(folder_path)

    for file_name in file_list:
        # 检查文件扩展名是否为txt
        if file_name.endswith('.txt'):
            file_path = os.path.join(folder_path, file_name)

            # 读取文件内容
            with open(file_path, 'r') as file:
                lines = file.readlines()

            # 处理文件中的负数
            processed_lines = []
            for line in lines:
                numbers = line.split()
                processed_numbers = []
                for number in numbers:
                    try:
                        number = float(number)
                        if number == 0:
                            number = int(number)
                        if number < 0:
                            number = abs(number)  # 将负数转换为正数
                    except ValueError:
                        pass
                    processed_numbers.append(str(number))

                processed_line = ' '.join(processed_numbers)
                processed_lines.append(processed_line)

            # 将处理后的内容写回文件
            with open(file_path, 'w') as file:
                file.write('\n'.join(processed_lines))


# 指定文件夹路径
folder_path = r'D:\aWorkHome\project\datasets\yolov8_cocotest\labels\val'  # 只需修改成你的txt路径
process_files_in_folder(folder_path)