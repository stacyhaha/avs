import os
from PIL import Image
import concurrent.futures
from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2

class ObjectDetection:
    def __init__(self, tf_checkpoint, ts_checkpoint):
        self.device = 'mps'
        self.tf_checkpoint = tf_checkpoint
        self.ts_checkpoint = ts_checkpoint
        self.model_tl = self.load_model(self.tf_checkpoint)
        self.model_ts = self.load_model(self.ts_checkpoint)

    def load_model(self, model_path):
        model = YOLO(model_path)
        model.fuse()
        return model

    def tl_predict(self, frame):
        results = self.model_tl(frame)
        return results

    def ts_predict(self, frame):
        results = self.model_ts(frame)
        return results

    def detect(self, input_data):
        tl_results = self.tl_predict(input_data)
        ts_results = self.ts_predict(input_data)
        speed = 0
        orientation = 0
        bias = 0


        detected_class = []

        for result in tl_results:
            if len(result.boxes) == 0:
                continue  # 如果没有检测到任何对象，跳过当前循环

            boxes = result.boxes
            names = result.names

            for i in range(len(boxes)):
                class_id = boxes.cls[i].item()  # 获取每个检测对象的类别ID
                class_name = names[class_id]  # 从类别ID获取类别名称
                box_size = boxes.xyxyn[i]  # 获取对应框的归一化坐标
                conf = boxes.conf[i]  # 获取对应框的置信分数

                xmin, ymin, xmax, ymax = box_size[0], box_size[1], box_size[2], box_size[3]
                width = xmax - xmin
                height = ymax - ymin
                area = width * height  # 计算框的面积比例

                # 检查面积比例是否超过1.1% 且置信分数大于0.8
                if area > 0.01 and conf > 0.8:
                    detected_class.append(class_name)  # 添加到检测类别列表

                # 输出信息，可选
                # print(f'Detection: {class_name} with confidence {conf:.2f} and area {area:.4f}')

            processed_img = result.plot(labels=False)
            # print(f'data type: {type(processed_img)}')


            # if processed_img is not None:
            #     processed_img = processed_img[:, :, [2, 1, 0]]  # 转换颜色通道顺序为RGB
            #     plt.imshow(processed_img)
            #     plt.axis('off')  # 关闭坐标轴
            #     plt.show()

        for result in ts_results:
            if len(result.boxes) == 0:
                continue  # 如果没有检测到任何对象，跳过当前循环

            boxes = result.boxes
            names = result.names

            for i in range(len(boxes)):
                class_id = boxes.cls[i].item()  # 获取每个检测对象的类别ID
                class_name = names[class_id]  # 从类别ID获取类别名称
                box_size = boxes.xyxyn[i]  # 获取对应框的归一化坐标
                conf = boxes.conf[i]  # 获取对应框的置信分数

                xmin, ymin, xmax, ymax = box_size[0], box_size[1], box_size[2], box_size[3]
                width = xmax - xmin
                height = ymax - ymin
                area = width * height  # 计算框的面积比例

                # 检查面积比例是否超过1.1% 且置信分数大于0.8
                if area > 0.01 and conf > 0.8:
                    detected_class.append(class_name)  # 添加到检测类别列表

                # 输出信息，可选
                # print(f'Detection: {class_name} with confidence {conf:.2f} and area {area:.4f}')


            processed_img = result.plot(labels=False)
            # print(f'data type: {type(processed_img)}')


            # if processed_img is not None:
            #     processed_img = processed_img[:, :, [2, 1, 0]]  # 转换颜色通道顺序为RGB
            #     plt.imshow(processed_img)
            #     plt.axis('off')  # 关闭坐标轴
            #     plt.show()

        # 设置优先级
        # red(減速再停車), yellow(減速), green(不進行動作)
        # go straight(不進行動作), limit 5 km(減速), parking place(執行停車), school area(減速並喇叭1秒), stop(停車), turn right(右轉), vehicle horn(喇叭1秒), zebra crossing(停車再行駛)
        # 前進 60,90,0 右轉30,90,-30 減速 30,90,0 停止 0,90,0
        priority = {
            'stop': 1,
            'red': 2,
            'zebra crossing': 3,
            'turn right': 4,
            'yellow': 5,
            'limit 5 km': 5,
            'school area': 6,
            'green': 7,
            'go straight': 8,
            'vehicle horn': 9,
            'parking place': 10  # 调整优先级，使其作用最小
        }

        # 初始化参数
        speed = 60  # 默认速度
        orientation = 90  # 默认方向
        bias = 0  # 默认无偏差
        horn = False  # 是否按喇叭

        # 检查每个类别，并根据优先级更新行驶参数
        current_priority = float('inf')  # 当前的优先级设置为无限高

        for category in detected_class:
            cat_priority = priority.get(category, float('inf'))  # 获取类别的优先级

            if cat_priority < current_priority:
                if category in ['stop', 'red', 'zebra crossing']:
                    speed = 0
                    bias = 0
                    horn = 'zebra crossing' in detected_class or 'school area' in detected_class

                elif category in ['turn right']:
                    speed = 30
                    orientation = 90
                    bias = -30

                elif category in ['yellow', 'limit 5 km', 'school area']:
                    speed = 30
                    orientation = 90
                    bias = 0
                    horn = 'school area' in detected_class

                elif category in ['vehicle horn']:
                    horn = True

                current_priority = cat_priority  # 更新当前处理的优先级

        # 输出最终决定的行驶参数
        # print(f'Speed: {speed}, Orientation: {orientation}, Bias: {bias}')
        # if horn:
            # print("Horn: On for 1 second")

        return detected_class, (speed, orientation, bias), processed_img


def process_file(file_path, model):
    valid_extensions = {'jpeg', 'jpg', 'dng', 'bmp', 'mpo', 'webp', 'pfm', 'png', 'tiff', 'tif', 'mpg', 'avi', 'mp4',
                        'wmv', 'mpeg', 'asf', 'webm', 'gif', 'ts', 'mkv', 'm4v', 'mov'}
    ext = file_path.split('.')[-1].lower()
    if ext in valid_extensions:
        predict, action, processed_img = model.detect(file_path)
        print(predict)
        print(action)

if __name__ == '__main__':
    tf_checkpoint = 'object_detection/object_detection_part/traffic_light/weights/best.pt'
    ts_checkpoint = 'object_detection/object_detection_part/traffic_signs/weights/best.pt'
    model = ObjectDetection(tf_checkpoint, ts_checkpoint)
    # 检查这是一个文件还是目录
    path = 'object_detection/object_detection_part/traffic_light_images/TL_test1.jpeg'
    imge = Image.open(path)
    model.detect(imge)
