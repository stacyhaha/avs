import torch
import numpy as np
import cv2
from time import time
from ultralytics import YOLO
import matplotlib.pyplot as plt


class ObjectDetection:

    def __init__(self):

        # self.capture_index = capture_index
        self.device = 'mps'
        self.model = self.load_model()

    def load_model(self):
        model = YOLO('/Users/laiweizhi/PycharmProjects/5004Project/runs/train3/weights/best.pt')
        model.fuse()
        return model

    def predict(self, frame):
        results = self.model(frame)
        return results

    def control_car(self, results):
        # 邏輯大致可以像這樣
        # 默認行為
        action= '繼續行駛'
        stop_threshold =200

        for result in results:
            boxes = result.boxes
            name = result.names
            class_name = name[boxes.cls.item()]
            box = boxes.xyxy
            box_size = (box[2] - box[0]) * (box[3] - box[1])

            if class_name == 'red':  # 假设红灯的类别ID为2
                action = "停车"
            elif class_name == 'yellow':  # 假设黄灯的类别ID为1
                if box_size > stop_threshold:
                    action = "加速通过"
                else:
                    action = "减速并停车"
            elif class_name == 'green':  # 假设绿灯的类别ID为3
                action = "正常行驶"

        print(f"执行动作: {action}")
        return action

    def plot_bboxes(self, results, frame):

        def plot_bboxes(self, results, frame):
            # 画出所有检测到的框
            for result in results.xyxy[0]:
                cv2.rectangle(frame, (int(result[0]), int(result[1])), (int(result[2]), int(result[3])), (0, 255, 0), 2)
            cv2.imshow('frame', frame)
        # xyxys= []
        # confidences = []
        # class_id = []
        #
        # for result in results:
        #     boxes = result.boxes.cpu().numpy()
        #
        #     xyxys.append(result.xyxy)
        #     confidences.append(result.conf)
        #     class_id.append(result.cls)
        #
        #     # for xyxy in xyxys:
        #         # confidences.append(xyxy[4])
        #         # class_id.append(xyxy[5])
        # # cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0,255,0))
        #
        #
        #
        # return results[0].plot(), xyxys, confidences, class_id

    def __call__(self, input_data):
        results = self.predict(input_data)
        class_lst = []

        for result in results:
            if len(result.boxes) == 0:
                continue  # 如果没有检测到任何对象，跳过当前循环

            boxes = result.boxes
            names = result.names

            # 处理多个检测结果
            for i in range(len(boxes)):
                class_id = boxes.cls[i].item()  # 获取每个检测对象的类别ID
                class_name = names[class_id]  # 从类别ID获取类别名称
                class_lst.append(class_name)  # 将类别名称添加到列表

                print(f'detected sign is: {class_name}')
                print(f'class type: {type(class_name)}')

            # 假设 result.plot() 为可视化检测结果的函数
            processed_img = result.plot()
            print(type(processed_img))

            if processed_img is not None:
                processed_img = processed_img[:, :, [2, 1, 0]]  # 转换颜色通道顺序为RGB
                plt.imshow(processed_img)
                plt.axis('off')  # 关闭坐标轴
                plt.show()

        return class_lst


class Traffic_sign_detection:

    def __init__(self):

        # self.capture_index = capture_index
        self.device = 'mps'
        self.model = self.load_model()

    def load_model(self):
        model = YOLO('/Users/laiweizhi/PycharmProjects/5004Project/ultralytics-main/runs/detect/train14/weights/best.pt')
        model.fuse()
        return model

    def predict(self, frame):
        results = self.model(frame)
        return results

    def control_car(self, results):
        # 邏輯大致可以像這樣
        # 默認行為
        action= '繼續行駛'
        stop_threshold =200

        for result in results:
            if len(result) == 0:
                continue
            boxes = result.boxes
            name = result.names
            class_name = name[boxes.cls.item()]
            box = boxes.xyxy
            box_size = (box[2] - box[0]) * (box[3] - box[1])

            if class_name == 'red':  # 假设红灯的类别ID为2
                action = "停车"
            elif class_name == 'yellow':  # 假设黄灯的类别ID为1
                if box_size > stop_threshold:
                    action = "加速通过"
                else:
                    action = "减速并停车"
            elif class_name == 'green':  # 假设绿灯的类别ID为3
                action = "正常行驶"

        print(f"执行动作: {action}")
        return action

    def __call__(self, input_data):
        results = self.predict(input_data)
        class_lst = []

        for result in results:
            if len(result.boxes) == 0:
                continue  # 如果没有检测到任何对象，跳过当前循环

            boxes = result.boxes
            names = result.names

            # 处理多个检测结果
            for i in range(len(boxes)):
                class_id = boxes.cls[i].item()  # 获取每个检测对象的类别ID
                class_name = names[class_id]  # 从类别ID获取类别名称
                class_lst.append(class_name)  # 将类别名称添加到列表

                print(f'detected sign is: {class_name}')
                print(f'class type: {type(class_name)}')

            # 假设 result.plot() 为可视化检测结果的函数
            processed_img = result.plot()
            print(type(processed_img))

            if processed_img is not None:
                processed_img = processed_img[:, :, [2, 1, 0]]  # 转换颜色通道顺序为RGB
                plt.imshow(processed_img)
                plt.axis('off')  # 关闭坐标轴
                plt.show()

        return class_lst

if __name__ == '__main__':
    # predict = ObjectDetection()
    # output =predict('/Users/laiweizhi/Desktop/test_images')
    # print(f'class_lst is: {output}, type is: {type(output)}')
    predict = Traffic_sign_detection()
    output = predict('/Users/laiweizhi/Desktop/test_images')
    print(f'class_lst is: {output}, type is: {type(output)}')