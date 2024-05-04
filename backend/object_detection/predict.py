import os

from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2

class ObjectDetection:
    def __init__(self):
        self.device = 'mps'
        self.tf_checkpoint = '/Users/laiweizhi/PycharmProjects/5004Project/runs/train3/weights/best.pt'
        self.ts_checkpoint = '/Users/laiweizhi/PycharmProjects/5004Project/ultralytics-main/runs/detect/train14/weights/best.pt'
        # 導入traffic_light(tl) and traffic_sign(ts)的權重
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

    def __call__(self, input_data):
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
                detected_class.append(class_name)  # 将类别名称添加到列表

                print(f'tl detection: {detected_class}')

                # print(f'detected class is: {class_name}')
                # print(f'class type: {type(class_name)}')

            # processed_img = result.plot()
            # print(type(processed_img))
            #
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
                detected_class.append(class_name)  # 将类别名称添加到列表

                print(f'ts detection: {detected_class}')


                # print(f'detected class is: {class_name}')
                # print(f'class type: {type(class_name)}')

            # processed_img = result.plot()
            # # print(type(processed_img))
            #
            # if processed_img is not None:
            #     processed_img = processed_img[:, :, [2, 1, 0]]  # 转换颜色通道顺序为RGB
            #     plt.imshow(processed_img)
            #     plt.axis('off')  # 关闭坐标轴
            #     plt.show()

        # print(f'class_lst is: {detected_class}, type is: {type(detected_class)}')
        if 'red' in detected_class or 'stop' in detected_class:
            speed = 0
            orientation = 90
            bias = 0
        # else:
        #     self.speed = 0
        #     self.orientation = 0
        #     self.bias = 0

        return detected_class, (speed, orientation, bias)

# red(減速再停車), yellow(減速), green(不進行動作)
# go straight(不進行動作), limit 5 km(減速), parking place(執行停車), school area(減速並喇叭1秒), stop(停車), turn right(右轉), vehicle horn(喇叭1秒), zebra crossing(停車再行駛)

if __name__ == '__main__':
    model = ObjectDetection()
    valid_extensions = {'jpeg', 'jpg', 'dng', 'bmp', 'mpo', 'webp', 'pfm', 'png', 'tiff', 'tif', 'mpg', 'avi', 'mp4',
                        'wmv', 'mpeg', 'asf', 'webm', 'gif', 'ts', 'mkv', 'm4v', 'mov'}
    directory = '/Users/laiweizhi/Desktop/object_detection_part/ts_test'
    count = 1

    for filename in os.listdir(directory):
        ext = filename.split('.')[-1].lower()
        if ext in valid_extensions:
            file_path = os.path.join(directory, filename)
            predict, action = model(file_path)
            # print(predict)
            print(action)
            print(count)
            # count+=1
