#!/usr/bin/env python3
import sys
import cv2
import time
import math
import rospy
import queue
import signal
import hiwonder
import apriltag
import threading
import numpy as np
import mediapipe as mp
from sensor_msgs.msg import Image
from PyQt5.QtCore import Qt, QPoint, QRect
from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5.QtGui import QPainter, QPixmap, QColor, QPen, QFont

# 巡线分拣

ROS_NODE_NAME = "patrol_sorting"

TAG_SIZE = 33.30
# 实例化
Misc = hiwonder.misc
chassis = hiwonder.MecanumChassis()
motor = hiwonder.motor.EncoderMotorController(1, 3)
at_detector = apriltag.Detector(apriltag.DetectorOptions(families='tag36h11'))

# 放置坐标
TARGET_POSITIONS = {
    1: (-100, -180, 0),
    2: (0, -180, 0),
    3: (100, -180, 0)}

TRACKING_POSITION = (0, -120, 120)


class TrackingSorting:
    # 初始化
    def __init__(self):
        self.speed = 30
        self.count = 0
        self.skip = 'line'
        self.tag_id = None
        self.centerX = 320
        self.st = False
        self.line_state = True
        self.sorting_state = False
        self.moving_block = None
        self.line_centerx = -1
        self.line_centery = -1
        self.line_breadth = 0
        self.camera_params = None
        self.pid_x = hiwonder.PID(0.18, 0.001, 0.02)

    # 加载相机内参
    def load_camera_params(self):
        self.camera_params = rospy.get_param('/camera_cal/block_params', self.camera_params)
        if self.camera_params is not None:
            self.K = np.array(self.camera_params['K'], dtype=np.float64).reshape(3, 3)
            self.R = np.array(self.camera_params['R'], dtype=np.float64).reshape(3, 1)
            self.T = np.array(self.camera_params['T'], dtype=np.float64).reshape(3, 1)


# 机器人跟踪线程
def move():
    rospy.sleep(8)
    while True:
        # 假设image是实时从摄像头获取的
        image = get_camera_image()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary_image = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
        edges = find_edges(binary_image)
        left_boundaries, right_boundaries = find_path_boundaries(edges)

        if np.any(left_boundaries != -1) and np.any(right_boundaries != -1):  # 确保有有效的边界
            adjust_robot_movement(left_boundaries, right_boundaries)
        else:
            # 如果没有有效的边界，停止机器人
            motor.set_speed(0, 1)
            motor.set_speed(0, 2)
            motor.set_speed(0, 3)
            motor.set_speed(0, 4)

        rospy.sleep(0.01)


# 作为子线程开启
th = threading.Thread(target=move)
th.setDaemon(True)
th.start()


# 画面像素坐标转换现实坐标函数
def camera_to_world(cam_mtx, r, t, img_points):
    inv_k = np.asmatrix(cam_mtx).I
    r_mat = np.zeros((3, 3), dtype=np.float64)
    cv2.Rodrigues(r, r_mat)
    # invR * T
    inv_r = np.asmatrix(r_mat).I  # 3*3
    transPlaneToCam = np.dot(inv_r, np.asmatrix(t))  # 3*3 dot 3*1 = 3*1
    world_pt = []
    coords = np.zeros((3, 1), dtype=np.float64)
    for img_pt in img_points:
        coords[0][0] = img_pt[0][0]
        coords[1][0] = img_pt[0][1]
        coords[2][0] = 1.0
        worldPtCam = np.dot(inv_k, coords)  # 3*3 dot 3*1 = 3*1
        # [x,y,1] * invRp
        scale = transPlaneToCam[2][0] / worldPtPlane[2][0]
        # zc * [x,y,1] * invR
        scale_worldPtPlane = np.multiply(scale, worldPtPlane)
        # [X,Y,Z]=zc*[x,y,1]*invR - invR*T
        worldPtPlaneReproject = np.asmatrix(scale_worldPtPlane) - np.asmatrix(transPlaneToCam)  # 3*1 dot 1*3 = 3*3
        pt = np.zeros((3, 1), dtype=np.float64)
        pt[0][0] = worldPtPlaneReproject[0][0]
        pt[1][0] = worldPtPlaneReproject[1][0]
        pt[2][0] = 0
        world_pt.append(pt.T.tolist())
    return world_pt


def rotation_mtx_to_euler(R):
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6
    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0
    return np.array([x, y, z])


# apriltag检测函数
def Apriltag(img):
    frame_gray = cv2.cvtColor(np.copy(img), cv2.COLOR_RGB2GRAY)
    tags = at_detector.detect(frame_gray)
    for tag in tags:
        corners = tag.corners.reshape(1, -1, 2).astype(int)
        center = tag.center.astype(int)
        cv2.drawContours(img, corners, -1, (255, 0, 0), 3)
        cv2.circle(img, tuple(center), 5, (255, 255, 0), 10)
        cv2.putText(img, "id:%d" % tag.tag_id, (center[0], center[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    if len(tags) > 0:
        if state.moving_block is None:
            state.moving_block = tags[0]
        else:
            new_tag = tags[0]
            if new_tag.tag_id != state.moving_block.tag_id:
                state.count = 0
                state.st = False
            else:
                state.count += 1
                if state.count > 20:
                    state.count = 0
                    if not state.sorting_state and not state.st:
                        state.st = True
                        state.tag_id = new_tag.tag_id

            state.moving_block = tags[0]
    else:
        state.count = 0
        state.st = False
        if state.moving_block is not None:
            state.moving_block = None
    img_h, img_w = img.shape[:2]
    cv2.line(img, (int(img_w / 2 - 10), int(img_h / 2)), (int(img_w / 2 + 10), int(img_h / 2)), (0, 255, 255), 2)
    cv2.line(img, (int(img_w / 2), int(img_h / 2 - 10)), (int(img_w / 2), int(img_h / 2 + 10)), (0, 255, 255), 2)
    return img


# 找出面积最大的轮廓
# 参数为要比较的轮廓的列表
def getAreaMaxContour(contours, area_min=10):
    contour_area_temp = 0
    contour_area_max = 0
    area_max_contour = None
    for c in contours:  # 历遍所有轮廓
        contour_area_temp = math.fabs(cv2.contourArea(c))  # 计算轮廓面积
        if contour_area_temp > contour_area_max:
            contour_area_max = contour_area_temp
            if contour_area_temp >= area_min:  # 只有在面积大于设定值时，最大面积的轮廓才是有效的，以过滤干扰
                area_max_contour = c
    return area_max_contour, contour_area_max  # 返回最大的轮廓


size = (640, 480)
roi = [(220, 260, 0, 640, 0.1),
       (280, 320, 0, 640, 0.4),
       (340, 420, 0, 640, 0.5)]
roi_h1 = roi[0][0]
roi_h2 = roi[1][0] - roi[0][0]
roi_h3 = roi[2][0] - roi[1][0]
roi_h_list = [roi_h1, roi_h2, roi_h3]


# 巡线视觉处理函数
def line_patrol(img_draw, color='red'):
    n = 0
    center_ = []
    weight_sum = 0
    centroid_x_sum = 0
    img_h, img_w = img_draw.shape[:2]
    frame_resize = cv2.resize(img_draw, size, interpolation=cv2.INTER_NEAREST)
    frame_gb = cv2.GaussianBlur(frame_resize, (3, 3), 3)

    # 将图像分割成上中下三个部分，这样处理速度会更快，更精确
    for r in roi:
        area_max = 0
        areaMaxContour = 0
        roi_h = roi_h_list[n]
        n += 1
        blobs = frame_gb[r[0]:r[1], r[2]:r[3]]
        frame_lab = cv2.cvtColor(blobs, cv2.COLOR_RGB2LAB)  # 将图像转换到LAB空间
        frame_mask = cv2.inRange(frame_lab, tuple(target_colors[color]['min']),
                                 tuple(target_colors[color]['max']))  # 对原图像和掩模进行位运算
        opened = cv2.morphologyEx(frame_mask, cv2.MORPH_OPEN, np.ones((6, 6), np.uint8))  # 开运算
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, np.ones((6, 6), np.uint8))  # 闭运算
        cnts = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)[-2]  # 找出所有轮廓
        cnt_large, area = getAreaMaxContour(cnts)  # 找到最大面积的轮廓
        if cnt_large is not None:  # 如果轮廓不为空
            rect = cv2.minAreaRect(cnt_large)  # 最小外接矩形
            box = np.int0(cv2.boxPoints(rect))  # 最小外接矩形的四个顶点
            for i in range(4):
                box[i, 1] = box[i, 1] + (n - 1) * roi_h + roi[0][0]
                box[i, 1] = int(Misc.val_map(box[i, 1], 0, size[1], 0, img_h))
            for i in range(4):
                box[i, 0] = int(Misc.val_map(box[i, 0], 0, size[0], 0, img_w))

            cv2.drawContours(img_draw, [box], -1, (0, 0, 255, 255), 2)  # 画出四个点组成的矩形
            # 获取矩形的对角点
            pt1_x, pt1_y = box[0, 0], box[0, 1]
            pt3_x, pt3_y = box[2, 0], box[2, 1]
            state.line_breadth = abs(pt3_x - pt1_x)
            center_x, center_y = (pt1_x + pt3_x) / 2, (pt1_y + pt3_y) / 2  # 中心点
            cv2.circle(img_draw, (int(center_x), int(center_y)), 5, (0, 0, 255), -1)  # 画出中心点
            center_.append([center_x, center_y])
            # 按权重不同对上中下三个中心点进行求和
            centroid_x_sum += center_x * r[4]
            weight_sum += r[4]

    if weight_sum is not 0:
        state.line_centery = int(center_y)
        state.line_centerx = int(centroid_x_sum / weight_sum)  # 求最终得到的中心点
        cv2.circle(img_draw, (state.line_centerx, int(center_y)), 10, (0, 255, 255), -1)  # 画出中心

    else:
        state.line_centerx = -1
        state.line_centery = -1

    return img_draw


def image_proc():
    image = image_queue.get(block=True)
    image = np.copy(image)
    if state.skip == 'line':
        # 使用新的中线检测处理，只处理图像下半3/5部分
        image = process_and_draw_center_line(image)
    # 保持其他处理不变
    image = cv2.resize(image, size)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imshow(ROS_NODE_NAME, image)
    cv2.waitKey(1)


def process_and_draw_center_line(image):
    height = image.shape[0]
    crop_start = int(height * 2 / 5)  # 开始裁剪的位置，图像的下半3/5部分
    cropped_image = image[crop_start:]  # 裁剪图像
    gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
    edges = cv2.Canny(binary_image, 50, 150)
    left_boundaries, right_boundaries = find_path_boundaries(edges)
    result_img = draw_center_line(cropped_image, left_boundaries, right_boundaries)
    image[crop_start:] = result_img  # 将处理后的部分替换回原图
    return image


def image_callback(ros_image):
    try:
        image = np.ndarray(shape=(ros_image.height, ros_image.width, 3), dtype=np.uint8, buffer=ros_image.data)
        if state.skip == 'line':
            image = cv2.flip(image, 1)
        image_queue.put_nowait(image)

    except:
        pass


def shutdown(signum, frame):
    print('shutdown')
    rospy.sleep(0.5)
    motor.set_speed(0, 1)
    motor.set_speed(0, 2)
    motor.set_speed(0, 3)
    motor.set_speed(0, 4)
    sys.exit()


signal.signal(signal.SIGINT, shutdown)

if __name__ == '__main__':
    rospy.init_node(ROS_NODE_NAME, log_level=rospy.DEBUG)
    rospy.sleep(0.2)
    state = TrackingSorting()
    state.load_camera_params()
    image_queue = queue.Queue(maxsize=1)

    jetmax = hiwonder.JetMax()
    sucker = hiwonder.Sucker()
    jetmax.go_home(2, 3)
    rospy.sleep(2)
    jetmax.set_position(TRACKING_POSITION, 1)
    target_colors = rospy.get_param('/lab_config_manager/color_range_list', {})
    del [target_colors['white']]
    image_sub = rospy.Subscriber('/usb_cam/image_rect_color', Image, image_callback, queue_size=1)

    while True:
        try:
            image_proc()
        except Exception as e:
            print(e)
            sys.exit()
